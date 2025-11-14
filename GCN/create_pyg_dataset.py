import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm

from constants import ALL_ATOM_LABELS, DICT_AA_SELECTION, SELECTED_ATOMS
from utils import is_connected, extract_pqr_data, build_edges_blockwise_less_atoms, get_atom_indices_type_aa_type_and_coords, build_edges_knn_radius_combined, build_edges_blockwise
from utils import project_to_unit_sphere, compute_spherical_harmonics

'''
    in: EnzyBase12k or other database of cif files + pqr files
    out: torch geometric Dataset
    
    You need to provide train.csv in root_dir_train and test.csv in root_dir_test
    (subsets of EnzyBase12k_metadata.csv)
    training dataset is later split into training and validation in the training scripts

'''

#%%

class EnzymeDataset(Dataset):
    def __init__(self, root, filename, y_mean, y_std, test=False, transform=None, pre_transform=None,
                 norm_X=False, norm_y=False, l_max=-1, edges_knn=False, less_aa=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.norm_X = norm_X
        self.norm_y = norm_y
        self.l_max = l_max
        self.edges_knn = edges_knn
        self.less_aa = less_aa
        self.y_mean = y_mean
        self.y_std = y_std
        super(EnzymeDataset, self).__init__(root, transform, pre_transform)
    
    # returns name of our dataset
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename
    
    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        #print(self.raw_paths)
        self.data = pd.read_csv(self.raw_paths[0])#.reset_index()
        #print(self.data)
        
        if "index" in self.data.columns:
            self.data.index = self.data["index"]
        else:
            self.data.index = range(len(self.data))
        
        if self.test:
            return [f'data_test_{i}.pt' for i in self.data.index]
        else:
            return [f'data_{i}.pt' for i in self.data.index]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        connected_count = 0
        disconnected_count = 0
        
        for index, row in tqdm(self.data.iterrows(), total=len(self.data)):
            data_path  = os.path.join(self.processed_dir, f'data_test_{index}.pt' if self.test else f'data_{index}.pt')
            if os.path.exists(data_path):
                continue
            
            uniprot_id = row['uniprot_id']
            cif_path = os.path.join(cif_dir, f'{uniprot_id}.cif')
            pqr_path = os.path.join(pqr_dir, f'{uniprot_id}.pqr')
            
            # Get labels info
            
            label = self._get_labels(row['ph_optimum'])
            if torch.isnan(label).any():
                print(f'No pH value for uniprot id {uniprot_id} in csv available.')
                continue
            
            if self.norm_y:
                label = (label - self.y_mean) / self.y_std
            
            
            node_features, edge_index = self.get_node_features_and_edge_index(pqr_path)
            
            # check for graph connectivity
            if is_connected(edge_index):
                connected_count += 1
            else:
                disconnected_count += 1
            
            data = Data(x=node_features, edge_index=edge_index, y=label, uniprot_id=uniprot_id)
            torch.save(data, data_path)
            
        print(f"Number of connected graphs: {connected_count}")
        print(f"Number of unconnected graphs: {disconnected_count}")
    
    def get_node_features_and_edge_index(self, pqr_path):
        uniprot_id = pqr_path.split('/')[-1]
        
        '''
            Incorporate x y z coordinates of all atoms
        '''
        # get indices of ATOM entries because sometimes an ATOM is missing
        # idx_type_coord is list with entries (4453, 'OE1', 800, 'GLN', (-28.593, 76.052, 71.292))

        try:
            atom_entries = extract_pqr_data(pqr_path, self.less_aa)
        except ValueError as e:
            print(e)
            print(f'for uniprot id {uniprot_id}')
        
        positions, atom_labels, charges, edge_index = [], [], [], []
        
        for dic in atom_entries:
            positions.append([dic['x'], dic['y'], dic['z']])
            atom_labels.append(dic['atom_label'])
            charges.append(dic['charge'])
        
        
        one_hot_labels = [[1 if label == l else 0 for l in ALL_ATOM_LABELS] for label in atom_labels]
        
        charges = torch.tensor(charges, dtype=torch.float32).unsqueeze(dim=1)
        one_hot_labels = torch.tensor(one_hot_labels, dtype=torch.float32)
        
        '''
            incorporate edges of protein structure (chemical bonds)
            
            # Beispiel: Radius-Graph
            edge_index = radius_graph(torch.tensor(positions), r=5.0)
            
            # Beispiel: kNN-Graph (k=6)
            edge_index = knn_graph(torch.tensor(positions), k=6)
        '''
        if self.less_aa:
            edge_index = build_edges_blockwise_less_atoms(atom_entries)
        elif self.edges_knn:
            edge_index = build_edges_knn_radius_combined(atom_entries, k=4, r=3, max_edges_per_node=3)
        else:
            edge_index = build_edges_blockwise(atom_entries)
        
        if self.norm_X:
            positions = np.array(positions, dtype=np.float32)
            positions = project_to_unit_sphere(positions)
        
        
        if self.l_max != -1:
            Y = compute_spherical_harmonics(self.l_max, positions)
            node_features = torch.tensor(np.hstack([Y, charges, one_hot_labels]), dtype=torch.float)
        else:
            node_features = torch.tensor(np.hstack([positions, charges, one_hot_labels]), dtype=torch.float)

        
        return node_features, edge_index
    
    def get_node_features_and_edge_index_surface(self, cif_file):
        # idx_type_coord is list with entries (4453, 'OE1', 800, 'GLN', (-28.593, 76.052, 71.292))
        _, idx_type_coord, _, _ = get_atom_indices_type_aa_type_and_coords(cif_file)
        
        positions, atom_labels, aa_labels = [], [], []
        idx_type_coord_grouped = {}
    
        for entry in idx_type_coord:
            idx_res = entry[2]
            idx_type_coord_grouped.setdefault(idx_res, []).append(list(entry))
        
        positions, atom_labels, aa_labels = zip(*[
            (entry[4], entry[1], entry[3]) 
            for value in idx_type_coord_grouped.values() 
            for entry in value 
            if entry[1] == 'CB' or entry[1] in DICT_AA_SELECTION[entry[3]]
        ])
        positions, atom_labels, aa_labels = list(positions), list(atom_labels), list(aa_labels)

        
        if self.norm_X:
            positions = np.array(positions, dtype=np.float32)
            positions = project_to_unit_sphere(positions)
        
        
        one_hot_labels = np.zeros((len(atom_labels), len(SELECTED_ATOMS)))
        for i, label in enumerate(atom_labels):
            one_hot_labels[i][SELECTED_ATOMS.index(label)] = 1
        
        edge_index = np.column_stack([np.arange(len(positions) - 1), np.arange(1, len(positions))])
        edge_index = np.empty((2, 0), dtype=int)
        
        if self.l_max != -1:
            Y = compute_spherical_harmonics(self.l_max, positions)
            node_features = torch.tensor(np.hstack([Y, one_hot_labels]), dtype=torch.float)
        else:
            node_features = torch.tensor(np.hstack([positions, one_hot_labels]), dtype=torch.float)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
        
        return node_features, edge_index
    
    def _get_adjacency_info(self, mol):
        """
        one could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'),
                              weights_only=False)
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'),
                              weights_only=False)
        return data
    
    def subtract_one_from_edge_index(self):
        """ Subtract 1 from the edge_index for all data points in the dataset. """
        for i in tqdm(range(len(self))):
            data = self.get(i)
            data.edge_index = data.edge_index - 1
            torch.save(data, os.path.join(self.processed_dir, f'data_test_{i}.pt' if self.test else f'data_{i}.pt'))
            
    def remove_last_edge(self):
        for i in tqdm(range(len(self))):
            data = self.get(i)
            data.edge_index = data.edge_index[:, :-1]
            torch.save(data, os.path.join(self.processed_dir, f'data_test_{i}.pt' if self.test else f'data_{i}.pt'))
            
    def remove_edges(self):
        for i in tqdm(range(len(self))):
            data = self.get(i)
            data.edge_index = torch.empty((2, 0), dtype=torch.long).contiguous()
            torch.save(data, os.path.join(self.processed_dir, f'data_test_{i}.pt' if self.test else f'data_{i}.pt'))
            
    def reduce_nodes(self, reduction_percentage=0.1):
        """
        reduces the number of nodes in each graph to the given percentage
        
        Args:
        - reduction_percentage (float): percentage of nodes to keep
        """
        for i in tqdm(range(len(self))):
            data = self.get(i)
            
            num_nodes = data.x.size(0)
            num_nodes_to_keep = max(1, int(num_nodes * reduction_percentage))
            
            # random choice of nodes to keep
            keep_indices = torch.randperm(num_nodes)[:num_nodes_to_keep]
            keep_indices = torch.sort(keep_indices).values
            
            # create new node features and edge indices
            new_x = data.x[keep_indices]
            
            data.x = new_x
            
            torch.save(data, os.path.join(self.processed_dir, f'data_test_{i}.pt' if self.test else f'data_{i}.pt'))
    
    def check_graph_connectivity(self):
        connected_count = 0
        total_graphs = len(self.processed_file_names)

        for file_name in tqdm(self.processed_file_names):
            data_path = os.path.join(self.processed_dir, file_name)
            data = torch.load(data_path)
            edge_index = data.edge_index

            if is_connected(edge_index):
                connected_count += 1

        print(f'Connected graphs: {connected_count}/{total_graphs}')
    

if __name__ == '__main__':
    #%% set parameters/paths

    ### set path to cif files and pqr files

    cif_dir = input("Path to CIF files: ").strip()
    pqr_dir = input("Path to pqr files: ").strip()
    root_dir_train = input("Target path for training data (e.g. ./pyg_datasets/train); in /pyg_datasets/train/raw/ a metadata_train.csv file must be provided: ").strip()
    root_dir_test = input("Target path for test data  (e.g. ./pyg_datasets/test); in /pyg_datasets/test/raw/ a metadata_test.csv file must be provided: ").strip()

    os.makedirs(root_dir_train, exist_ok=True)
    os.makedirs(root_dir_test, exist_ok=True)

    train_file_name = 'train.csv'
    test_file_name = 'test.csv'

    set_norm_X = input("Norm_x ... project coordinates to unit sphere (True/False): ").strip().lower() == 'true'
    set_norm_y = input("Norm_y ... z-standardised pH labels (True/False): ").strip().lower() == 'true'
    set_l_max = int(input("l_max (-1 = xyz, >=0 replaces coordinates by spherical harmonics): ").strip())
    set_less_aa = input("Use only 13 amino acid and selected atoms (True/False): ").strip().lower() == 'true'

    # not used anymore, could be compromised
    set_edges_knn=False
    
    #%%
    train_file_path = os.path.join(root_dir_train, 'raw', train_file_name)
    test_file_path = os.path.join(root_dir_test, 'raw', test_file_name)
    
    df_train_dataset = pd.read_csv(train_file_path)
    df_test_dataset = pd.read_csv(test_file_path)

    pH_labels_train = np.asarray(df_train_dataset['ph_optimum'].to_list())
    pH_labels_test = np.asarray(df_test_dataset['ph_optimum'].to_list())
    
    pH_labels_all = np.concatenate([pH_labels_train, pH_labels_test], axis = 0)
    y_mean, y_std = pH_labels_all.mean(), pH_labels_all.std()
    
    print('y_mean is', y_mean)
    print('y_std is', y_std)
    
    train_dataset = EnzymeDataset(root=root_dir_train, filename=train_file_name, y_mean=y_mean, y_std=y_std, test=False, norm_X=set_norm_X, 
                                                  norm_y=set_norm_y, l_max=set_l_max, edges_knn=set_edges_knn, less_aa=set_less_aa)
    
    test_dataset = EnzymeDataset(root=root_dir_test, filename=test_file_name, y_mean=y_mean, y_std=y_std, test=True, norm_X=set_norm_X, 
                                                 norm_y=set_norm_y, l_max=set_l_max, edges_knn=set_edges_knn, less_aa=set_less_aa)
    