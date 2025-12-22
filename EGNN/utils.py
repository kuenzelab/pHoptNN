#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.nn import knn_graph, radius_graph
from e3nn.o3 import spherical_harmonics

from Bio.PDB.MMCIFParser import MMCIFParser

from constants import ALL_ATOM_LABELS, ALL_LABELS_BACKBONE, DICT_AA_SELECTION, DICT_AA_RESIDUE_EDGES

AA_MAP_EXTENDED_SIMPLE = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
     'MSE': 'X', 'FME': 'X', 'FT6': 'X', '4AF': 'X', 'PCA': 'X',
     'OCS': 'X', 'SEC': 'X', 'CSD': 'X', 'PHI': 'X', 'IAS': 'X',
     'R1A': 'X', 'BIF': 'X', 'ACE': 'X', 'P9S': 'X', 'END': 'X',
     'CME': 'X', 'CGU': 'X', 'UNK': 'X'}

def project_to_unit_sphere(positions):
    positions = np.asarray(positions, dtype=np.float32)
    centered_positions = positions - np.mean(positions, axis=0)
    radii = np.linalg.norm(centered_positions, axis=1)
    radii[radii == 0] = 1e-12
    unit_positions = centered_positions / radii[:, None]
    return np.asarray(unit_positions, dtype=np.float32) 

def compute_spherical_harmonics(l_max, unit_positions):
    l_values = list(range(l_max + 1))
    unit_positions_tensor = torch.tensor(unit_positions, dtype=torch.float32)
    Y = spherical_harmonics(l_values, unit_positions_tensor, normalize=False)
    return Y

def cif_parser_ATOM_HETATM_noncanonical(cif_file):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_file)

    chain_atom_data = {}
    chain_ids = set()

    model = structure[0]
    for chain in model:
        chain_id = chain.id
        chain_ids.add(chain_id)

        if chain_id not in chain_atom_data:
            chain_atom_data[chain_id] = []

        for residue in chain:
            comp_id = residue.resname

            if comp_id not in AA_MAP_EXTENDED_SIMPLE:
                continue

            seq_id = residue.id[1]

            for atom in residue:
                atom_entry = {
                    "atom_number": atom.serial_number if hasattr(atom, 'serial_number') else None,
                    "atom_symbol": atom.element,
                    "atom_symbol_long": atom.name,
                    "residue_name": comp_id,
                    "residue_number": seq_id,
                    "coord_x": atom.coord[0],
                    "coord_y": atom.coord[1],
                    "coord_z": atom.coord[2],
                    "B_factor_or_plDDT": atom.bfactor,
                }
                chain_atom_data[chain_id].append(atom_entry)

    return chain_atom_data, chain_ids, structure

def get_atom_indices_type_aa_type_and_coords(cif_file):
    chain_atom_data, _, _ = cif_parser_ATOM_HETATM_noncanonical(cif_file)
    
    list_of_atom_indices = []
    idx_type_coord = []
    atom_counts = {key: 0 for key in ALL_ATOM_LABELS}
    append_list_of_indices = list_of_atom_indices.append
    append_idx_type_coord = idx_type_coord.append
    
    for chain_data in chain_atom_data.values():
        for atom_entry in chain_data:
            index = atom_entry['atom_number']
            atom_type = atom_entry['atom_symbol_long']
            residue_name = atom_entry['residue_name']
            xyz = (atom_entry['coord_x'], atom_entry['coord_y'], atom_entry['coord_z'])
            idx_res = atom_entry['residue_number']
            
            append_list_of_indices(index)
            append_idx_type_coord((index, atom_type, idx_res, residue_name, xyz))
            atom_counts[atom_type] = atom_counts.get(atom_type, 0) + 1
    
    min_index = list_of_atom_indices[0]
    max_index = list_of_atom_indices[-1]
    missing_atom_idx = [i for i in range(min_index, max_index + 1) if i not in set(list_of_atom_indices)]
    
    return len(list_of_atom_indices), idx_type_coord, missing_atom_idx, atom_counts

def get_atom_indices_type_aa_type_and_coords_only_backbone(cif_file):
    chain_atom_data, _, _ = cif_parser_ATOM_HETATM_noncanonical(cif_file)
    
    list_of_atom_indices = []
    idx_type_coord = []
    atom_counts = {key: 0 for key in ALL_ATOM_LABELS}
    append_list_of_indices = list_of_atom_indices.append
    append_idx_type_coord = idx_type_coord.append
    
    for chain_data in chain_atom_data.values():
        for atom_entry in chain_data:
            index = atom_entry['atom_number']
            atom_type = atom_entry['atom_symbol_long']
            if atom_type not in ALL_LABELS_BACKBONE:
                continue
            residue_name = atom_entry['residue_name']
            xyz = (atom_entry['coord_x'], atom_entry['coord_y'], atom_entry['coord_z'])
            idx_res = atom_entry['residue_number']
            
            append_list_of_indices(index)
            append_idx_type_coord((index, atom_type, idx_res, residue_name, xyz))
            atom_counts[atom_type] = atom_counts.get(atom_type, 0) + 1
    
    min_index = list_of_atom_indices[0]
    max_index = list_of_atom_indices[-1]
    missing_idx = [i for i in range(min_index, max_index + 1) if i not in set(list_of_atom_indices)]
    
    return len(list_of_atom_indices), idx_type_coord, missing_idx, atom_counts

def extract_pqr_data(pqr_file, less_aa=False):
    atoms = []
    current_residue = -1
    last_residue_id = None
    
    with open(pqr_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                parts = line.split()
                
                if len(parts[4]) > 1 and parts[4][0].isalpha() and parts[4][1:].isdigit():
                    chain_id = parts[4][0]
                    residue_id = int(parts[4][1:])
                    
                    parts_new = parts[:4] + [chain_id] + [residue_id] + parts[5:]
                    parts = parts_new.copy()
                else:
                    residue_id = int(parts[5])
                
                residue_3_letter = parts[3]
                atom_label = parts[2]
                
                if less_aa and atom_label not in DICT_AA_SELECTION[residue_3_letter]:
                    continue

                
                if residue_id != last_residue_id:
                    current_residue += 1
                    last_residue_id = residue_id
                
                if parts[2].startswith('H'):
                    continue
                
                try:
                    atom_number = int(parts[1])
                    x, y, z, charge = map(float, parts[6:10])
                except (ValueError, IndexError):
                    continue
                
                atom_dict = {
                    'atom_number': atom_number,
                    'atom_label': atom_label,
                    'residue_name': residue_3_letter,
                    'residue_id': current_residue,
                    'x': x,
                    'y': y,
                    'z': z,
                    'charge': charge,
                }
                
                atoms.append(atom_dict)
    
    return atoms

def pqr_remove_hydrogen(in_file, out_file):
    with open(in_file, 'r') as infile, open(out_file, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM") and not line.split()[2].startswith('H'):
                outfile.write(line)

def build_edges_blockwise(atoms):
    edges = []
    residue_atoms = {}
    last_C_index = -1
    last_CA_index = -1
    
    for i, atom in enumerate(atoms):
        residue_id = atom['residue_id']
        if residue_id not in residue_atoms:
            residue_atoms[residue_id] = []
        residue_atoms[residue_id].append((i, atom))
    
    for residue_id, atom_list in residue_atoms.items():
        atom_dict = {a['atom_label']: idx for idx, a in atom_list}
        residue_name = atom_list[0][1]['residue_name']
        
        if residue_id != 0 and last_C_index >= 0 and 'N' in atom_dict:
            new_N_index = atom_dict['N']
            edges.append((last_C_index, new_N_index))
            edges.append((new_N_index, last_C_index))
        elif residue_id != 0 and last_C_index >= 0:
            min_index = min(atom_dict.values())
            edges.append((last_C_index, min_index))
            edges.append((min_index, last_C_index))
        
        bonds = [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'OXT')]
        for bond in bonds:
            if bond[0] in atom_dict and bond[1] in atom_dict:
                i, j = atom_dict[bond[0]], atom_dict[bond[1]]
                edges.append((i, j))
                edges.append((j, i))
                
                if bond[1] == 'C':
                    last_C_index = j
                if bond[1] == 'CA':
                    last_CA_index = j
        
        chain_atoms = [idx for name, idx in atom_dict.items() if name not in ['N', 'CA', 'C', 'O', 'OXT']]
        if not chain_atoms:
            continue
        
        if last_CA_index >= 0:
            edges.append((chain_atoms[0], last_CA_index))
            edges.append((last_CA_index, chain_atoms[0]))
        
        if residue_name in DICT_AA_RESIDUE_EDGES:
            for atom1, atom2 in DICT_AA_RESIDUE_EDGES[residue_name]:
                if atom1 in atom_dict and atom2 in atom_dict:
                    i, j = atom_dict[atom1], atom_dict[atom2]
                    edges.append((i, j))
                    edges.append((j, i))
                    
    connected_nodes = set([node for edge in edges for node in edge])
    isolated_nodes = [i for i in range(len(atoms)) if i not in connected_nodes]
    atom_entries_isolated = [atoms[i] for i in isolated_nodes]
    if isolated_nodes:
        print(atom_entries_isolated)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def build_edges_blockwise_less_atoms(atoms):
    edges = []
    residue_atoms = {}
    last_CA_index = -1
    
    for i, atom in enumerate(atoms):
        residue_id = atom['residue_id']
        if residue_id not in residue_atoms:
            residue_atoms[residue_id] = []
        residue_atoms[residue_id].append((i, atom))
    
    for residue_id, atom_list in residue_atoms.items():
        atom_dict = {a['atom_label']: idx for idx, a in atom_list}
        
        if residue_id != 0 and last_CA_index >= 0 and 'CA' in atom_dict:
            new_CA_index = atom_dict['CA']
            edges.append((last_CA_index, new_CA_index))
            edges.append((new_CA_index, last_CA_index))
        elif residue_id != 0 and last_CA_index >= 0 and atom_dict.values():
            min_index = min(atom_dict.values())
            edges.append((last_CA_index, min_index))
            edges.append((min_index, last_CA_index))
        
        chain_atoms = [idx for name, idx in atom_dict.items() if name not in ['N', 'CA', 'C', 'O', 'OXT']]
        if not chain_atoms:
            continue
        
        if last_CA_index >= 0:
            edges.append((chain_atoms[0], last_CA_index))
            edges.append((last_CA_index, chain_atoms[0]))
        
        side_chain_tuples = [(i,j) for i in chain_atoms for j in chain_atoms if i != j]
        edges.extend(side_chain_tuples)
        
        last_CA_index = atom_dict.get('CA', last_CA_index)
                    
    connected_nodes = set([node for edge in edges for node in edge])
    isolated_nodes = [i for i in range(len(atoms)) if i not in connected_nodes]
    atom_entries_isolated = [atoms[i] for i in isolated_nodes]
    if isolated_nodes:
        print(atom_entries_isolated)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def build_edges_knn_radius_combined(atoms, k=4, r=2.5, max_edges_per_node=3):
    positions = torch.tensor([[a['x'], a['y'], a['z']] for a in atoms])
    
    knn_edges = knn_graph(positions, k=k)
    
    radius_edges = radius_graph(positions, r=r)
    
    combined_edges = torch.cat([knn_edges, radius_edges], dim=1)
    combined_edges = torch.unique(combined_edges, dim=1)
    
    node_edge_count = torch.bincount(combined_edges[0])
    mask = node_edge_count[combined_edges[0]] <= max_edges_per_node
    combined_edges = combined_edges[:, mask]
    
    return combined_edges.t().contiguous()

def is_connected(edge_index):
    num_nodes = edge_index.max().item() + 1
    adj_matrix = csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())),
                            shape=(num_nodes, num_nodes))
    n_components, _ = connected_components(csgraph=adj_matrix, directed=False)
    
    return n_components == 1
