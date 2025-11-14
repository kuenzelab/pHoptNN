import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
import sys
import time
from tqdm import tqdm

import random
import numpy as np
import pandas as pd
import math
import json
from ast import literal_eval

#from my_utils_own import get_hist_and_bins, calculate_gauss_weight, calculate_weights, bin_inverse_weights_for_batch, my_weighted_mse_loss
#from my_utils_own import focal_mse_loss

from create_dataset import EnzymeDataset
# mixed-precision training
from torch.cuda.amp import autocast, GradScaler

from collections import Counter
from scipy.ndimage import convolve1d
from utils_lds import get_lds_kernel_window
from loss_lds import weighted_mse_loss

from my_models import GCN, GCN_NNConv

from itertools import product
import argparse

#%%

parser = argparse.ArgumentParser(description="Train EGNN with different bin sizes.")
# =============================================================================
# parser.add_argument('--num_bins', type=int, required=True, help="Number of bins")
# parser.add_argument('--ks', type=int, required=True, help="ks in Gauss-Kernel")
# parser.add_argument('--sigma', type=float, required=True, help="sigma in Gauss-Kernel")
# =============================================================================
parser.add_argument('--root_dir', type=str, required=True, help="data root directory")
parser.add_argument('--l_max', type=int, required=True, help="l_max of spherical harmonics")
parser.add_argument('--edges_knn', action='store_true', help="is dataset create with KNN edges") # wenn flag vorhanden, True, else False
parser.add_argument('--less_aa', action='store_true', help="is dataset reduced to 13 aa") # wenn flag vorhanden, True, else False

args = parser.parse_args()

#%%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU, that's great!")
else:
    print('You are not using GPU which will be slow!')


ACTIVATION_FUNCTIONS = [nn.ReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6, nn.Tanh, nn.Sigmoid]
OPTIMIZERS = ['Adam', 'Rprop']
WEIGHTING_METHODS = ['bin_inv3', 'bin_inv3_root' ,'bin_inv5', 'bin_inv5_root',
                     'bin_inv10', 'bin_inv10_root', 'LDS_inv', 'LDS_inv_root',
                     'focal_loss']
POOLING_METHODS = ['mean', 'add', 'max', 'topk', 'set2set', 'global_att']


#%%

#### Paths and Hyperparameter

#root = "/home/sc.uni-leipzig.de/cc77miri/datasets_torch_geo/EnzDataset_atoms_surface_l_1/"
#root = '/home/iwe11/Christian/datasets_large/datasets_torch_geo/EnzyBase12k_all_atoms_charge_l1_train_test_50_50'
#root = 'C:\\Users\\HP\\Documents\\Studium\\Bioinformatik\\5.Semester\\masterproject_ph_ann\\datasets_torch_geo\\EnzyBase12k_all_atoms_charge_l1_train_test_50_50'
#root = 'C:\\Users\\HP\\Documents\\Studium\\Bioinformatik\\5.Semester\\masterproject_ph_ann\\reference_models\\EnzyBase12k_reduced_filtered.csv'
root = args.root_dir
data_csv = 'EnzyBase12k_final_overview.csv'
train_file_path = 'train.csv'
test_file_path = 'test.csv'

num_individuals = 10
num_generations = 1000
num_parents = 4
num_elitism_count = 1 # how many fittest individuals to keep

debug_mode = False
save_models_test_interval = False
save_losses = True

models_path = "./my_models_all_atoms_charge"

csv_name_to_save = os.path.join(root, 'losses_overview.csv')
csv_loss_name = './losses_all_atoms_charge/losses_l_max_1_'

load_model = False
load_model_config = False
model_path = os.path.join(models_path, "egnn_aa_grain_top_3.pth")

l_max = args.l_max

#learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
#pooling_list = ['mean', 'add', 'max', 'topk', 'set2set', 'global_att']
#lr_pool_list = list(product(learning_rates, pooling_list))

log_interval = 5 # how many batches to wait before logging training status
test_interval = 1 # how many epochs to wait before logging test
loss_fn = torch.nn.MSELoss()

dtype = torch.float32


def limit_value(value, min_val, max_val):
    return max(min(value, max_val), min_val)

def create_random_config():
    config = {
        'num_epochs': random.randint(3,10),
        'lr': round(10**random.uniform(-5, -1), 7),
        'activation_fn': random.choice(ACTIVATION_FUNCTIONS),
        'dropout': round(random.uniform(0.05, 0.5), 2),
        'optimizer_name': random.choice(OPTIMIZERS),
        'batch_size': random.choice([1, 2, 4, 8, 16]),
        'loss_weighting': 'LDS_Yang',
        'weight_decay': 0.0,
        'hidden_sizes_list': [],
        'pooling': random.choice(POOLING_METHODS),
        'ks': random.randint(1,10),
        'sigma': round(random.uniform(0.1, 1.5), 3),
        'weighting_factor': round(random.uniform(0.1, 2), 3),
        'num_bins': random.randint(14, 140)
        }
    
    config['weight_decay'] = round(config['lr'] * random.uniform(0.001, 0.1), 9)
    
    
    values = list(range(2, 11))
    weights = [1/i for i in range(2, 11)]
    num_layers = random.choices(values, weights=weights, k=1)[0]
    
    hidden_sizes_list = [random.randint(64, 128)]
    
    for i in range(1, num_layers):
        # Der nächste Wert ist kleiner als der vorherige
        last_value = hidden_sizes_list[i - 1]
        next_value = max(random.randint(last_value // 2, last_value - 1), 1)
        hidden_sizes_list.append(next_value)
        
    config['hidden_sizes_list'] = hidden_sizes_list

    return config

def generate_json_files_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        # Aktivierungsfunktion aus String extrahieren
        activation_fn_str = row['activation_fn']
        #activation_fn_clean = activation_fn_str.split("'")[-2].split('.')[-1]

        # Dictionary für JSON
        config = {
            'num_epochs': int(row['num_epochs']),
            'lr': float(row['lr']),
            'activation_fn': activation_fn_str,
            'dropout': float(row['dropout']),
            'optimizer_name': row['optimizer_name'],
            'batch_size': int(row['batch_size']),
            'loss_weighting': row['loss_weighting'],
            'weight_decay': float(row['weight_decay']),
            'hidden_sizes_list': literal_eval(row['hidden_sizes_list']),
            'pooling': row['pooling'],
            'ks': int(row['ks']),
            'sigma': float(row['sigma']),
            'weighting_factor': float(row['weighting_factor']),
            'num_bins': int(row['num_bins']),
            'Test_loss': float(row['Test_loss']),
            'MSE_test_loss': float(row['MSE_test_loss'])
        }

        with open(f'gen_0_ind_{idx}.json', 'w') as f:
            json.dump(config, f, indent=4)
            
def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    activation_map = {f"<class '{fn.__module__}.{fn.__name__}'>": fn for fn in ACTIVATION_FUNCTIONS}
    
    config = {
        'num_epochs': int(data['num_epochs']),
        'lr': float(data['lr']),
        'activation_fn': activation_map.get(data['activation_fn'], None),
        'dropout': float(data['dropout']),
        'optimizer_name': data['optimizer_name'],
        'batch_size': int(data['batch_size']),
        'loss_weighting': data['loss_weighting'],
        'weight_decay': float(data['weight_decay']),
        'hidden_sizes_list': data['hidden_sizes_list'],
        'pooling': data['pooling'],
        'ks': int(data['ks']),
        'sigma': float(data['sigma']),
        'weighting_factor': float(data['weighting_factor']),
        'num_bins': int(data['num_bins'])
    }

    return config

def crossover_and_mutate(parents):    
    hyp_para_config = {}
    '''
        cross_over
    '''
    for key in parents[0].keys():
        hyp_para_config[key] = random.choice([parent[key] for parent in parents])
    
    '''
        mutation
    '''
    num_mutations = random.randint(1, 12)
    
    keys_for_mutation = list(hyp_para_config.keys())
    keys_for_mutation.remove('loss_weighting')
    
    selected_keys = random.sample(keys_for_mutation, num_mutations)
    
    for key in selected_keys:
        extreme = random.random() < 0.1  # 10% Cchance of extremer mutation
        
        if key == 'num_epochs':
            #hyp_para_config[key] = random.randint(2, 2) if extreme else random.randint(2, 2)
            hyp_para_config[key] = random.randint(10, 20) if extreme else random.randint(3, 10)
        if key == 'lr':
            scale_factor = random.uniform(0.5, 2) if extreme else random.uniform(0.8, 1.2)
            hyp_para_config[key] = limit_value(hyp_para_config[key] * scale_factor, 1e-7, 1e-1)
        elif key == 'activation_fn':
            hyp_para_config[key] = random.choice(ACTIVATION_FUNCTIONS)
        elif key == 'dropout':
            hyp_para_config[key] = round(random.uniform(0.0, 0.9), 2) if extreme else round(random.uniform(0.05, 0.5), 2)
        elif key == 'optimizer_name':
            hyp_para_config[key] = random.choice(OPTIMIZERS)
        elif key == 'hidden_sizes_list':
            if random.random() < 0.7:
                num_layers = random.randint(1, 2)
            else:
                num_layers = random.randint(3, 5)
            
            hyp_para_config[key] = sorted([random.randint(10, 128) for _ in range(num_layers)], reverse=True)
        elif key == 'batch_size':
            hyp_para_config[key] = random.choice([1, 2, 32, 64]) if extreme else random.choice([4, 8, 16])
        elif key == 'pooling':
            hyp_para_config[key] = random.choice(POOLING_METHODS)
        elif key == 'weighting_factor':
            hyp_para_config[key] = round(random.uniform(0.01, 5), 3) if extreme else round(random.uniform(0.1, 2), 3)
        elif key == 'weight_decay':
            scale = random.uniform(0.0001, 1.0) if extreme else random.uniform(0.001, 0.1)
            hyp_para_config[key] = hyp_para_config['lr'] * scale
        elif key == 'ks':
            hyp_para_config[key] = random.randint(1, 30) if extreme else random.randint(1, 10)
        elif key == 'sigma':
            hyp_para_config[key] = round(random.uniform(0.05, 3.), 3) if extreme else round(random.uniform(0.1, 1.5), 3)
        elif key == 'num_bins':
            hyp_para_config[key] = random.randint(hyp_para_config[key] - 10, hyp_para_config[key] + 10) if extreme else random.randint(hyp_para_config[key] - 5, hyp_para_config[key] + 5)
    
    return hyp_para_config

def roulette_wheel_selection(population, num_parents):
    valid_population = [ind for ind in population if ind['Test_loss'] > 0 and math.isfinite(ind['Test_loss'])]
    
    if not valid_population:
        raise ValueError("No valid individuals for selection.")
    
    # Berechne die Fitnesswerte (1 / Test_loss)
    fitness_values = [1 / ind['MSE_test_loss'] for ind in valid_population]
    total_fitness = sum(fitness_values)
    
    # Wahrscheinlichkeiten berechnen
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    
    # Wähle num_parents Individuen basierend auf ihren Wahrscheinlichkeiten ohne Zurücklegen
    selected_indices = set()
    parents = []
    while len(parents) < num_parents:
        selected_index = random.choices(range(len(valid_population)), weights=probabilities, k=1)[0]
        
        if selected_index not in selected_indices:
            selected_indices.add(selected_index)
            parents.append(valid_population[selected_index])

    return parents

def get_bin_idx(label, num_bins=140):
    lower_bound = (0 - y_mean) / y_std
    upper_bound = (14 - y_mean) / y_std
    bin_size = (upper_bound - lower_bound) / num_bins
    idx = min(num_bins - 1, int((label - lower_bound) / bin_size))
    return idx

#%%

def train(epoch, train_loader, individual_config):
    scaler = GradScaler()
    train_loss = 0.0
    
    #print(f'Len of loader is {len(loader)}')
    #print(f'Len of data/batch size is {len(loader[0]["num_atoms"])}')
    #for i, data in enumerate(loader):
    #for train_count, data in enumerate(tqdm(loader)):
    for train_count, batch in enumerate(tqdm(train_loader)):
        try:
            batch = batch.to(device)
        except RuntimeError as err:
            print(err)
            print(batch)
            print(batch.x)
            print(batch.edge_index)
            print(batch.y)
            sys.exit()
            
        # Optimizer und Gradienten zurücksetzen
        optimizer.zero_grad(set_to_none=True)
        
        label = batch.y
        
        # standardise label - obsolet, since dataset y is already normed
        #label_norm = (label - y_mean) / y_std
        
        with autocast():
            try:
                pred = model(batch, device)
            except RuntimeError:
                break
            
            bin_indices = [get_bin_idx(one_label.item(), num_bins=individual_config['num_bins']) for one_label in label]
            
            label_weights = [weights[bin_index_per_label.index(bin_idx)] for bin_idx in bin_indices]
            label_weights = torch.tensor(label_weights, dtype=dtype, device=device)
            
            my_mse_loss = weighted_mse_loss(pred, label, label_weights)
            #loss = loss_fn(pred, (label - median) / mad)
            
            if debug_mode == True:
                print("Out is", pred)
                #print("Loss is ", loss, loss.shape)
                print("Data.y is:", label)
                #print(f"Weights of batch {train_count}:", weights_batch)
                print(f"Own calculated loss is: {my_mse_loss}", my_mse_loss.shape)
                print(f"out: {pred}, shape of out: {pred.shape}")
                print(f"batch.y: {label}, shape of batch.y: {label.shape}")
                
# =============================================================================
#             if torch.isnan(my_mse_loss):
#                 epoch_train_loss = 420
#                 break
# =============================================================================
                
        scaler.scale(my_mse_loss).backward()
        #my_mse_loss.backward()
        
        scaler.step(optimizer)
        #optimizer.step()
        
        scaler.update()
        
        train_loss += my_mse_loss.item()
        
    epoch_train_loss = train_loss/len(train_loader)
        
# =============================================================================
#     if train_count % 1000 == 0:
#         print(f'Training for batch {train_count}, epoch {epoch} finished.')
# =============================================================================

    return epoch_train_loss


def test(epoch, test_loader, individual_config):
    test_loss = 0.0
    
    test_loss_MSE = 0.0
    
    #print(f'Len of loader is {len(loader)}')
    #print(f'Len of data/batch size is {len(loader[0]["num_atoms"])}')
    #for i, data in enumerate(loader):
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        
        label = batch.y
        label_destd = label * y_std + y_mean
        
        # standardise label - obsolet, since dataset y is already normed
        #label_norm = (label - y_mean) / y_std
        
        try:
            pred = model(batch, device)
        except RuntimeError:
            break
        
        pred_destd = pred * y_std + y_mean
        
        epoch_loss = loss_fn(pred_destd, label_destd)
        test_loss_MSE += epoch_loss.item()
        
        
        bin_indices = [get_bin_idx(one_label.item(), num_bins=individual_config['num_bins']) for one_label in label]
        
        label_weights = [weights[bin_index_per_label.index(bin_idx)] for bin_idx in bin_indices]
        label_weights = torch.tensor(label_weights, dtype=dtype, device=device)
        my_mse_loss = weighted_mse_loss(pred, label, label_weights)
        
# =============================================================================
#         if torch.isnan(my_mse_loss).any():
#             return 420., 420.
# =============================================================================
        
        test_loss += my_mse_loss.item()
        
    epoch_test_loss = test_loss/len(test_loader)

    epoch_test_loss_MSE = test_loss_MSE / len(test_loader)

    return epoch_test_loss, epoch_test_loss_MSE

def save_model(model, model_name_params, models_path):
    try:
        # Speichere nur die Modellparameter (empfohlen)
        torch.save(model.state_dict(), os.path.join(models_path, model_name_params))  # Beispiel: 'model_params.pth'
    except Exception as e:
        print(f"Error saving model parameters: {e}")
        
# load model:
# =============================================================================
# model = YourModelClass(*args)  # Ersetze mit deiner Modellklasse und Argumenten
# model.load_state_dict(torch.load(os.path.join(models_path, model_name_params)))
# model.to(device)
# =============================================================================

if __name__ == '__main__':
    csv_path = os.path.join(root, 'raw', data_csv)
    data_df = pd.read_csv(csv_path)
    
    label_values = np.asarray(data_df['ph_optimum'].to_list())
    
    y_mean, y_std = label_values.mean(), label_values.std()
    
    y_mean = torch.tensor(y_mean).to(device)
    y_std = torch.tensor(y_std).to(device)
    
    df_with_losses = pd.DataFrame()
    
    runtimes = []
    
    #%%
    ### Load Data
    
    train_dataset = EnzymeDataset(root=root, filename=train_file_path, y_mean=y_mean, y_std=y_std, test=False, norm_X=True, 
                                                  norm_y=True, l_max=l_max, only_surface_atoms=False,
                                                  edges_knn=args.edges_knn, less_aa=args.less_aa)
    
    test_dataset = EnzymeDataset(root=root, filename=test_file_path, y_mean=y_mean, y_std=y_std, test=True, norm_X=True, 
                                                 norm_y=True, l_max=l_max, only_surface_atoms=False,
                                                 edges_knn=args.edges_knn, less_aa=args.less_aa)
    
    
    #%%
    
    ### Hyperparameter Optimization Loop
    
    #for k in range(num_models):
    k = 0
    start_time = time.time()
    '''
        create random hyperparameter configuration
    '''
    model_name_params = ''
    
    num_node_features = train_dataset.num_node_features
    #num_node_features = train_dataset[0].x.shape[1]
    
    population = [create_random_config() for _ in range(num_individuals)]
    
# =============================================================================
#     print("First population:")
#     for dictionary in population:
#         for key, value in dictionary.items():
#             print(f"  {key}: {value}")
# =============================================================================
    
    population_sorted = []
    runtimes = []
    
    for generation in range(num_generations):
        start_time = time.time()
        if generation != 0:
            parents = roulette_wheel_selection(population_sorted, num_parents)
            
            for i in range(num_elitism_count, num_individuals):
                population_sorted[i] = crossover_and_mutate(parents)
            
            population = population_sorted.copy()
        
        for indi_count, individual_config in enumerate(population):
            # after first generation, population[0] is fittest model, which don't
            # need to be trained again
            if generation != 0 and (indi_count in range(num_elitism_count)):
                continue
            
# =============================================================================
#             while True:
#                 try:
# =============================================================================
            # load data
            train_loader = DataLoader(train_dataset,
                                      batch_size=individual_config['batch_size'], shuffle=True)
            
            test_loader = DataLoader(test_dataset,
                                      batch_size=individual_config['batch_size'], shuffle=False)
            
            # initialize model
            print('Initialize model.')
            try:
                if load_model:
                    model = torch.load(model_path)
                else:
                    model = GCN(num_node_features, num_layers=len(individual_config['hidden_sizes_list']), hidden_sizes_list=individual_config['hidden_sizes_list'],
                                dropout=individual_config['dropout'], activation_func=individual_config['activation_fn'], k=0.2, pool=individual_config['pooling'])
            except:
                print("Error in model generation")
                
            model.to(device)
            print("model is on device")
            
            
            if individual_config['optimizer_name'] == 'Adam':
                optimizer = optim.AdamW(model.parameters(), lr=individual_config['lr'], weight_decay=individual_config['weight_decay'])
            elif individual_config['optimizer_name'] == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=individual_config['lr'], weight_decay=individual_config['weight_decay'])
            elif individual_config['optimizer_name'] == 'Rprop':
                optimizer = optim.Rprop(model.parameters(), lr=individual_config['lr'])
            else:
                print("Error in Optimizer Initialisation")
                
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epochs'])
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-10)
            #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            if individual_config['loss_weighting'] == 'bin_inv3':
                num_bins = [0,5,9,14]
                root_ = False
            elif individual_config['loss_weighting'] == 'bin_inv3_root':
                num_bins = [0,5,9,14]
                root_ = True
            elif individual_config['loss_weighting'] == 'bin_inv5':
                num_bins = 5
                root_ = False
            elif individual_config['loss_weighting'] == 'bin_inv5_root':
                num_bins = 5
                root_ = True
            elif individual_config['loss_weighting'] == 'bin_inv10':
                num_bins = 10
                root_ = False
            elif individual_config['loss_weighting'] == 'bin_inv10_root':
                num_bins = 10
                root_ = True
            elif individual_config['loss_weighting'] == 'LDS_inv':
                num_bins = 100
                root_ = False
            elif individual_config['loss_weighting'] == 'LDS_inv_root':
                num_bins = 100
                root_ = True
                
            
            #sqrt is already executed in calculate_weights
                
            if individual_config['loss_weighting'] == 'LDS_Yang':
                # preds, labels: [Ns,], "Ns" is the number of total samples
                
                label_values = torch.tensor(label_values, dtype=torch.float32).to(device)
                
                labels_norm = (label_values - y_mean) / y_std
                
                labels = labels_norm.tolist()
                
                # assign each label to its corresponding bin (start from 0)
                # with your defined get_bin_idx(), return bin_index_per_label: [Ns,] 
                bin_index_per_label = [get_bin_idx(label, num_bins=individual_config['num_bins']) for label in labels]
        
                # calculate empirical (original) label distribution: [Nb,]
                # "Nb" is the number of bins
                Nb = max(bin_index_per_label) + 1
                num_samples_of_bins = dict(Counter(bin_index_per_label))
                emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
        
                # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
                lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=individual_config['ks'], sigma=individual_config['sigma'])
                # calculate effective label distribution: [Nb,]
                eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        
        
                # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
                eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
                weights = [np.float32(1 / (x**individual_config['weighting_factor'])) for x in eff_num_per_label]
        
                # calculate loss
                #loss = weighted_mse_loss(preds, labels, weights=weights)
            
            #pH_labels_all_norm = (pH_labels_all - mean) / std
            
            #%%
            #optimizer = optimizer.to(device)
            
            train_losses = []
            test_losses = []
            test_losses_MSE = []
            
            print(f'Start training for model {k}')
            #try:
            epoch_test_loss = float('inf')
            for epoch in range(individual_config['num_epochs']):
                #print(f"Epoch {epoch}")
                model.train()
                #try:
                epoch_train_loss = train(epoch, train_loader, individual_config)
        # =============================================================================
        #             except RuntimeError as err:
        #                 print(f'Runtime Error in train function for model {k}')
        #                 print(err)
        #                 save_losses = False
        #                 break
        # =============================================================================
                
# =============================================================================
#                 if epoch_train_loss == 420 or epoch_train_loss == 0.0:
#                     save_losses = False
#                     break
# =============================================================================
                
                train_losses.append(epoch_train_loss)
                print(f'Epoch {epoch} - train_loss: {epoch_train_loss:.4f}')
                
                if epoch % test_interval == 0:
                    # Berechnung des Testverlusts
                    model.eval()
                    test_loss = 0.0
                    with torch.inference_mode():
                        try:
                            epoch_test_loss, epoch_test_loss_MSE = test(epoch, test_loader, individual_config)
                        except RuntimeError as err:
                            print(f'Runtime Error in test function for model {k}')
                            print(err)
                            save_losses = False
                            break
                        
                        if epoch > 10 and epoch_test_loss > 1000:
                            save_losses = False
                            break
                    
                    print(f'Epoch {epoch} - train_loss: {epoch_train_loss:.4f} - test_loss: {epoch_test_loss:.4f} - MSE test_loss: {epoch_test_loss_MSE:.4f}')
                    test_losses.append(epoch_test_loss)
                    test_losses_MSE.append(epoch_test_loss_MSE)
                    
                    if save_models_test_interval:
                        save_model(model, model_name_params, models_path)
                    
                
                # after each epoch
                torch.cuda.empty_cache()
                #scheduler.step()
            
            
            
            #df_only_losses = pd.DataFrame({'train_loss': train_losses, 'test_loss': test_losses, 'test_loss_MSE': test_losses_MSE})
            #losses_path = csv_loss_name + '_lr_' + str(individual_config['lr']) + '_pool_' + str(individual_config['pooling']) + '.csv'
            #df_only_losses.to_csv(losses_path, mode='a', header=True, index=False)
            

            test_loss = globals().get("epoch_test_loss", None)
            population[indi_count]['Test_loss'] = 420.0 if test_loss in [None, 0.0] else round(test_loss, 8)
            
            mse_loss = globals().get("epoch_test_loss_MSE", None)
            population[indi_count]['MSE_test_loss'] = 420.0 if mse_loss in [None, 0.0] else round(mse_loss, 8)

            
            single_row_dict = {key: [value] for key, value in population[indi_count].items()}
            df_single = pd.DataFrame(single_row_dict)
            
            if k == 0:
                df_single.to_csv(csv_name_to_save, mode='a', header=True, index=False)
            else:
                df_single.to_csv(csv_name_to_save, mode='a', header=False, index=False)
            
            k += 1
# =============================================================================
#                     if df_with_losses.empty:
#                         df_with_losses = pd.DataFrame(single_row_dict)
#                     else:
#                         df = pd.DataFrame(single_row_dict)
#                         df_with_losses = pd.concat([df_with_losses, df])
# =============================================================================
                    
# =============================================================================
#                     break
#                 except Exception:
#                     print(f'Exeption occured, create new random indivual for count {indi_count}.')
#                     population[indi_count] = create_random_config()
# =============================================================================
                    
        population_sorted = sorted(population, key=lambda x: x['Test_loss'])
        
        runtime = time.time() - start_time
        runtimes.append(runtime)
        
        if len(runtimes) % 10 == 0:
            print(runtimes)
        
        print()
        print(f"Overall training runtime of generation {generation} : {runtime}.")
        print()
    
    print(f"Mean Runtime for all {num_generations} generations: {np.mean(runtimes)}")
    #print(f"Mean Runtime for all {num_models} models: {np.mean(runtimes)}")
    