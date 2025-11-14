import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
import sys
import time
from tqdm import tqdm
import ast
import random
import numpy as np
import pandas as pd
import json

from create_torch_geo_dataset import EnzymeDataset

from collections import Counter
from scipy.ndimage import convolve1d
from utils_lds import get_lds_kernel_window
from loss_lds import weighted_mse_loss

from my_models import GCN

import argparse

#%%

parser = argparse.ArgumentParser(description="Train torch geo model with different bin sizes.")

parser.add_argument('--root_dir', type=str, required=True, help="data root directory")
parser.add_argument('--l_max', type=int, required=True, help="l_max of spherical harmonics")
parser.add_argument('--edges_knn', action='store_true', help="is dataset create with KNN edges") # wenn flag vorhanden, True, else False
parser.add_argument('--less_aa', action='store_true', help="is dataset reduced to 13 aa") # wenn flag vorhanden, True, else False
parser.add_argument('--df_individuals', type=str, required=True, help='Path to CSV df with 10 individuals of current generation')
parser.add_argument('--idx_individual', type=int, required=True, help='Row index of df_individuals')
parser.add_argument('--idx_gen', type=int, required=True, help='Number of generation')

parser.add_argument('--save_losses', action='store_true', help="if set, loss is saved in each epoch")
parser.add_argument('--save_model', action='store_true', help="if set, model is saved each epoch")

args = parser.parse_args()

#%%

generator = torch.Generator().manual_seed(42)

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

len_valid_dataset = 0.875

debug_mode = False

models_path = "/data/horse/ws/chcl580g-gnn_torch_geo/my_models"

csv_name_to_save = os.path.join(root, 'losses_actual_generation_trained.csv')

# for args.save_losses
csv_loss_logging_path = os.path.join(root, 'losses', f'losses_ind_{args.idx_individual}.csv')

# for args.save_model
model_save_path = os.path.join(root, 'models', f'model_{args.idx_individual}.pt')
model_params_save_path = os.path.join(root, 'models', f'model_{args.idx_individual}_params.pt')


load_model = False
load_model_config = False
load_model_path = os.path.join(models_path, "egnn_aa_grain_top_3.pth")

l_max = args.l_max

#learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
#pooling_list = ['mean', 'add', 'max', 'topk', 'set2set', 'global_att']
#lr_pool_list = list(product(learning_rates, pooling_list))

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

def config_from_csv_row(csv_path, row_idx=0):
    df = pd.read_csv(csv_path)
    row = df.iloc[row_idx]

    activation_map = {f"<class '{fn.__module__}.{fn.__name__}'>": fn for fn in ACTIVATION_FUNCTIONS}

    config = {
        'num_epochs': int(row['num_epochs']),
        'lr': float(row['lr']),
        'activation_fn': activation_map.get(row['activation_fn'], None),
        'dropout': float(row['dropout']),
        'optimizer_name': row['optimizer_name'],
        'batch_size': int(row['batch_size']),
        'loss_weighting': row['loss_weighting'],
        'weight_decay': float(row['weight_decay']),
        'hidden_sizes_list': ast.literal_eval(row['hidden_sizes_list']),
        'pooling': row['pooling'],
        'ks': int(row['ks']),
        'sigma': float(row['sigma']),
        'weighting_factor': float(row['weighting_factor']),
        'num_bins': int(row['num_bins'])
    }

    return config

def get_bin_idx(label, num_bins=140):
    lower_bound = (0 - y_mean) / y_std
    upper_bound = (14 - y_mean) / y_std
    bin_size = (upper_bound - lower_bound) / num_bins
    idx = min(num_bins - 1, int((label - lower_bound) / bin_size))
    return idx

#%%

def train(epoch, train_loader, individual_config):
    #scaler = GradScaler()
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
        
        #with autocast():
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
                
        #scaler.scale(my_mse_loss).backward()
        my_mse_loss.backward()
        
        #scaler.step(optimizer)
        optimizer.step()
        
        #scaler.update()
        
        train_loss += my_mse_loss.item()
        
    epoch_train_loss = train_loss/len(train_loader)
        
# =============================================================================
#     if train_count % 1000 == 0:
#         print(f'Training for batch {train_count}, epoch {epoch} finished.')
# =============================================================================

    return epoch_train_loss


def test(epoch, valid_loader, individual_config):
    test_loss = 0.0
    
    test_loss_MSE = 0.0
    
    #print(f'Len of loader is {len(loader)}')
    #print(f'Len of data/batch size is {len(loader[0]["num_atoms"])}')
    #for i, data in enumerate(loader):
    for batch in tqdm(valid_loader):
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
        
    epoch_test_loss = test_loss/len(valid_loader)

    epoch_test_loss_MSE = test_loss_MSE / len(valid_loader)

    return epoch_test_loss, epoch_test_loss_MSE
        
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
    
    config = config_from_csv_row(args.df_individuals, args.idx_individual)
    
    #%%
    ### Load Data
    
    train_dataset = EnzymeDataset(root=root, filename=train_file_path, y_mean=y_mean, y_std=y_std, test=False, norm_X=True, 
                                                  norm_y=True, l_max=l_max, only_surface_atoms=False,
                                                  edges_knn=args.edges_knn, less_aa=args.less_aa)
    test_dataset = EnzymeDataset(root=root, filename=test_file_path, y_mean=y_mean, y_std=y_std, test=True, norm_X=True, 
                                                  norm_y=True, l_max=l_max, only_surface_atoms=False,
                                                  edges_knn=args.edges_knn, less_aa=args.less_aa)
    
    
    num_train = int(len(train_dataset) * len_valid_dataset)
    num_val = len(train_dataset) - num_train
    
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val], generator=generator)
    
    #%%
    
    ### Hyperparameter Optimization Loop
    
    #for k in range(num_models):
    k = 0
    
    '''
        create random hyperparameter configuration
    '''
    model_name_params = 'best_all_atoms_l_1.pth'
    
    num_node_features = train_dataset.num_node_features
    #num_node_features = train_dataset[0].x.shape[1]
    
    start_time = time.time()
    
    # load data
    #train_loader = DataLoader(train_dataset,
    #                          batch_size=config['batch_size'], shuffle=True)
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
    
    # initialize model
    print('Initialize model.')
    try:
        if load_model:
            model = torch.load(load_model_path)
        else:
            model = GCN(num_node_features, num_layers=len(config['hidden_sizes_list']), hidden_sizes_list=config['hidden_sizes_list'],
                        dropout=config['dropout'], activation_func=config['activation_fn'], k=0.2, pool=config['pooling'])
    except:
        print("Error in model generation")
        
    model.to(device)
    print("model is on device")
    
    
    if config['optimizer_name'] == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer_name'] == 'Rprop':
        optimizer = optim.Rprop(model.parameters(), lr=config['lr'])
    else:
        print("Error in Optimizer Initialisation")
        
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epochs'])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-10)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=10,
                                                           min_lr= 0.001 * config['lr']
                                                           )
    
    #sqrt is already executed in calculate_weights
        
    if config['loss_weighting'] == 'LDS_Yang':
        # preds, labels: [Ns,], "Ns" is the number of total samples
        
        label_values = torch.tensor(label_values, dtype=torch.float32).to(device)
        
        labels_norm = (label_values - y_mean) / y_std
        
        labels = labels_norm.tolist()
        
        # assign each label to its corresponding bin (start from 0)
        # with your defined get_bin_idx(), return bin_index_per_label: [Ns,] 
        bin_index_per_label = [get_bin_idx(label, num_bins=config['num_bins']) for label in labels]

        # calculate empirical (original) label distribution: [Nb,]
        # "Nb" is the number of bins
        Nb = max(bin_index_per_label) + 1
        num_samples_of_bins = dict(Counter(bin_index_per_label))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=config['ks'], sigma=config['sigma'])
        # calculate effective label distribution: [Nb,]
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')


        # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
        eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        weights = [np.float32(1 / (x**config['weighting_factor'])) for x in eff_num_per_label]

        # calculate loss
        #loss = weighted_mse_loss(preds, labels, weights=weights)
    
    #pH_labels_all_norm = (pH_labels_all - mean) / std
    
    #%%
    #optimizer = optimizer.to(device)
    
    train_losses = []
    test_losses = []
    test_losses_MSE = []
    train_times = []
    test_times = []
    
    print(f'Start training for model {k}')
    #try:
    epoch_test_loss = float('inf')
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        #print(f"Epoch {epoch}")
        model.train()
        #try:
        epoch_train_loss = train(epoch, train_loader, config)
        
        train_losses.append(epoch_train_loss)
        
        epoch_train_time = time.time() - start_time
        train_times.append(epoch_train_time)
        
        if epoch % test_interval == 0:
            start_time = time.time()
            # Berechnung des Testverlusts
            model.eval()
            test_loss = 0.0
            with torch.inference_mode():
                try:
                    epoch_test_loss, epoch_test_loss_MSE = test(epoch, valid_loader, config)
                    scheduler.step(epoch_test_loss)
                except RuntimeError as err:
                    print(f'Runtime Error in test function for model {k}')
                    print(err)
                    save_losses = False
                    break
                
                if epoch > 10 and epoch_test_loss > 1000:
                    save_losses = False
                    break
            
            print(f'Epoch {epoch} - train_loss: {epoch_train_loss:.6f} - test_loss: {epoch_test_loss:.6f} - MSE test_loss: {epoch_test_loss_MSE:.4f}')
            test_losses.append(epoch_test_loss)
            test_losses_MSE.append(epoch_test_loss_MSE)
            
            epoch_test_time = time.time() - start_time
            test_times.append(epoch_test_time)
            
            if args.save_model and epoch_test_loss < best_val_loss:
                best_val_loss = epoch_test_loss
                try:
                    torch.save(model.state_dict(), model_params_save_path)
                except Exception as e:
                    print(f"Error while saving model params: {e}")
                    
                try:
                    torch.save(model, model_save_path)
                except Exception as e:
                    print(f"Error while saving model: {e}")
        
        # after each epoch
        torch.cuda.empty_cache()
    
    test_loss = globals().get("epoch_test_loss", None)
    config['Test_loss'] = 420.0 if test_loss in [None, 0.0] else round(test_loss, 8)
    
    mse_loss = globals().get("epoch_test_loss_MSE", None)
    config['MSE_test_loss'] = 420.0 if mse_loss in [None, 0.0] else round(mse_loss, 8)
    
    single_row_dict = {key: [value] for key, value in config.items()}
    df_single = pd.DataFrame(single_row_dict)
    
    if not args.save_losses:
        #write_header = not os.path.exists(csv_name_to_save)
        df_single.to_csv(csv_name_to_save, mode='a', header=False, index=False)
        
        df_single.to_csv(os.path.join(root, 'losses_overview.csv'), mode='a', header=False, index=False)
        
        config['activation_fn'] = str(config['activation_fn'])
        with open(os.path.join(root, 'json_results', f'gen_{args.idx_gen}_ind_{args.idx_individual}.json'), 'w') as f:
            json.dump(config, f, indent=4)
    else:
        # save 3x losses, train_time_per_epoch, val_time_per_epoch
        df_losses = pd.DataFrame({'train_loss': train_losses, 'val_loss': test_losses, 'val_loss_MSE': test_losses_MSE,
                                  'epoch_train_time': train_times, 'epoch_val_time': test_times})
        df_losses.to_csv(csv_loss_logging_path, index=False)
    
# =============================================================================
#     print()
#     print(f"Overall training runtime: {runtime}.")
#     print()
# =============================================================================
