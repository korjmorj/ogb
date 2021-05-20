import torch
from torch_geometric.data import DataLoader, Data, DataListLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from gnn import GNN

import os
import os.path as osp
from tqdm import tqdm
import argparse
import time
import numpy as np
import random


### importing OGB-LSC
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator
from ogb.lsc import PCQM4MDataset
from ogb.utils import smiles2graph
import os
import os.path as osp
import shutil
from ogb.utils import smiles2graph
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

reg_criterion = torch.nn.L1Loss()

class PCQM4MEvaluator_cutted:
    def __init__(self):
        
        pass 

    def eval(self, input_dict):
        

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)
        assert(len(y_true.shape) == 1)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}

    def save_test_submission(self, input_dict, dir_path):
        '''
            save test submission file at dir_path
        '''
        assert('y_pred' in input_dict)
        y_pred = input_dict['y_pred']

        if not osp.exists(dir_path):
            os.makedirs(dir_path)
            
        filename = osp.join(dir_path, 'y_pred_pcqm4m')
        assert(isinstance(filename, str))
        assert(isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor))
 

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred = y_pred.astype(np.float32)
        np.savez_compressed(filename, y_pred = y_pred)


def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    print("y_true", y_true.shape)
    print("y_pred", y_pred.shape)

    return evaluator.eval(input_dict)["mae"]

def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--part', type=float, default=0.1)
    parser.add_argument('--create_test', type=bool, default=False) #переменная для создания полного тестового датасета
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '', help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PCQM4MDataset(root = 'dataset/', only_smiles = True)
    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator_cutted()
    
    ##ОГРОМНЫЙ КОСТЫЛЬ 
    def data_cutter(part, what):
        part_rows = int(len(split_idx[what])*part)
        graphs = []
        labels = []
        if what=='test':
          for idx in split_idx[what]:
              graph_obj = smiles2graph(dataset[idx][0])
              
              gap = dataset[idx][1]
              molecule = Data(x=torch.tensor(graph_obj['node_feat']), edge_index=torch.tensor(graph_obj['edge_index']), edge_attr=torch.tensor(graph_obj['edge_feat']), pos=torch.tensor(graph_obj['num_nodes']), y=gap)
              graphs.append(molecule)
        else:
          for i in range(part_rows):
            graph_obj = smiles2graph(dataset[split_idx[what][0]][0])
            
            gap = dataset[split_idx[what][0]][1]
            molecule = Data(x=torch.tensor(graph_obj['node_feat']), edge_index=torch.tensor(graph_obj['edge_index']), edge_attr=torch.tensor(graph_obj['edge_feat']), pos=torch.tensor(graph_obj['num_nodes']), y=gap)
            graphs.append(molecule)
            
        return graphs
    class cutted_dataset(object):
        def __init__(self, dataset=None, part=None, what=None):
 
            self.dataset = dataset
            self.part = part
            self.what = what

 
            self.data_cutter()

        def data_cutter(self):
            part_rows = int(len(split_idx[self.what])*self.part)

            self.graphs = []
            
            if self.what=='test':
                for idx in split_idx[self.what]:
                    graph_obj = smiles2graph(self.dataset[idx][0])
                    gap = torch.tensor(self.dataset[idx][1])
                    molecule = Data(x=torch.tensor(graph_obj['node_feat']), edge_index=torch.tensor(graph_obj['edge_index']), edge_attr=torch.tensor(graph_obj['edge_feat']), pos=torch.tensor(graph_obj['num_nodes']), y=gap)
                    self.graphs.append(graph_obj)
                    
            else:
                for i in range(part_rows):
                    idx = random.randint(0, len(split_idx[self.what]))
                    graph_obj = smiles2graph(self.dataset[split_idx[self.what][idx]][0])
                    gap = torch.tensor(self.dataset[split_idx[self.what][idx]][1])
                    molecule = Data(x=torch.tensor(graph_obj['node_feat']), edge_index=torch.tensor(graph_obj['edge_index']), edge_attr=torch.tensor(graph_obj['edge_feat']), pos=torch.tensor(graph_obj['num_nodes']), y=gap)
                    self.graphs.append(molecule)
                    
                    
            


        def __getitem__(self, idx):

            return self.graphs[idx] #, self.labels[idx]

        def __len__(self):

            return len(self.graphs)

        def __repr__(self):  # pragma: no cover
            return '{}({})'.format(self.__class__.__name__, len(self))

    train_data = cutted_dataset(dataset, args.part, 'train')
    print('train', len(train_data))
    valid_data = cutted_dataset(dataset, args.part, 'valid')
    print('valid', len(valid_data))
    if args.create_test:
        test_data = cutted_dataset(dataset, args.part, 'test')
        print('test', len(test_data))
    else:
        print('сегодня без теста')
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    test_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    ##КОНЕЦ ОГРОМНОГО КОСТЫЛЯ
    
    #if args.train_subset:
        #subset_ratio = 0.1
        #subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
        #train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    #else:
        #train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #if args.save_test_dir is not '':
        #test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.checkpoint_dir is not '':
        os.makedirs(args.checkpoint_dir, exist_ok = True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', virtual_node = True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', virtual_node = False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', virtual_node = True, **shared_params).to(device)
    else:
        raise ValueError('Invalid GNN type')

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir is not '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir is not '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            if args.create_test:
                if args.save_test_dir is not '':
                    print('Predicting on test data...')
                    y_pred = test(model, device, test_loader)
                    print('y_pred shape', y_pred.shape)
                    print('Saving test submission file...')
                    evaluator.save_test_submission({'y_pred': y_pred}, args.save_test_dir)

        scheduler.step()
            
        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir is not '':
        writer.close()


if __name__ == "__main__":
    main()
