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
    print("y_pred", y_pred)

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
    #dataset = PCQM4MDataset(root = 'dataset/', only_smiles = True)
    #split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()
    
    ##ОГРОМНЫЙ КОСТЫЛЬ 
    class PCQM4MDataset_cutted(object):
        def __init__(self, root = 'dataset', smiles2graph = smiles2graph, only_smiles=False, part=None, what=None):


            self.original_root = root
            self.smiles2graph = smiles2graph
            self.only_smiles = only_smiles
            self.folder = osp.join(root, 'pcqm4m_kddcup2021')
            self.version = 1
            self.what=what
            self.part=part

        # Old url hosted at Stanford
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m_kddcup2021.zip'
        # New url hosted by DGL team at AWS--much faster to download
            self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip'

        # check version and update if necessary
            if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
                print('PCQM4M dataset has been updated.')
                if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                    shutil.rmtree(self.folder)

            #super(PCQM4MDataset, self).__init__()

        # Prepare everything.
        # download if there is no raw file
        # preprocess if there is no processed file
        # load data if processed file is found.
            if self.only_smiles:
                self.prepare_smiles()
            else:
                self.prepare_graph()

        def download(self):
            if decide_download(self.url):
                path = download_url(self.url, self.original_root)
                extract_zip(path, self.original_root)
                os.unlink(path)
            else:
                print('Stop download.')
                exit(-1)

        def prepare_smiles(self):
            raw_dir = osp.join(self.folder, 'raw')
            if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
            # if the raw file does not exist, then download it.
                self.download()

            data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
            smiles_list = data_df['smiles'].values
            homolumogap_list = data_df['homolumogap'].values
            self.graphs = list(smiles_list)
            self.labels = homolumogap_list

        def prepare_graph(self):
        
            processed_dir = osp.join(self.folder, 'processed')
            raw_dir = osp.join(self.folder, 'raw')
            pre_processed_file_path = osp.join(processed_dir, 'data_processed')

            if osp.exists(pre_processed_file_path):        
            # if pre-processed file already exists
                loaded_dict = torch.load(pre_processed_file_path, 'rb')
                self.graphs, self.labels = loaded_dict['graphs'], loaded_dict['labels']
        
            else:
            # if pre-processed file does not exist
            
                if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
                # if the raw file does not exist, then download it.
                    self.download()

                data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
                smiles_list = data_df['smiles']
                homolumogap_list = data_df['homolumogap']

                print('Converting SMILES strings into graphs...')
                split_idx = self.get_idx_split()
                part_rows = int(len(split_idx[self.what])*self.part)
                print(self.what, part_rows)
                print(split_idx[self.what][i])
                print('one smiles', smiles_list[split_idx[self.what][0]])
                self.graphs = []
                self.labels = []
            
                if self.what=='test':
                    for i in len(split_idx[self.what]):
                        smiles = smiles_list[split_idx[self.what][i]]
                        homolumogap = homolumogap_list[split_idx[self.what][i]]
                        graph = self.smiles2graph(smiles)
                
                        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
                        assert(len(graph['node_feat']) == graph['num_nodes'])

                        self.graphs.append(graph)
                        self.labels.append(homolumogap)

                    self.labels = np.array(self.labels)
                    print(self.labels)
                else:
                    for i in tqdm(range(part_rows)):

                        smiles = smiles_list[split_idx[self.what][i]]
                        homolumogap = homolumogap_list[split_idx[self.what][i]]
                        graph = self.smiles2graph(smiles)
                
                        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
                        assert(len(graph['node_feat']) == graph['num_nodes'])

                        self.graphs.append(graph)
                        self.labels.append(homolumogap)

                    self.labels = np.array(self.labels)
                    print(self.labels)

                print('Saving...')
                torch.save({'graphs': self.graphs, 'labels': self.labels}, pre_processed_file_path, pickle_protocol=4)

        def get_idx_split(self):
            split_dict = torch.load(osp.join(self.folder, 'split_dict.pt'))
            return split_dict

        def __getitem__(self, idx):
 

            if isinstance(idx, (int, np.integer)):
                return self.graphs[idx], self.labels[idx]

            raise IndexError(
                'Only integer is valid index (got {}).'.format(type(idx).__name__))

        def __len__(self):

            return len(self.graphs)

        def __repr__(self):  # pragma: no cover
            return '{}({})'.format(self.__class__.__name__, len(self))
        
    train_data = PCQM4MDataset_cutted(root = 'dataset', smiles2graph = smiles2graph, only_smiles=False, part=args.part, what='train')
    print('train', len(train_data))
    valid_data = PCQM4MDataset_cutted(root = 'dataset', smiles2graph = smiles2graph, only_smiles=False, part=args.part, what='valid')
    print('valid', len(valid_data))
    test_data = PCQM4MDataset_cutted(root = 'dataset', smiles2graph = smiles2graph, only_smiles=False, part=args.part, what='test')
    print('test', len(test_data))
    
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
