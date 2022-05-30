# -*- coding: utf-8 -*-
import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils
from sat.models import GraphTransformer
from sat.data import GraphDataset
from sat.utils import count_parameters
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES
from sat.utils import add_zeros, extract_node_feature
from timeit import default_timer as timer

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator


def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer on OGBG-PPA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=3, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=128, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--abs-pe', type=str, default=None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=10, help="number of epochs for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', action='store_true', help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=128, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='graph',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=2, help="number of layers for GNNs")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--se', type=str, default="gnn", 
            help='Extractor type: khopgnn, or gnn')

    parser.add_argument('--aggr', type=str, default='add',
                        help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
    parser.add_argument('--not_extract_node_feature', action='store_true')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        pedir = 'None' if args.abs_pe is None else '{}_{}'.format(args.abs_pe, args.abs_pe_dim)
        outdir = outdir + '/{}'.format(pedir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0

    tic = timer()
    for i, data in enumerate(loader):
        size = len(data.y)
        if epoch < args.warmup:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.abs_pe == 'lap':
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(data.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.abs_pe = data.abs_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, data.y.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    y_pred = []
    y_true = []

    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            output = model(data)
            loss = criterion(output, data.y.squeeze())
            
            y_true.append(data.y.cpu())
            y_pred.append(output.argmax(dim=-1).view(-1, 1).cpu())

            running_loss += loss.item() * size

    toc = timer()
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    evaluator = Evaluator(name=args.dataset)
    score = evaluator.eval({'y_pred': y_pred,
                         'y_true': y_true})['acc']
    print('{} loss: {:.4f} score: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, score, toc - tic))
    return score, epoch_loss


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = '../datasets'
    num_edge_features = 7

    if args.not_extract_node_feature:
        transform = add_zeros
        input_size = 1
    else:
        from functools import partial
        transform = partial(extract_node_feature, reduce=args.aggr)
        input_size = num_edge_features

    dataset = PygGraphPropPredDataset(name=args.dataset, root=data_path,
                                      transform=transform)
    split_idx = dataset.get_idx_split()

    train_dset = GraphDataset(dataset[split_idx['train']], degree=True,
        k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
        return_complete_index=False)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    print(len(train_dset))
    print(train_dset[0])

    val_dset = GraphDataset(dataset[split_idx['valid']], degree=True,
        k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
        return_complete_index=False)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dset)
            abs_pe_encoder.apply_to(val_dset)

    if 'pna' in args.gnn_type or args.gnn_type == 'mpnn':
        deg = torch.cat([
            utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_dset])
    else:
        deg = None
    print(deg)

    model = GraphTransformer(in_size=input_size,
                             num_class=dataset.num_classes,
                             d_model=args.dim_hidden,
                             dim_feedforward=2*args.dim_hidden,
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             gnn_type=args.gnn_type,
                             k_hop=args.k_hop,
                             use_edge_attr=args.use_edge_attr,
                             num_edge_features=num_edge_features,
                             edge_dim=args.edge_dim,
                             se=args.se,
                             deg=deg,
                             in_embed=False,
                             edge_embed=False,
                             global_pool=args.global_pool)
    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup)

    lr_steps = args.lr / (args.warmup * len(train_loader))
    def warmup_lr_scheduler(s):
        lr = s * lr_steps
        return lr

    test_dset = GraphDataset(dataset[split_idx['test']], degree=True,
        k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
        return_complete_index=False)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(test_dset)

    print("Training...")
    best_val_loss = float('inf')
    best_val_score = 0
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, warmup_lr_scheduler, epoch, args.use_cuda)
        val_score, val_loss = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val')
        test_score, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

        if epoch >= args.warmup:
            lr_scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['val_score'].append(val_score)
        logs['test_score'].append(test_score)
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    total_time = timer() - start_time
    print("best epoch: {} best val score: {:.4f}".format(best_epoch, best_val_score))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_score, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

    print("test ACC {:.4f}".format(test_score))

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(args.outdir + '/logs.csv')
        results = {
            'test_score': test_score,
            'test_loss': test_loss,
            'val_score': best_val_score,
            'val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results.csv',
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model.pth')


if __name__ == "__main__":
    main()
