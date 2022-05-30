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
from timeit import default_timer as timer

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator

from torchvision import transforms
from utils import ASTNodeEncoder, get_vocab_mapping
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq


def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer on OGBG-CODE2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="ogbg-code2",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=4, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=4, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=256, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--abs-pe', type=str, default=None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=2, help="number of epochs for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', action='store_true', help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=128, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='gcn',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=2, help="number of layers for GNNs")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')

    parser.add_argument('--se', type=str, default="gnn", 
            help='Extractor type: khopgnn, or gnn')

    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction')

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
        pred_list = model(data)

        loss = 0
        for i in range(len(pred_list)):
            loss += criterion(pred_list[i].to(torch.float32), data.y_arr[:,i])

        loss = loss / len(pred_list)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, evaluator, arr_to_seq, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    seq_ref_list = []
    seq_pred_list = []

    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            pred_list = model(data)

            loss = 0
            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1))
                loss += criterion(pred_list[i].to(torch.float32), data.y_arr[:,i])
            mat = torch.cat(mat, dim = 1)
            loss = loss / len(pred_list)
            
            seq_pred = [arr_to_seq(arr) for arr in mat]

            seq_ref = [data.y[i] for i in range(len(data.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

            running_loss += loss.item() * size

    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    score = evaluator.eval({"seq_ref": seq_ref_list, "seq_pred": seq_pred_list})['F1']
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
    num_edge_features = 2

    dataset = PygGraphPropPredDataset(name=args.dataset, root=data_path)
    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target seqence less or equal to {} is {}%.'.format(
        args.max_seq_len,
        np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list))
    )

    split_idx = dataset.get_idx_split()


    ### building vocabulary for sequence predition. Only use training data.
    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)

    ### set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset.transform = transforms.Compose([
        augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)
    ])

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    print(nodeattributes_mapping)

    filter_mask = np.array([dataset[i].num_nodes for i in split_idx['train']]) <= 1000
    train_dset = GraphDataset(dataset[split_idx['train'][filter_mask]], degree=True,
            k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
            return_complete_index=False)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    print(len(train_dset))
    print(train_dset[0])

    filter_mask = np.array([dataset[i].num_nodes for i in split_idx['valid']]) <= 1000
    val_dset = GraphDataset(dataset[split_idx['valid'][filter_mask]], degree=True,
            k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
            return_complete_index=False)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)
    print(np.sum(filter_mask))
    print(len(split_idx['valid']))

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

    node_encoder = ASTNodeEncoder(
        args.dim_hidden,
        num_nodetypes = len(nodetypes_mapping['type']),
        num_nodeattributes = len(nodeattributes_mapping['attr']),
        max_depth = 20
    )

    model = GraphTransformer(in_size=node_encoder,
                             num_class=len(vocab2idx),
                             d_model=args.dim_hidden,
                             dim_feedforward=4*args.dim_hidden,
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
                             in_embed=True,
                             edge_embed=False,
                             max_seq_len=args.max_seq_len,
                             global_pool=args.global_pool)
    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)
    evaluator = Evaluator(name=args.dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # lr_scheduler = None
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup)

    # warmup_lr_scheduler = None
    lr_steps = args.lr / (args.warmup * len(train_loader))
    def warmup_lr_scheduler(s):
        lr = s * lr_steps
        return lr

    filter_mask = np.array([dataset[i].num_nodes for i in split_idx['test']]) <= 1000
    test_dset = GraphDataset(dataset[split_idx['test'][filter_mask]], degree=True,
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
        val_score, val_loss = eval_epoch(model, val_loader, criterion, evaluator, arr_to_seq, args.use_cuda, split='Val')
        test_score, test_loss = eval_epoch(model, test_loader, criterion, evaluator, arr_to_seq, args.use_cuda, split='Test')

        if epoch >= args.warmup and lr_scheduler is not None:
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
    test_score, test_loss = eval_epoch(model, test_loader, criterion, evaluator, arr_to_seq, args.use_cuda, split='Test')

    print("test auROC {:.4f}".format(test_score))

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
