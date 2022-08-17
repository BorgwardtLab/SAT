# -*- coding: utf-8 -*-
import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric import utils
from sat.models import GraphTransformer
from sat.data import GraphDataset
from sat.utils import count_parameters
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES


def load_args():
    parser = argparse.ArgumentParser(
        description='Model visualization: SAT vs transformer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--visu', action='store_true', help='perform visualization')
    parser.add_argument('--graph-idx', type=int, default=-1,
                        help='graph to interpret')
    parser.add_argument('--outpath', type=str, default='../logs_visu',
                        help='visualization output path')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    return args

def load_model(datapath, dataset, deg=None):
    model = torch.load(datapath, map_location=torch.device('cpu'))
    args, state_dict = model['args'], model['state_dict']
    model = GraphTransformer(in_size=dataset.num_features,
                             num_class=dataset.num_classes,
                             d_model=args.dim_hidden,
                             dim_feedforward=2*args.dim_hidden,
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             sparse=args.sparse,
                             gnn_type=args.gnn_type,
                             use_edge_attr=args.use_edge_attr,
                             num_edge_features=dataset.num_edge_features,
                             edge_dim=args.edge_dim,
                             k_hop=args.k_hop,
                             n_hidden_graphs=args.n_hidden_graphs,
                             size_hidden_graphs=args.size_hidden_graphs,
                             max_step=args.max_step,
                             se=args.se,
                             deg=deg,
                             in_embed=False,
                             edge_embed=False,
                             global_pool=args.global_pool)
    model.load_state_dict(state_dict)
    return model, args

def compute_attn(datapath, dataset, graph_idx):
    model, args = load_model(datapath, dataset)
    model.eval()

    graph = dataset[graph_idx]
    y_true = graph.y.squeeze().item()
    if y_true != 0:
        return None

    graph_dset = GraphDataset(dataset[[graph_idx]], degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)

    graph_loader = DataLoader(graph_dset, batch_size=1, shuffle=False)

    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(graph_dset)

    attns = []
    def get_attns(module, input, output):
        attns.append(output[1])
    for i in range(args.num_layers):
        model.encoder.layers[i].self_attn.register_forward_hook(get_attns)

    for g in graph_loader:
        with torch.no_grad():
            y_pred = model(g, return_attn=True)
            y_pred = y_pred.argmax(dim=-1)
            y_pred = y_pred.item()
    print('Ground truth: ', y_true, 'Prediction: ', y_pred)
    if y_pred != 0:
        return None

    attn = attns[-1].mean(dim=-1)[-1]
    return attn

def draw_graph_with_attn(
    graph,
    outdir,
    filename,
    nodecolor=["tag", "attn"],
    dpi=300,
    edge_vmax=None,
    args=None,
    eps=1e-6,
):
    if len(graph.edges) == 0:
        return
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    fig = plt.figure(figsize=(4*len(nodecolor), 4), dpi=dpi)

    node_colors = defaultdict(list)

    titles = {
        'tag': 'molecule',
        'attn1': 'SAT attn',
        'attn2': 'Transformer attn'
    }


    for i in graph.nodes():
        for key in nodecolor:
            node_colors[key].append(graph.nodes[i][key])

    vmax = {}
    cmap = {}
    for key in nodecolor:
        vmax[key] = 19
        cmap[key] = 'tab20'
        if 'attn' in key:
            vmax[key] = max(node_colors[key])
            cmap[key] = 'Reds'

    pos_layout = nx.kamada_kawai_layout(graph, weight=None)
    
    for i, key in enumerate(nodecolor):
        ax = fig.add_subplot(1, len(nodecolor), i+1)
        ax.set_title(titles[key], fontweight='bold')
        nx.draw(
            graph,
            pos=pos_layout,
            with_labels=False,
            font_size=4,
            node_color=node_colors[key],
            vmin=0,
            vmax=vmax[key],
            cmap=cmap[key],
            width=1.3,
            node_size=100,
            alpha=1.0,
        )
        if 'attn' in key:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap[key], norm=plt.Normalize(vmin=0, vmax=vmax[key]))
            sm._A = []
            plt.colorbar(sm, cax=cax)

    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    save_path = os.path.join(outdir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def main():
    all_args = load_args()
    data_path = '../datasets/TUDataset'

    dataset = datasets.TUDataset(data_path, 'Mutagenicity')

    if all_args.graph_idx < 0:
        sizes = [graph.num_nodes for graph in dataset]
        indices = np.argsort(sizes)[::-1]
        indices = [i for i in indices if sizes[i] <= 30][:50]
    else:
        indices = [all_args.graph_idx]

    for graph_idx in indices:
        print("Graph ", graph_idx)
        graph = dataset[graph_idx]

        print("Computing attention for SAT")
        attn1 = compute_attn('../pretrained_models/sat/Mutagenicity/model.pth', dataset, graph_idx=graph_idx)
        if attn1 is None:
            continue
        print("Computing attention for Transformer")
        attn2 = compute_attn('../pretrained_models/transformer/Mutagenicity/model.pth', dataset, graph_idx=graph_idx)
        if attn2 is None:
            continue

        graph.tag = graph.x.argmax(dim=-1)
        graph.attn1 = attn1
        graph.attn2 = attn2

        graph = utils.to_networkx(graph, node_attrs=['tag', 'attn1', 'attn2'], to_undirected=True)
        draw_graph_with_attn(
            graph, all_args.outpath,
            'graph{}.png'.format(graph_idx),
            nodecolor=['tag', 'attn1', 'attn2']
        )


if __name__ == "__main__":
    main()

