# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, Linear, ReLU
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from typing import Optional, List, Dict
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree



GNN_TYPES = [
    'graph', 'graphsage', 'gcn',
    'gin', 'gine',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4',
    'rwgnn', 'khopgnn'
]

EDGE_GNN_TYPES = [
    'gine', 'gcn',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4'
]


def get_simple_gnn_layer(gnn_type, embed_dim, **kwargs):
    edge_dim = kwargs.get('edge_dim', None)
    if gnn_type == "graph":
        return gnn.GraphConv(embed_dim, embed_dim)
    elif gnn_type == "graphsage":
        return gnn.SAGEConv(embed_dim, embed_dim)
    elif gnn_type == "gcn":
        if edge_dim is None:
            return gnn.GCNConv(embed_dim, embed_dim)
        else:
            return GCNConv(embed_dim, edge_dim)
    elif gnn_type == "gin":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINConv(mlp, train_eps=True)
    elif gnn_type == "gine":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    elif gnn_type == "pna":
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=4, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna2":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                           aggregators=aggregators, scalers=scalers,
                           deg=deg, towers=4, pre_layers=1, post_layers=1,
                           divide_input=True, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna2_ram":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = PNAConv_towers(embed_dim, embed_dim,
                    aggregators=aggregators, scalers=scalers,
                    deg=deg, towers=4, divide_input=True, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna3":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=1, pre_layers=1, post_layers=1,
                            divide_input=False, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna4":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=8, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer


    elif gnn_type == "mpnn":
        aggregators = ['sum']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=4, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer
    else:
        raise ValueError("Not implemented!")


class GCNConv(gnn.MessagePassing):
    def __init__(self, embed_dim, edge_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Embedding(1, embed_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = nn.Linear(edge_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = utils.degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
