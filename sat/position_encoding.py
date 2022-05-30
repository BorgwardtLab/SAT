import os
import pickle
import torch
from torch_scatter import scatter_add
import torch_geometric.utils as utils
import numpy as np


class PositionEncoding(object):
    def apply_to(self, dataset):
        dataset.abs_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.abs_pe_list.append(pe)

        return dataset


class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = utils.get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()


class RWEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        W0 = normalize_adj(graph.edge_index, num_nodes=graph.num_nodes).tocsc()
        W = W0
        vector = torch.zeros((graph.num_nodes, self.pos_enc_dim))
        vector[:, 0] = torch.from_numpy(W0.diagonal())
        for i in range(self.pos_enc_dim - 1):
            W = W.dot(W0)
            vector[:, i + 1] = torch.from_numpy(W.diagonal())
        return vector.float()


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)


POSENCODINGS = {
    'lap': LapEncoding,
    'rw': RWEncoding,
}
