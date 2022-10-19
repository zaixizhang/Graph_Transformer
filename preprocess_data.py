import numpy as np
import torch
import os.path as osp
import pickle
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
import scipy.sparse as sp
from numpy.linalg import inv
from torch_geometric.datasets import Planetoid, Amazon
from torch.nn.functional import normalize
import torch_geometric.transforms as T
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_sparse import coalesce
from sklearn.preprocessing import label_binarize
import scipy.io


def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def eigenvector(L):
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.tensor(EigVec[:, 1:11], dtype = torch.float32)


def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def process_data(p=None):
    name = 'cora'
    dataset = Planetoid(root='./data/', name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                shape=(data.y.shape[0], data.y.shape[0]),
                                dtype=np.float32)
    normalized_adj = adj_normalize(adj)
    c = 0.15
    k = 10
    Samples = 4 # sampled subgraphs for each node
    power_adj_list = [normalized_adj]
    for m in range(5):
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])

    eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * normalized_adj).toarray())

    #create subgraph samples
    data_list = []
    for id in range(data.y.shape[0]):
        sub_data_list = []
        s = eigen_adj[id]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-k:]
        s = eigen_adj[id]
        s[id] = 0
        s = np.maximum(s, 0)
        sample_num = np.minimum(k, (s > 0).sum())
        for _ in range(Samples):
            if sample_num > 0:
                sample_index = np.random.choice(a=np.arange(data.y.shape[0]), size=sample_num, replace=False, p=s/s.sum())
            else:
                sample_index = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index, dtype=int),
                                    torch.tensor(top_neighbor_index[: k-sample_num], dtype=int)])
            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])
            attn_bias = attn_bias.permute(1, 2, 0)

            label = data.y[node_feature_id]
            sub_data_list.append([attn_bias, node_feature_id, label])
        data_list.append(sub_data_list)

    feature = data.x
    torch.save(data_list, './dataset/'+name+'/data.pt')
    torch.save(feature, './dataset/'+name+'/feature.pt')


if __name__ == '__main__':
    process_data()



