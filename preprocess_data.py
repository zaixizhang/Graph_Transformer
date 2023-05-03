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
from torch_geometric.datasets import Planetoid, Amazon, Actor
from torch.nn.functional import normalize
import torch_geometric.transforms as T
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_sparse import coalesce
from tqdm import tqdm


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
    name = 'pubmed'
    dataset = Planetoid(root='./data/', name=name)
    #dataset = Actor(root='./data/')
    data = dataset[0]
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
    normalized_adj = adj_normalize(adj)
    column_normalized_adj = column_normalize(adj)
    sp.save_npz('./dataset/'+name+'/normalized_adj.npz', normalized_adj)
    sp.save_npz('./dataset/' + name + '/column_normalized_adj.npz', column_normalized_adj)
    c = 0.15
    k1 = 15
    Samples = 8 # sampled subgraphs for each node
    power_adj_list = [normalized_adj]
    for m in range(5):
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])
    torch.save(data.x, './dataset/' + name + '/x.pt')
    torch.save(data.y, './dataset/' + name + '/y.pt')
    torch.save(data.edge_index, './dataset/' + name + '/edge_index.pt')

    sampling_matrix = c * inv((sp.eye(adj.shape[0]) - (1 - c) * normalized_adj).toarray()) # power_adj_list[1].toarray()
    feature = data.x

    #create subgraph samples
    data_list = []
    for id in tqdm(range(data.y.shape[0])):
        s = sampling_matrix[id]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-k1:]

        # use sampling matrix for node sampling
        # can use random sampling here
        s = sampling_matrix[id]
        s[id] = 0
        s = np.maximum(s, 0)
        sample_num1 = np.minimum(k1, (s > 0).sum())
        #create subgraph samples for ensemble
        sub_data_list = []
        for _ in range(Samples):
            if sample_num1 > 0:
                sample_index1 = np.random.choice(a=np.arange(data.y.shape[0]), size=sample_num1, replace=False, p=s/s.sum())
            else:
                sample_index1 = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index1, dtype=int), torch.tensor(top_neighbor_index[: k1-sample_num1], dtype=int)])
            # create attention bias (positional encoding)
            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])
            attn_bias = attn_bias.permute(1, 2, 0)

            label = data.y[node_feature_id].long()
            feature_id = node_feature_id
            assert len(feature_id) == k1+1
            sub_data_list.append([attn_bias, feature_id, label])
        data_list.append(sub_data_list)

    torch.save(data_list, './dataset/'+name+'/data.pt')
    torch.save(feature, './dataset/'+name+'/feature.pt')


def node_sampling(p=None):
    print('Sampling Nodes!')
    name = 'pubmed'
    edge_index = torch.load('./dataset/'+name+'/edge_index.pt')
    data_x = torch.load('./dataset/'+name+'/x.pt')
    data_y = torch.load('./dataset/'+name+'/y.pt')
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                shape=(data_y.shape[0], data_y.shape[0]), dtype=np.float32)
    normalized_adj = sp.load_npz('./dataset/'+name+'/normalized_adj.npz')
    column_normalized_adj = sp.load_npz('./dataset/' + name + '/column_normalized_adj.npz')
    c = 0.15
    k1 = 14
    Samples = 8 # sampled subgraphs for each node
    power_adj_list = [normalized_adj]
    for m in range(5):
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])

    sampling_matrix = c * inv((sp.eye(adj.shape[0]) - (1 - c) * column_normalized_adj).toarray()) # power_adj_list[0].toarray()
    feature = data_x

    #create subgraph samples
    data_list = []
    for id in range(data_y.shape[0]):
        s = sampling_matrix[id]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-k1:]

        s = sampling_matrix[id]
        s[id] = 0
        s = np.maximum(s, 0)
        sample_num1 = np.minimum(k1, (s > 0).sum())
        sub_data_list = []
        for _ in range(Samples):
            if sample_num1 > 0:
                sample_index1 = np.random.choice(a=np.arange(data_y.shape[0]), size=sample_num1, replace=False, p=s/s.sum())
            else:
                sample_index1 = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index1, dtype=int), torch.tensor(top_neighbor_index[: k1-sample_num1], dtype=int)])

            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])
            attn_bias = attn_bias.permute(1, 2, 0)

            sub_data_list.append([attn_bias, node_feature_id, data_y[node_feature_id].long()])
        data_list.append(sub_data_list)

    return data_list, feature


if __name__ == '__main__':
    # preprocess data
    process_data()



