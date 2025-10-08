import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable

import torch
from torch import Tensor
from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.data import NeighborSampler,Data

class EdgeIndex(NamedTuple):
    edge_index: Tensor
    delta_t: Optional[Tensor]
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        delta_t = self.delta_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, delta_t, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    delta_t: Optional[Tensor]
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        delta_t = self.delta_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, delta_t, e_id, self.size)

class GraphSampler(torch.utils.data.DataLoader):
   
    def __init__(self, datasets,node_idx,batch_size=128,size=[2,2],n_size=10,mode="train",**kwargs):
        
        self.batch_size = batch_size
        self.datasets = datasets
        self.sizes = size
        self.node_idx = node_idx
        self.mode = mode

        self.node_emb = self.datasets["x"][0] # entity and relation
        self.hyperedge_emb =self.datasets["x"][1] # hyperedge embeding

        self.x = torch.cat([self.node_emb, self.hyperedge_emb ], dim=0)

        self.label = datasets["labels"]  -  self.datasets["n_entity"] #  label 转为 从0 开始的 index， label 是关系的id

        self.num_nodes = self.datasets["n_entity"] + self.datasets["n_relation"]  + self.datasets["n_hyperedge"]
        self.n_node = self.datasets["n_relation"] +  self.datasets["n_entity"]

        self.traj2traj_edge_type = self.datasets["hyper_edge_type"]

        self.edge_index = self.datasets["edge_index"]
        # 超边之间的连接
        self.traj2traj_adj_t = SparseTensor(
            row= self.edge_index[1][0],
            col= self.edge_index[1][1],
            value=torch.arange(self.edge_index[1].size(1)),  # 超边之间
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        # 实体和超边之间的连接
        self.ci2traj_adj_t = SparseTensor(
            row=self.edge_index[0][0],
            col=self.edge_index[0][1],
            value=torch.arange(self.edge_index[0].size(1)),
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        self.neg_info = self.datasets["neg_info"]

        # 需要构建一个新的sampler， 增加了负采样的sample
        self.edge2neg = self.neg_info["pos_to_neg_edge"]
        self.n_size = n_size
        self.negative_sampler = NegtiveSampler(self.datasets,sizes = size)
        self.relation_sampler = RelationSampler(self.datasets,size = size )

        super(GraphSampler, self).__init__(node_idx.view(-1).tolist(), collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch):
        # print("sampler a hyperedge batch begin.....")
        sample_idx = [i - self.n_node for i in batch]  # 输入是超边的id，然后转为从0开始的index，这样才能获取到正确的label
        lable = self.label[sample_idx] 

        n_id = torch.tensor(batch, dtype=torch.long)   # 但是采样中心还是使用原来的 id，因为在整个图结构当中是这样的，不然采样会不正确
        adjs = [] 
        for i, size in enumerate(self.sizes):
            if i == len(self.sizes) - 1:
                # Sample ci2traj one-hop checkin relation
                adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = None
            else:
                # Sample traj2traj multi-hop relation
                old_nid = n_id
                adj_t, n_id = self.traj2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.traj2traj_edge_type[e_id]
                split_idx = len(n_id)

            size = adj_t.sparse_sizes()[::-1]
            adjs.append((adj_t, edge_attr,  edge_type, e_id, size))
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        input_x =  self.x[n_id]
        out = (n_id,input_x, adjs, lable,split_idx)

        neg_list = []
        for id_x in batch:
            n_id_list = np.random.choice(self.edge2neg[id_x],self.n_size,replace=True)
            neg_list.extend(n_id_list)
        
        neg_out = self.negative_sampler.sample(neg_list)    
        rel_out = self.relation_sampler.sample(lable)

        return out,neg_out,rel_out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

class NegtiveSampler():
   
    def __init__(self, 
                 datasets,size=[2,2],mode="train",**kwargs
                 ):
        
        self.datasets = datasets
        self.sizes = size
        self.mode = mode

        self.node_emb = self.datasets["x"][0] # entity and relation
        self.hyperedge_emb =self.datasets["x"][1]
        self.x = torch.cat([self.node_emb, self.hyperedge_emb ], dim=0)

        self.label = datasets["labels"]  -  self.datasets["n_entity"] # label 转为 从0 开始的 index
        
        self.num_nodes = self.datasets["n_entity"]+ self.datasets["n_relation"]  + self.datasets["n_hyperedge"]
        self.n_node = self.datasets["n_relation"]+  self.datasets["n_entity"]

        self.traj2traj_edge_type = self.datasets["hyper_edge_type"]

        self.edge_index = self.datasets["edge_index"]

        self.neg_info = self.datasets["neg_info"]
        # 实体和超边之间的连接
        self.ci2traj_adj_t = SparseTensor(
            row=self.edge_index[0][0],
            col=self.edge_index[0][1],
            value=torch.arange(self.edge_index[0].size(1)),
            sparse_sizes=(self.neg_info["max_node_id"], self.neg_info["max_node_id"])
        ).t()

        # 需要构建一个新的sampler， 增加了负采样的sample
        self.neg_traj2traj_adj_t = SparseTensor(
            row=self.neg_info["neg_edge_index"][0],
            col=self.neg_info["neg_edge_index"][1],
            value=torch.arange(self.neg_info["neg_edge_index"].size(1)),
            sparse_sizes=(self.neg_info["max_node_id"], self.neg_info["max_node_id"])
        ).t()

    # 传入一个 负采样的id 列表
    def sample(self, batch):
        lable = None 
        n_id = torch.tensor(batch, dtype=torch.long)   # 但是采样中心还是使用原来的 id，因为在整个图结构当中是这样的，不然采样会不正确
        adjs = [] 
        split_idx = None
        for i, size in enumerate(self.sizes):
            if i == len(self.sizes) - 1:
                adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = None
            else:
                old_nid = n_id
                adj_t, n_id = self.neg_traj2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = None
                split_idx = len(n_id)
            size = adj_t.sparse_sizes()[::-1]
            adjs.append((adj_t, edge_attr,  edge_type, e_id, size))
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        entity_list = list(n_id[split_idx:])
        entity_emb = self.x[n_id[split_idx:]]
        out = (n_id, entity_emb, adjs, lable, split_idx)
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

class RelationSampler(torch.utils.data.DataLoader):
   
    def __init__(self, 
                 datasets,batch_size=128,size=[10,2],mode="train",**kwargs
                 ):
        
        self.batch_size = batch_size
        self.datasets = datasets
        self.sizes = size
        self.mode = mode
        self.node_emb = self.datasets["x"][0] # entity and relation
        self.hyperedge_emb =self.datasets["x"][1]
        self.x = torch.cat([self.node_emb, self.hyperedge_emb ], dim=0)

        self.label = datasets["labels"]  -  self.datasets["n_entity"] # label 转为 从0 开始的 index
        
        self.num_nodes = self.datasets["n_entity"]+ self.datasets["n_relation"]  + self.datasets["n_hyperedge"]
        self.n_node = self.datasets["n_relation"]+  self.datasets["n_entity"]

        edge_index = self.datasets["e2rel_index"]
        # 超边之间的连接
        self.traj2traj_adj_t = SparseTensor(
            row= edge_index[0],
            col= edge_index[1],
            value=torch.arange(edge_index.size(1)),  # 超边之间
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        en2edge_index = self.datasets['edge_index'][0]
        # 实体和超边之间的连接
        self.ci2traj_adj_t = SparseTensor(
            row=en2edge_index[0],
            col=en2edge_index[1],
            value=torch.arange(en2edge_index.size(1)),
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()
        node_idx = [i for i in range(self.datasets["n_relation"])]
        batch_size = self.datasets["n_relation"]
        super(RelationSampler, self).__init__(node_idx, collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch):
        n_id = torch.tensor(batch, dtype=torch.long)   # 但是采样中心还是使用原来的 id，因为在整个图结构当中是这样的，不然采样会不正确
        adjs = [] 
        for i, size in enumerate(self.sizes):
            if i == len(self.sizes) - 1:
                adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = None
            else:
                # Sample traj2traj multi-hop relation
                old_nid = n_id
                adj_t, n_id = self.traj2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = None
                split_idx = len(n_id)
            size = adj_t.sparse_sizes()[::-1]
            adjs.append((adj_t, edge_attr,  edge_type, e_id, size))
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (n_id, self.x[n_id], adjs,split_idx)
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)