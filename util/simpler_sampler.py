import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.data import NeighborSampler,Data



class SimplerGraphSampler(torch.utils.data.DataLoader):
   
    def __init__(self,dataset_info,dataloader,node_idx,size=[2,2],mode="train",**kwargs):
        
        self.dataloader = dataloader

        self.edge2edge_index = SparseTensor(
            row= edge_index[0],
            col= edge_index[1],
            value=torch.arange(edge_index.size(1)),  # 超边之间
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        self.edge2rel_index = SparseTensor(
            row= edge_index[0],
            col= edge_index[1],
            value=torch.arange(edge_index.size(1)),  # 超边之间
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        self.edge2ent_index = SparseTensor(
            row= edge_index[0],
            col= edge_index[1],
            value=torch.arange(edge_index.size(1)),  # 超边之间
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        self.edge2edge_type = 
        self.edge2rel_type = 
        self.edge2ent_type = 

        self.n_entity 
        self.n_relation
        self.n_node 


        node_idx = torch.tensor([0,1])
        super(SimplerGraphSampler, self).__init__(node_idx.view(-1).tolist(), collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch):
        # 通过base dataloader 生成基础的数据和负采样
        # 对实体和关系进行图采样

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
        input_x =  self.node_emb[n_id[split_idx:]]
        out = (n_id,input_x, adjs, lable,split_idx)
        rel_pos_out = self.relation_sampler.sample(lable)

        neg_list = self.gen_neg_rel(batch)
        negative_sample = np.concatenate(neg_list)
        rel_neg_out = self.relation_sampler.sample(negative_sample)
        return out,rel_pos_out,rel_neg_out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
