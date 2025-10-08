import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.data import NeighborSampler,Data
from scipy.sparse import coo_matrix
from util.dataloader import NagativeSampleDataset,BidirectionalOneShotIterator
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import math

def generate_hyper_graph(train_triples, n_entity, n_node):
    # 构建基础的超图: 实体对作为超边,首先构建训练集的超图,超边的id也从0开始
    edge_id = 0

    head_id = []
    head_edge = []

    tail_id = []
    tail_edge = []

    r_id = []
    r_edge = []

    for h,r,t in train_triples:
        head_id.append(h)
        head_edge.append(edge_id)

        tail_id.append(t)
        tail_edge.append(edge_id) 

        r_id.append(r)
        r_edge.append(edge_id)

        edge_id += 1
    
    entity = head_id + tail_id
    edge = tail_id + tail_edge

    # 计算 训练集当中共享实体超边的数据
    edge2ent_maxtrix = coo_matrix((
            np.ones(len(entity)),
            (np.array(entity), np.array(edge))
        ),shape=(n_entity,edge_id)).tocsr()

    share_entity = edge2ent_maxtrix.T * edge2ent_maxtrix
    share_entity = share_entity.tocoo()

    share_entity_row = share_entity.row 
    share_entity_col = share_entity.col 
    edge_type_node = np.zeros_like(share_entity_row)

    edge_type_share_ent = torch.LongTensor(edge_type_node)
    
    edge_index_row_share_ent = torch.LongTensor(share_entity_row)
    edge_index_col_share_ent = torch.LongTensor(share_entity_col)
    
    edge_index = torch.stack([
        edge_index_row_share_ent + n_node,
        edge_index_col_share_ent + n_node
    ])
    print("Build Train edge connection finished......")
    # 超边和关系之间的连接

    row = torch.as_tensor(r_id, dtype=torch.long)
    col = torch.as_tensor(r_edge, dtype=torch.long) + n_node

    row_add = torch.cat([row,col],dim=-1)
    col_add = torch.cat([col,row],dim=-1)

    edge2rel_index = torch.stack([row_add,col_add])
    edge2rel_edge_type = np.ones_like(r_id)
    edge2rel_edge_type = torch.LongTensor(np.concatenate([edge2rel_edge_type, edge2rel_edge_type]))
    # edge2rel_edge_type = torch.LongTensor(edge2rel_edge_type)
    

    # 构建超边和头实体之间的关系

    row = torch.as_tensor(head_id, dtype=torch.long)
    col = torch.as_tensor(head_edge, dtype=torch.long)+ n_node

    row_add = torch.cat([row,col],dim=-1)
    col_add = torch.cat([col,row],dim=-1)

    edge2head_index = torch.stack([row_add,col_add])
    edge2head_edge_type = np.ones_like(head_id)
    edge2head_edge_type = torch.LongTensor(np.concatenate([edge2head_edge_type, edge2head_edge_type])) + 1

    row = torch.as_tensor(tail_id, dtype=torch.long)
    col = torch.as_tensor(tail_edge, dtype=torch.long)+ n_node

    row_add = torch.cat([row,col],dim=-1)
    col_add = torch.cat([col,row],dim=-1)


    edge2tail_index = torch.stack([row_add, col_add])
    edge2tail_edge_type = np.ones_like(head_id)
    edge2tail_edge_type = torch.LongTensor(np.concatenate([edge2tail_edge_type, edge2tail_edge_type])) + 2

    edge2ent_index = torch.cat([edge2head_index,edge2tail_index],dim=-1)
    edge2ent_type = torch.cat([edge2head_edge_type,edge2tail_edge_type],dim=-1)
    

    # print("node number: %d" % edge_id)
    # print("node number: %d" % n_node)

    base_data = {
        "edge_index": edge_index,
        "edge_type_share_ent": edge_type_share_ent,
        "edge2rel_index": edge2rel_index,
        "edge2rel_edge_type":edge2rel_edge_type,
        "edge2ent_index": edge2ent_index,
        "edge2ent_type": edge2ent_type,
        "max_edge_id" : edge_id + n_node
    }

    return base_data

class SimplerGraphSampler(torch.utils.data.DataLoader):
   
    def __init__(self,dataset_info,embedding,dataloader,size=[2,2],batch_size=1024,mode="train",**kwargs):
        
        self.dataloader = dataloader
        self.dataset = dataset_info
        self.n_node  =dataset_info["max_edge_id"] 
        self.node_emb = embedding
        self.batch_size = batch_size

        self.sizes = size

        self.edge2edge_index = SparseTensor(
            row= self.dataset["edge_index"][0],
            col= self.dataset["edge_index"][1],
            value=torch.arange(self.dataset["edge_index"].size(1)),  # 超边之间
            sparse_sizes=(self.n_node, self.n_node)
        ).t()

        self.edge2rel_index = SparseTensor(
            row= self.dataset["edge2rel_index"][0],
            col= self.dataset["edge2rel_index"][1],
            value=torch.arange(self.dataset["edge2rel_index"].size(1)),  # 超边之间
            sparse_sizes=(self.n_node, self.n_node)
        ).t()

        self.edge2ent_index = SparseTensor(
            row= self.dataset["edge2ent_index"][0],
            col= self.dataset["edge2ent_index"][1],
            value=torch.arange(self.dataset["edge2ent_index"].size(1)),  # 超边之间
            sparse_sizes=(self.n_node, self.n_node)
        ).t()

        self.edge2edge_type =  self.dataset["edge_type_share_ent"]
        self.edge2rel_type =  self.dataset["edge2rel_edge_type"]
        self.edge2ent_type =  self.dataset["edge2ent_type"]

        node_idx = torch.tensor([0,1])
        super(SimplerGraphSampler, self).__init__(node_idx.view(-1).tolist(), collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch):
        # 通过base dataloader 生成基础的数据和负采样
        # 对实体和关系进行图采样
        pos, negative_sample, subsampling_weight, mode = next(self.dataloader)
        head, relation, tail = torch.split(pos,[1,1,1],dim=1)
        rel_out = self.sample_relation(relation.view(-1).contiguous())

        ent = torch.cat([head,tail, negative_sample],dim=-1)
        ent = ent.view(-1)
        ent_out = self.sample_ent(ent)
        return rel_out, ent_out, subsampling_weight, mode
    
    def sampler_helper(self, positive_sample, negative_sample):
        head, relation, tail = torch.split(positive_sample,[1,1,1],dim=-1)
        relation = relation.view(-1).contiguous()
        rel_out = self.sample_relation(relation)

        ent = torch.cat([head,tail, negative_sample],dim=-1)
        ent = ent.view(-1)
       
        ent_out = self.sample_ent(ent)
        return rel_out, ent_out

    def sample_ent(self, n_id):
        assert n_id.is_contiguous()
        adjs = []
        for i, size in enumerate(self.sizes):
            if i == len(self.sizes)-1 :
                adj_t, n_id = self.edge2ent_index.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.edge2ent_type[e_id]
                split_idx = len(n_id)   
               
            elif i == 0:
                adj_t, n_id = self.edge2ent_index.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.edge2ent_type[e_id]
                split_idx = len(n_id)
                # print(n_id)
            else:
                # Sample traj2traj multi-hop relation
                old_nid = n_id
                adj_t, n_id = self.edge2edge_index.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.edge2edge_type[e_id]
                split_idx = len(n_id) 
                # print(n_id)
            
            size = adj_t.sparse_sizes()[::-1]  
            adjs.append((adj_t, edge_attr,  edge_type, e_id, size))  
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        input_x =  self.node_emb(n_id)
        out = (n_id, input_x, adjs, split_idx)
        return out

    def sample_relation(self, n_id):
        assert n_id.is_contiguous()
        adjs = []
        for i, size in enumerate(self.sizes):
            if i == len(self.sizes) - 1:
                adj_t, n_id = self.edge2ent_index.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.edge2ent_type[e_id]
             
            elif i == 0:
                # Sample traj2traj multi-hop relation
                adj_t, n_id = self.edge2rel_index.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.edge2rel_type[e_id]
                split_idx = len(n_id) 
              

            else:
                adj_t, n_id = self.edge2edge_index.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.edge2edge_type[e_id]
                split_idx = len(n_id)
                  
            size = adj_t.sparse_sizes()[::-1]  
            adjs.append((adj_t, edge_attr,  edge_type, e_id, size))  
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]

        input_x =  self.node_emb(n_id)
        out = (n_id, input_x, adjs, split_idx)
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


def build_dataset(dataset, config):

    graph_data = generate_hyper_graph(dataset.train, dataset.nentity, dataset.nentity+dataset.nrelation)

    graph_data["n_entity"] = dataset.nentity

    hr_t = DataLoader(NagativeSampleDataset(dataset.train, dataset.nentity, dataset.nrelation, config["n_size"], 'hr_t'),
                batch_size=config["batch_size"],
                shuffle=True, 
                num_workers=max(1, 4//2),
                collate_fn=NagativeSampleDataset.collate_fn
    )
    h_rt = DataLoader(NagativeSampleDataset(dataset.train, dataset.nentity, dataset.nrelation, config["n_size"], 'h_rt'),
                batch_size=config["batch_size"],
                shuffle=True, 
                num_workers=max(1, 4//2),
                collate_fn=NagativeSampleDataset.collate_fn
    )
    dataloader = BidirectionalOneShotIterator(hr_t, h_rt)
    embedding = nn.Embedding(graph_data["max_edge_id"], config["dim"])
    init_range =  6.0 / math.sqrt(config["dim"])
    nn.init.uniform_(embedding.weight, -init_range, init_range)
    sampler = SimplerGraphSampler(graph_data,embedding, dataloader, size=[2,2,2])
    return sampler, embedding