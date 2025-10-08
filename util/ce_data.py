import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.data import NeighborSampler,Data
import math

def load_data():

    # graph_info = torch.load("/home/skl/yl/ce_project/relation_cl/pre_handle_data/ce_data_single_from_train_graph_info.pkl")
    # train_info = torch.load("/home/skl/yl/ce_project/relation_cl/pre_handle_data/ce_data_single_from_train_train_info.pkl")
    graph_info = torch.load("/home/skl/yl/ce_project/relation_cl/pre_handle_data/ce_data_single_graph_info.pkl")
    train_info = torch.load("/home/skl/yl/ce_project/relation_cl/pre_handle_data/ce_data_single_train_info.pkl")
    return graph_info, train_info

def build_graph_sampler(config):
    graph_info, train_info = load_data()
    base_node_num = graph_info["base_node_num"]

    node_emb = nn.Embedding(base_node_num, config["dim"])
    graph_info["node_emb"] = node_emb.weight

    init_range =  6.0 / math.sqrt(config["dim"])
    nn.init.uniform_(node_emb.weight, -init_range, init_range)

    sampler = CEGraphSampler(graph_info, train_info, 
        train_info["train_id_list"],
            batch_size=config["batch_size"],
            size=config["neibor_size"],
            n_size=config["n_size"])
    
    valid_sampler = CEGraphSampler(graph_info, train_info, 
        train_info["valid_id_list"],
            batch_size=16,
            size=[50,15],
             mode="valid"
            )
    test_sampler = CEGraphSampler(graph_info, train_info, 
        train_info["test_id_list"],
            batch_size=16,
            size=[50,15],
             mode="valid"
            )
    return sampler,valid_sampler, test_sampler,graph_info,train_info,node_emb
# graph_info = {
#     "train_edge_index": edge_index_train,
#     "train_edge_type": edge_type_train,
#     "valid_edge_index": new_edge_index,
#     "valid_edge_type": new_edge_type,
#     "node2edge_index": v2e_index,
#     "edge2rel_index": e2r_index,
#     "c_num": c_num,
#     "e_num": e_num,
#     "max_train_id": max_train_num,
#     "max_edge_id":edge_id,
# }

# train_info = {
#     "train_id_list": train_id_list,
#     "valid_id_list": valid_id_list,
#     "test_id_list": test_id_list,
#     "edgeid2label": edgeid2label,
#     "edgeid2true_train": edgeid2true_train,
#     "edgeid2true_all": edgeid2true_train,
# }

class CEGraphSampler(torch.utils.data.DataLoader):
   
    def __init__(self, graph_info,train_info,node_idx,batch_size=128,size=[2,2],n_size=10,mode="train",**kwargs):
        
        self.batch_size = batch_size
        self.graph_info = graph_info
        self.train_info = train_info

        self.sizes = size
        self.node_idx = node_idx
        self.mode = mode
        self.node_emb = self.graph_info["node_emb"]

        edgeid2label_list = []
        for key in self.train_info["edgeid2label"]:
            edgeid2label_list.append([key,self.train_info["edgeid2label"][key]])
        sorted(edgeid2label_list,key=lambda x:x[0])
        labels = [x[1] for x in edgeid2label_list]
        self.label = torch.LongTensor(labels) # - self.graph_info["c_num"]
        self.n_size = n_size
        
        if mode == "train":
            self.mask_true = self.train_info["edgeid2true_train"]
        else:
            self.mask_true = self.train_info["edgeid2true_all"]
            self.n_size = 5000

        self.num_nodes = self.graph_info["max_edge_id"]

        self.n_node = self.graph_info["base_node_num"]
        self.e_num = self.graph_info["e_num"]
        self.c_num = self.graph_info["c_num"]

        if mode == 'train':
            print("train graph")
            self.edge_index = self.graph_info["train_edge_index"]
            self.traj2traj_edge_type = self.graph_info["train_edge_type"]
        else:
            self.edge_index = self.graph_info["valid_edge_index"]
            self.traj2traj_edge_type = self.graph_info["valid_edge_type"]

        # 超边之间的连接
        self.traj2traj_adj_t = SparseTensor(
            row= self.edge_index[0],
            col= self.edge_index[1],
            value=torch.arange(self.edge_index.size(1)),  # 超边之间
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        self.e2v_index = self.graph_info["node2edge_index"]

        # 实体和超边之间的连接
        self.ci2traj_adj_t = SparseTensor(
            row=self.e2v_index[0],
            col=self.e2v_index[1],
            value=torch.arange(self.e2v_index.size(1)),
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        # 需要构建一个新的sampler， 增加了负采样的sample
        node_idx = torch.tensor(node_idx)
        print("sampler range:")
        print(torch.min(node_idx))
        print(torch.max(node_idx))
        print("*********************")
        self.relation_sampler = RelationSampler(self.graph_info,self.train_info,size = size )
        super(CEGraphSampler, self).__init__(node_idx.view(-1).tolist(), collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch):
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
        # rel_pos_out = self.relation_sampler.sample(lable,self.mode)
        out = (n_id,input_x, adjs, lable - self.c_num,split_idx)
        # neg_list = self.gen_neg_rel(batch)
        # negative_sample = np.concatenate(neg_list)
        # rel_neg_out = self.relation_sampler.sample(negative_sample,self.mode)
        rel_pos_out,rel_neg_out = None, None
        return out,rel_pos_out,rel_neg_out
    
    def gen_neg_rel(self, n_ids):
        neg_list = []
        for edge_id in n_ids:
            if self.mode == "train":
                negative_sample_list = []
                negative_sample_size = 0
                while negative_sample_size < self.n_size:
                    n_rel_id = np.random.randint(self.c_num,self.c_num+self.e_num, self.n_size*2)
                    mask = np.in1d(
                            n_rel_id, 
                            list(self.mask_true[edge_id]), 
                            assume_unique=True, 
                            invert=True
                        )
                    n_rel_id = n_rel_id[mask] # filter true triples
                    negative_sample_size += n_rel_id.size
                    negative_sample_list.append(n_rel_id)
                    if negative_sample_size >= self.n_size:
                        break
                negative_sample = np.concatenate(negative_sample_list)[:self.n_size]
            else:
                negative_sample = [i + self.c_num for i in range(self.e_num)]
            neg_list.append(negative_sample)
        return neg_list

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

class RelationSampler(torch.utils.data.DataLoader):
   
    def __init__(self, 
                 graph_info,train_info,batch_size=128,size=[10,2],mode="train",**kwargs
                 ):
        
        self.graph_info = graph_info
        self.train_info = train_info
        self.sizes = size
        self.mode = mode
        self.batch_size = batch_size
        self.node_emb = self.graph_info["node_emb"] # entity and relation

        edge_index = self.graph_info["edge2rel_index"]
        self.num_nodes = self.graph_info["max_edge_id"]
        # 超边之间的连接
        self.traj2traj_adj_t = SparseTensor(
            row= edge_index[0],
            col= edge_index[1],
            value=torch.arange(edge_index.size(1)),  # 超边之间
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        en2edge_index =self.graph_info["node2edge_index"]
        # 实体和超边之间的连接
        self.ci2traj_adj_t = SparseTensor(
            row=en2edge_index[0],
            col=en2edge_index[1],
            value=torch.arange(en2edge_index.size(1)),
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()
    
        attr2e_index =self.graph_info["attr2e_index"]
        # 实体和超边之间的连接
        self.attr2e_index = SparseTensor(
            row=attr2e_index[0],
            col=attr2e_index[1],
            value=torch.arange(attr2e_index.size(1)),
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()
        node_idx  = torch.tensor([0])
        super(RelationSampler, self).__init__(node_idx, collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch, mode):
        # if mode != "train":
            # print("rel samper")
            # print(batch)
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
        out = (n_id, self.node_emb[n_id[split_idx:]], adjs,split_idx)
        attr_out = self.sampler_attr(batch)

        return out,attr_out

    def sampler_attr(self, batch):
        adjs = []
        n_id = torch.tensor(batch, dtype=torch.long)
        split_idx = len(n_id)
        # print("begin")
        # print(n_id)

        adj_t, n_id = self.attr2e_index.sample_adj(n_id, self.sizes[0], replace=False)
        row, col, e_id = adj_t.coo()
        edge_attr = None
        edge_type = None
        # print(n_id)
        size = adj_t.sparse_sizes()[::-1]
        adjs.append((adj_t,edge_attr,edge_type, e_id, size))
        out = (n_id, self.node_emb[n_id[split_idx:]], adjs,split_idx)
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
