from util.dataloader import *
from util.sampler import *
from collections import defaultdict

import numpy as np
# 从基础的训练数据构建负采样的 dataloader
# 这里会生成一堆负样本 不能这么做
# 基于封闭假设，针对每个训练三元组生成制定数量的负样本，分别替换头实体和替换尾实体
# 对每个负三元组，构建超边，负三元组的超边和原来的超边之间建立连接，保证是单向的连接
# 
def build_neg_dataset(train, train2edgeid,edge_info, graph_info, n_entity, n_size):
    n_hyper_edge_begin = graph_info["edge_id_begin"]
    n_hyper_edge_end =  graph_info["max_train_hyper_edge_id"]
    hyper_edge_num = graph_info["hyper_edge_num"]  # 包括了测试超边在内的图
    neg_begin = hyper_edge_num + n_hyper_edge_begin
    train_set = set(train)

    triple2negTriple = defaultdict(list)
    triple2edge = {}
    edge2ent = defaultdict(set)

    for h,r,t in train_set:
        # 生成随机采样的样本
        neg_t =  np.random.randint(0, n_entity, n_size*2)
        for net in neg_t:
            if (h,r, net) in train_set: continue 
            # 真正的负样本
            triple2edge[(h,r,net)] = hyper_edge_num
            edge2ent[hyper_edge_num].add(h)
            edge2ent[hyper_edge_num].add(net)
            triple2negTriple[train2edgeid[(h,t)]].append(hyper_edge_num + n_hyper_edge_begin)
            hyper_edge_num += 1

        neg_h =  np.random.randint(0, n_entity, n_size*2)
        for neh in neg_h:
            if (h,r, neh) in train_set: continue 
            # 真正的负样本
            triple2edge[(neh,r,t)] = hyper_edge_num
            edge2ent[hyper_edge_num].add(h)
            edge2ent[hyper_edge_num].add(net)
            triple2negTriple[train2edgeid[(h,t)]].append(hyper_edge_num + n_hyper_edge_begin)
            hyper_edge_num += 1
    neg_end = hyper_edge_num + n_hyper_edge_begin
    # 生成负样本，并且给给负样本设置了超边的id，然后将负样本的超边和训练图连接起来
    
    edge_indx = edge_info["hypergraph_edge_index"]
    edge_type = edge_info["hypergraph_edge_type"]
    new_edge_index, new_edge_type = merge_neg2train(edge_indx,edge_type, edgeInfo,n_hyper_edge_begin,n_hyper_edge_end, neg_begin,neg_end,edge2ent )

    neg_info = {
        "triple2edge": triple2edge,
        "total_neg_size": neg_end - neg_begin,
        "neg_edge_index": new_edge_index,
        "neg_edge_type": new_edge_type,
        "pos_to_neg_edge": triple2negTriple
    }
    return neg_info




