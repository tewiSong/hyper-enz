
import pickle
from collections import defaultdict
# from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix
import torch
import numpy as np
import json 
import tqdm

def build_dict_for_single_data(data):
    cset = set()
    eset = set()
    for c_list, e in data:
        for c in c_list:
            cset.add(c)
        eset.add(e)

    cset = sorted(list(cset))
    eset = sorted(list(eset))

    c2id = {
        cset[i]:i for i in range(len(cset))
    }

    e2id = {
        eset[i]: i+ len(cset) for i in range(len(eset))
    }
    return cset,eset, e2id,c2id

def build_dict_for_double_data(data):
    cset = set()
    eset = set()
    for left,right, e in data:
        for c in left:
            cset.add(c)
        for c in right:
            cset.add(c)
        eset.add(e)
    cset = sorted(list(cset))
    eset = sorted(list(eset))

    c2id = {
        cset[i]:i for i in range(len(cset))
    }

    e2id = {
        eset[i]: i+len(cset) for i in range(len(eset))
    }
    return cset,eset, e2id,c2id


def build_hyedge(data, c2id, e2id,clist2edgeId):
    hyId_list = []
    hy_c_node = []
    hy_c_edge = []

    hy_e_node = []
    hy_e_edge = []

    hy2e = {}

    for c_list, e in data:
        # 化合物到超边
        edge_id = clist2edgeId[c_list]
        for c in list(c_list):
            hy_c_node.append(c2id[c])
            hy_c_edge.append(edge_id)
        # 酶到超边
        hy_e_node.append(e2id[e])
        hy_e_edge.append(edge_id)

        hy2e[edge_id]= e2id[e]
        hyId_list.append(edge_id)
     
    c2Hyper_index =  [hy_c_node,hy_c_edge ]
    e2Hyper_index =  [hy_e_node,hy_e_edge ]

    return hyId_list, c2Hyper_index, e2Hyper_index, hy2e


def build_share_node_index(node_id, edge_id, edge_id_start, max_node_edge, max_edge_id):
    edge_id = [e - edge_id_start for e in edge_id] # id 的下标为0

    edge2entity_train = coo_matrix((
        np.ones(len(node_id)),
        (np.array(node_id), np.array(edge_id))), shape=(max_node_edge, max_edge_id)).tocsr()

    edge2edge_train = edge2entity_train.T * edge2entity_train
    share_entity = edge2edge_train.tocoo()

    share_entity_row_entity = torch.LongTensor(share_entity.row) + edge_id_start
    share_entity_col_entity = torch.LongTensor(share_entity.col) + edge_id_start

    return share_entity_row_entity, share_entity_col_entity,edge2entity_train

# 这里修改共享酶的超边之间的连接：
def share_e(node_list, edge_list, c_num):
    e2edgeSet = defaultdict(set)
    for i in range(len(node_list)):
        e2edgeSet[node_list[i]].add(edge_list[i])

    node1 = []
    node2 = []
    edge_type = []
   
    count = 0
    for e in e2edgeSet.keys():
        edge_list = list(e2edgeSet[e])
        for i in range(len(edge_list)):
            for j in range(i+1, len(edge_list)):
                edge1 = edge_list[i]
                edge2 = edge_list[j]

                node1.append(edge1)
                node2.append(edge2)

                node1.append(edge2)
                node2.append(edge1)
                edge_type.append(e - c_num)
                edge_type.append(e - c_num)
                count +=1 
                if count % 10000000 == 0: 
                    print("count : %d" % count)

    node1, node2, edge_type  = torch.LongTensor(node1),torch.LongTensor(node2),torch.LongTensor(edge_type)
    return node1, node2, edge_type


def transe_single2id(datas, e2id, clist2edgeId, edge_id_start):
    train_triples = []
    single_train = []
    edge_id = edge_id_start

    for left, right,e in datas:
        left = tuple(sorted(left))
        right = tuple(sorted(right))

        if left not in clist2edgeId:
            clist2edgeId[left] = edge_id
            edge_id += 1

        if right not in clist2edgeId:
            clist2edgeId[right] = edge_id
            edge_id += 1
        
        train_triples.append((clist2edgeId[left], e2id[e], clist2edgeId[right]))

        single_train.append((left,e))
        single_train.append((right,e))
    return train_triples, single_train, clist2edgeId, edge_id

def build_single_graph(train_data,valid_data, test_data, c2id, e2id, edge_start_id):
    '''
        data: left, right e 组成的数据利润表

    '''
    c_num = len(c2id)
    e_num = len(e2id)
    single_train = []

    clist2edgeId = dict()
    edge_id = edge_start_id
    print("edge start id: ", edge_id)
    
    train_triples, single_train, clist2edgeId, edge_id = transe_single2id(train_data, e2id, clist2edgeId,edge_id)
    valid_triples, single_valid, clist2edgeId, edge_id = transe_single2id(valid_data, e2id, clist2edgeId,edge_id)
    test_triples, single_test, clist2edgeId, edge_id = transe_single2id(test_data, e2id, clist2edgeId,edge_id)

    print("edge end id : ", edge_id)

    # 超边的数量
    hyedge_num = len(clist2edgeId)
    # 记录超边的列表

    # 实体到超边的连接
    train_id_list, train_c2Hy_index, train_e2Hy_index, Hy2E = build_hyedge(single_train, c2id, e2id,clist2edgeId)
    valid_id_list, valid_c2Hy_index, valid_e2Hy_index, validHy2E = build_hyedge(single_valid, c2id, e2id,clist2edgeId)
    test_id_list, test_c2Hy_index, test_e2Hy_index, testHy2E = build_hyedge(single_test, c2id, e2id,clist2edgeId)

    # 构建实体到超边的连接
    # 全部的超边连接
    entity_edge = train_c2Hy_index[1] + valid_c2Hy_index[1] + test_c2Hy_index[1]
    entity = train_c2Hy_index[0] + valid_c2Hy_index[0] + test_c2Hy_index[0]
    v2e_all_index = torch.stack([torch.as_tensor(entity, dtype=torch.long), torch.as_tensor(entity_edge, dtype=torch.long)])
    e2v_all_index = torch.stack([torch.as_tensor(entity_edge, dtype=torch.long), torch.as_tensor(entity, dtype=torch.long)])


    # 只有训练集当中的连接
    entity_edge = train_c2Hy_index[1] 
    entity = train_c2Hy_index[0] 
    v2e_train_index = torch.stack([torch.as_tensor(entity_edge, dtype=torch.long), torch.as_tensor(entity, dtype=torch.long)])

    # 超边共享化合物的连接
    share_entity_row_entity, share_entity_col_entity, edge2entity_train \
        = build_share_node_index( train_c2Hy_index[0],train_c2Hy_index[1],
                                    edge_start_id, c_num, hyedge_num
                                    )
    left_edge_set = set()
    right_edge_set = set()
    for left,e,right in train_triples:
        left_edge_set.add(left)
        right_edge_set.add(right)

    edge_type_node = []

    print("计算共享化合物的超边类型: ", len(share_entity_row_entity))
    for i in tqdm.tqdm(range(len(share_entity_row_entity))):
        id1 = share_entity_row_entity[i].item()
        id2 = share_entity_col_entity[i].item()
        if id1 in left_edge_set and id2 in left_edge_set:
            edge_type_node.append(e_num)
        elif id1 in right_edge_set and id2 in right_edge_set:
            edge_type_node.append(e_num+1)
        elif id1 in right_edge_set and id2 in left_edge_set:
            edge_type_node.append(e_num+2)
        elif id1 in left_edge_set and id2 in right_edge_set:
            edge_type_node.append(e_num+2)
        else:
            print("error")
    print("超边类型3种")

    edge_type_node = torch.LongTensor(edge_type_node)

    
    # 共享酶之间的超边
    # shareE_node1 , shareE_node, shareE_edge_type = share_e(train_e2Hy_index[0],train_e2Hy_index[1], c_num)

    # 构建完成训练集当中所有超边的连接
    edge_index_row = torch.cat([share_entity_row_entity],dim=-1)
    edge_index_col = torch.cat([share_entity_col_entity],dim=-1)
    edge_type_train = torch.cat([edge_type_node],dim=-1)
    edge_index_train = torch.stack([
        edge_index_row ,
        edge_index_col
    ])

    # 把测试集和训练图连接起来：通过共享化合物来实现
    val_c_node_temp =  valid_c2Hy_index[0] + test_c2Hy_index[0]
    val_r_edge_temp = [c - edge_start_id for c in list(valid_c2Hy_index[1] + test_c2Hy_index[1])]

    edge2entity_valid = coo_matrix((
        np.ones(len(val_c_node_temp)),
        (np.array(val_c_node_temp), np.array(val_r_edge_temp))), shape=(c_num, hyedge_num)).tocsr()
    print(edge2entity_valid.shape)
    valid2train_edge = edge2entity_valid.T * edge2entity_train
    print(valid2train_edge.shape)

    share_entity = valid2train_edge.tocoo()
    share_entity_row = torch.LongTensor(share_entity.row)
    share_entity_col = torch.LongTensor(share_entity.col)
    edge_type_node_valid = []
    print("build valid to train")
    for left,e,right in valid_triples:
        left_edge_set.add(left)
        right_edge_set.add(right)

    for left,e,right in test_triples:
        left_edge_set.add(left)
        right_edge_set.add(right)
    count = 0
    print("计算共享化合物的超边类型: ", len(share_entity_row))
    for i in  tqdm.tqdm(range(len(share_entity_row))):
        id1 = share_entity_row[i].item()+ edge_start_id
        id2 = share_entity_col[i].item()+ edge_start_id
        if id1 in left_edge_set and id2 in left_edge_set:
            edge_type_node_valid.append(e_num)
        elif id1 in right_edge_set and id2 in right_edge_set:
            edge_type_node_valid.append(e_num+1)
        elif id1 in right_edge_set and id2 in left_edge_set:
            edge_type_node_valid.append(e_num+2)
        elif id1 in left_edge_set and id2 in right_edge_set:
            edge_type_node_valid.append(e_num+2)
        else:
            count += 1
    
    print("build valid error: %d total : %d" % (count, len(share_entity_row)))

    # edge_type_node_valid = torch.LongTensor(np.zeros_like(share_entity_row)) + e_num
    edge_type_node_valid = torch.LongTensor(edge_type_node_valid) 

    edge_index_valid = torch.stack([
        share_entity_row + edge_start_id,  # valid 
        share_entity_col + edge_start_id   # train 
    ])

    new_edge_index = torch.cat((edge_index_train, edge_index_valid),dim=-1)
    new_edge_type = torch.cat((edge_type_train, edge_type_node_valid),dim=-1)

    sing_graph = {
        "train_edge_index": edge_index_train,
        "train_edge_type": edge_type_train,

        "valid_edge_index": new_edge_index,
        "valid_edge_type": new_edge_type,

        "v2e_all_index": v2e_all_index,
        "v2e_train_index": v2e_train_index,
        "e2v_all_index": e2v_all_index,
        "max_edge_id":edge_start_id + hyedge_num,
        "base_node_num":edge_start_id,
        "clist2edgeId":clist2edgeId
    }
    train_info = {
        "train_id_list": train_id_list,
        "valid_id_list": valid_id_list,
        "test_id_list": test_id_list,

        "train_triple": train_triples,
        "valid_triple": valid_triples,
        "test_triple": test_triples,     
        # "edgeid2label": Hy2E,
    }
    return sing_graph, train_info