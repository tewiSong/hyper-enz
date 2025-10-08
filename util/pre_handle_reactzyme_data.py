
import pickle
from collections import defaultdict
# from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix
import torch
import numpy as np
import json 

from data_pre_handle_utils import *
import os
# 处理酶和化合物的数据


def load_reactzype_data():
    # with open("../brenda_all_highf/train.json", "r") as f:
    #     train_data = json.load(f)
    # with open("../brenda_all_highf/valid.json", "r") as f:
    #     valid_data = json.load(f)
    # with open("../brenda_all_highf/test.json", "r") as f:
    #     test_data = json.load(f)
    # with open("../brenda_data/filter_data/train.json", "r") as f:
    #     train_data = json.load(f)
    # with open("../brenda_data/filter_data/valid.json", "r") as f:
    #     valid_data = json.load(f)
    # with open("../brenda_data/filter_data/test.json", "r") as f:
    #     test_data = json.load(f)
    with open("../reactzyme/result/train.json", "r") as f:
        train_data = json.load(f)
    with open("../reactzyme/result/valid.json", "r") as f:
        valid_data = json.load(f)
    with open("../reactzyme/result/test.json", "r") as f:
        test_data = json.load(f)
    return train_data, valid_data, test_data

train_data, valid_data, test_data = load_reactzype_data()
    
print(len(train_data ) +len(valid_data ) +len(test_data ))

def check_same_data(train, valid, test):
    train_set = set()
    for left, right, e in train:
        left = sorted(left)
        right = sorted(right)
        data  = tuple(left + right + [e])
        train_set.add(data)
    
    clean_valid = []
    for left, right, e in valid:
        left = sorted(left)
        right = sorted(right)
        data  = tuple(left + right + [e])
        if data not in train_set:
            clean_valid.append((left,right,e))
            
    clean_test = []
    for left, right, e in test:
        left = sorted(left)
        right = sorted(right)
        data  = tuple(left + right + [e])
        if data not in train_set:
            clean_test.append((left,right,e))
    return clean_valid, clean_test

valid_data,test_data = check_same_data(train_data,valid_data,test_data)

print(len(train_data ) +len(valid_data ) +len(test_data ))

cset,eset, e2id,c2id = build_dict_for_double_data(train_data)


c_num = len(cset)
e_num = len(eset)


# 开始构建超图
base_node_num = c_num + e_num
edge_id = 0
print("begin single_graph")
sing_graph, train_info =  build_single_graph(train_data,valid_data,test_data, c2id, e2id, base_node_num)

# 重新构建一个超图数据
# 首先需要将ko 的 id 和 层次lable 进行混合编码：
# 构造一个空白的embedding 作为 0（这个地方存疑）
graph_info = {
    "c_num": c_num,
    "e_num": e_num,
    "base_node_num":base_node_num,
    "train": train_data,
    "valid": valid_data,
    "test": test_data,
    "c2id": c2id,
    "e2id": e2id,
}
graph_info.update(sing_graph)

torch.save(graph_info,"../pre_handle_data/reactzyme/reactzyme_no_e_add_edge_type_graph_info.pkl")
torch.save(train_info,"../pre_handle_data/reactzyme/reactzyme_no_e_add_edge_type_train_info.pkl")
