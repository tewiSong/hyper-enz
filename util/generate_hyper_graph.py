from scipy.sparse import coo_matrix
from torch_sparse import SparseTensor
from torch_geometric.data import Data
import torch
from collections import defaultdict 
import numpy as np

from scipy.sparse import coo_matrix

# 提前完成负采样并将负样本的边和训练集的边进行连接
def build_neg_dataset(train, train2edgeid,edge_info, graph_info, n_entity, n_size):
    n_hyper_edge_begin = graph_info["edge_id_begin"]
    n_hyper_edge_end =  graph_info["max_edge_id_train"]

    hyper_edge_num = graph_info["hyper_edge_num"]  # 包括了测试超边在内的图
    base_hyper_edge_num = hyper_edge_num
    neg_begin = hyper_edge_num + n_hyper_edge_begin
    train_set = set(train)

    triple2negTriple = defaultdict(list)
    triple2edge = {}
    edge2ent = defaultdict(set)
    print("train set size: %d" % len(train_set))
    
    for h,r,t in train_set:
        # 生成随机采样的样本
        neg_t =  np.random.randint(0, n_entity, n_size*2)
        pos_edge_id = train2edgeid[(h,r,t)]
        for net in neg_t:
            if (h,r, net) in train_set: continue 
            # 真正的负样本
            triple2edge[(h,r,net)] = hyper_edge_num
            edge2ent[hyper_edge_num].add(h)
            edge2ent[hyper_edge_num].add(net)
            triple2negTriple[pos_edge_id].append(hyper_edge_num + n_hyper_edge_begin)
            hyper_edge_num += 1

        neg_h =  np.random.randint(0, n_entity, n_size*2)
        for neh in neg_h:
            if (neh,r, t) in train_set: continue 
            # 真正的负样本
            triple2edge[(neh,r,t)] = hyper_edge_num
            edge2ent[hyper_edge_num].add(neh)
            edge2ent[hyper_edge_num].add(t)
            triple2negTriple[pos_edge_id].append(hyper_edge_num + n_hyper_edge_begin)
            hyper_edge_num += 1

    neg_end = hyper_edge_num + n_hyper_edge_begin
    # 生成负样本，并且给给负样本设置了超边的id，然后将负样本的超边和训练图连接起来
    
    edge_indx = edge_info["hypergraph_edge_index"]
    edge_type = edge_info["hypergraph_edge_type"]
    print("generate neg sampler finished!.......")
    print("start to merge to train graph")
    new_edge_index, new_edge_type = merge_neg2train(edge_indx,edge_type,edge_info,n_hyper_edge_begin,n_hyper_edge_end, neg_begin,neg_end,edge2ent,n_entity,base_hyper_edge_num)

    neg_info = {
        "triple2edge": triple2edge,
        "total_neg_size": neg_end - neg_begin,
        "neg_edge_index": new_edge_index,
        "neg_edge_type": new_edge_type,
        "pos_to_neg_edge": triple2negTriple,
        "max_node_id": neg_end
    }
    return neg_info

# 构建超图
def build_triple_entity_as_edge(train_triples, n_entity, n_node, valid, test,modelConfig, n_size= 100):

    load_from_file = True
    if load_from_file:
            print("save datasets info: %s" % modelConfig['dataset'])
            base_path = "/home/skl/yl/relation_cl/pre_handle_data/"
            edge_info_file_name = "%s_edge_info.pkl" % (modelConfig['dataset'])
            edge_index_info_name = "%s_edge_index_info.pkl"% (modelConfig['dataset'])
            graph_info_name = "%s_graph_info.pkl"% (modelConfig['dataset'])

            edge_index_info = torch.load(base_path+ edge_index_info_name)
            graph_info = torch.load(base_path+graph_info_name)
            neg_info = torch.load(base_path+"%s_neg_info.pkl" % modelConfig['dataset'])
            test_and_valid_neg_info = torch.load(base_path+"%s_valid_test_neg_info.pkl" % modelConfig['dataset'])
    else:
        entity = []
        entity_edge = []

        relation = []
        relation_edge = []

        edge2rel = {}
        edge2ent = defaultdict(set)

        n_hyper_edge = 0
        ht2edge_id = {}
        # 构建基础的超图: 实体对作为超边,首先构建训练集的超图,超边的id也从0开始

        train_id_list = []
        label_list = []
        for h,r,t in train_triples:
            edge_idx = n_hyper_edge 

            entity.append(h)
            entity_edge.append(edge_idx)

            entity.append(t)
            entity_edge.append(edge_idx)

            relation.append(r)
            relation_edge.append(edge_idx)

            edge2rel[edge_idx] = (r)

            edge2ent[edge_idx].add(h)
            edge2ent[edge_idx].add(t)

            # use for train model
            ht2edge_id[(h,r,t)] = edge_idx + n_node
            train_id_list.append(edge_idx + n_node)
            label_list.append([edge_idx+n_node,r])
            n_hyper_edge += 1
        
        max_train_hyper_edge_id = n_node + n_hyper_edge
        print("Begin build Train edge connection")
        # 计算 训练集当中共享超边的数据
        edge2entity_maxtrix = coo_matrix((
                np.ones(len(entity)),
                (np.array(entity), np.array(entity_edge))
            )).tocsr()
        
        entity2edge_maxtrix = edge2entity_maxtrix.T
        share_entity = entity2edge_maxtrix * edge2entity_maxtrix
        share_entity = share_entity.tocoo()

        share_entity_row = share_entity.row 
        share_entity_col = share_entity.col 
        edge_type_node = np.zeros_like(share_entity_row)
        edge_type = torch.LongTensor(np.concatenate([edge_type_node, edge_type_node]))
        edge_index_row = torch.LongTensor(np.concatenate([share_entity_row,share_entity_col]))
        edge_index_col = torch.LongTensor(np.concatenate([share_entity_col,share_entity_row]))
        edge_index = torch.stack([
            edge_index_row + n_node,
            edge_index_col + n_node
        ])
        print("Build Train edge connection finished......")
        # 超边和关系之间的连接
        edge2rel_maxtrix = SparseTensor(
            row=torch.as_tensor(relation, dtype=torch.long) ,
            col=torch.as_tensor(relation_edge, dtype=torch.long)+ n_node,
            value=torch.as_tensor(range(0, len(relation)), dtype=torch.long)
        )

        e2rel = torch.stack([edge2rel_maxtrix.storage.col(), edge2rel_maxtrix.storage.row()])
        # 以上构建了训练集当中的超图，和超边之间的连接关系，然后将测试和验证集合 节点-> 到超边的之间建立
        val_edge2rel = defaultdict(set)
        val_edge2ent = defaultdict(set)
        valid_list = []
        test_list = []
        
        for h,r,t in valid:
            edge_idx = n_hyper_edge # + n_node

            valid_list.append(edge_idx+n_node)
            label_list.append([edge_idx+n_node,r])

            entity.append(h)
            entity_edge.append(edge_idx)

            entity.append(t)
            entity_edge.append(edge_idx)

            ht2edge_id[(h,r,t)] = edge_idx + n_node

            val_edge2ent[edge_idx].add(h)
            val_edge2ent[edge_idx].add(t)
            val_edge2rel[edge_idx].add(r)
            n_hyper_edge += 1

        for h,r,t in test:
            edge_idx = n_hyper_edge # + n_node

            test_list.append(edge_idx+n_node)
            label_list.append([edge_idx+n_node,r])

            entity.append(h)
            entity_edge.append(edge_idx)

            entity.append(t)
            entity_edge.append(edge_idx)

            ht2edge_id[(h,r,t)] = edge_idx + n_node

            val_edge2ent[edge_idx].add(h)
            val_edge2ent[edge_idx].add(t)
            val_edge2rel[edge_idx].add(r)

            n_hyper_edge += 1

        # 构建了节点到超边的连接
        v2e = SparseTensor(
            row=torch.as_tensor(entity_edge, dtype=torch.long) + n_node,
            col=torch.as_tensor(entity, dtype=torch.long),
            value=torch.as_tensor(range(0, len(entity)), dtype=torch.long)
        )
        v2e_index = torch.stack([v2e.storage.col(), v2e.storage.row()])

        graph_info = {
            "edge_id_begin": n_node, # included
            "max_edge_id_train": max_train_hyper_edge_id, # not included
            "max_edge_id": n_node + n_hyper_edge, # not included
            "hyper_edge_num": n_hyper_edge
        }
        # 这里的index 都是从0 开始的

        edge_index_info = {
            "v2e_edge_index":v2e_index,
            "hypergraph_edge_index":edge_index,
            "hypergraph_edge_type":edge_type,
            "e2rel": e2rel,

            "train_id_list": train_id_list,
            "valid_id_list": valid_list,
            "test_id_list": test_list,
            "label_list" : label_list,
            "ht2edge_id": ht2edge_id,

            "val_edge2ent": val_edge2ent,
            "val_edge2rel": val_edge2rel,

            "edge2ent": edge2ent,
            "edge2rel": edge2rel,

        }
        print("try to merge valid positive data to train")
        new_edge_index, new_edge_type = merge_valid2train(edge_index, edge_type ,edge_index_info, n_node, max_train_hyper_edge_id, n_node + n_hyper_edge)
        edge_index_info["valid_edge_index"] = new_edge_index
        edge_index_info["valid_edge_type"] = new_edge_type
        print("save datasets info: %s" % modelConfig['dataset'])
        base_path = "/home/skl/yl/relation_cl/pre_handle_data/"
        edge_info_file_name = "%s_edge_info.pkl" % (modelConfig['dataset'])
        edge_index_info_name = "%s_edge_index_info.pkl"% (modelConfig['dataset'])
        graph_info_name = "%s_graph_info.pkl"% (modelConfig['dataset'])
        torch.save(edge_index_info, base_path+ edge_index_info_name)
        torch.save(graph_info,base_path+graph_info_name)

        print("begin build train data neg sample")
        neg_info = build_neg_dataset(train_triples,ht2edge_id, edge_index_info, graph_info, n_entity, n_size)
        print("begin build valid data neg sample")
        valid_neg_info = build_neg_dataset(valid,ht2edge_id, edge_index_info, graph_info, n_entity, n_size)
        print("begin build test data neg sample")
        test_neg_info = build_neg_dataset(test,ht2edge_id, edge_index_info, graph_info, n_entity, n_size)

        test_and_valid_neg_info = {
            "valid":valid_neg_info,
            "test": test_neg_info
        }
        torch.save(neg_info,base_path+"%s_neg_info.pkl" % modelConfig['dataset'])
        torch.save(test_and_valid_neg_info,base_path+"%s_valid_test_neg_info.pkl" % modelConfig['dataset'])

    # 构建基础的图结构
    train_id_list,valid_list,test_list,label_list = edge_index_info["train_id_list"],edge_index_info["valid_id_list"],edge_index_info["test_id_list"],edge_index_info["label_list"]
    new_edge_index,new_edge_type = edge_index_info["valid_edge_index"],edge_index_info["valid_edge_type"]
    e2rel = edge_index_info["e2rel"]

    print("data info init over.....")
    return graph_info,edge_index_info["v2e_edge_index"],edge_index_info["hypergraph_edge_index"],edge_index_info["hypergraph_edge_type"],edge_index_info["ht2edge_id"],edge_index_info,(train_id_list,valid_list,test_list,label_list), (new_edge_index,new_edge_type),e2rel,neg_info,test_and_valid_neg_info

def merge_neg2train(edge_index, edge_type, edgeInfo, n_ent_rel, max_base_edge, neg_begin, neg_end, val_edge2ent, n_entity,base_hyper_edge_num):
    node1_e = []
    node2_e = []

    edge2ent, edge2rel = edgeInfo["edge2ent"], edgeInfo["edge2rel"]
    # 训练图里面的

    train_edge = []
    train_node = []

    for key in edge2ent:
        for ent in edge2ent[key]:
            train_edge.append(key)
            train_node.append(ent)

    
    print("train edge node link number: %d" % len(train_edge))
    valid_edge = []
    valid_node = []
    for key in val_edge2ent:
        for ent in val_edge2ent[key]:
            valid_edge.append(key - base_hyper_edge_num)
            valid_node.append(ent)
    print("valid edge node link number: %d" % len(valid_edge))


    edge2entity_train = coo_matrix((
                np.ones(len(train_node)),
                (np.array(train_edge), np.array(train_node))
            ),shape=(len(edge2ent.keys()),n_entity)).tocsr()

    edge2entity_valid = coo_matrix((
                np.ones(len(valid_edge)),
                (np.array(valid_edge), np.array(valid_node))
            ),shape=(len(val_edge2ent.keys()),n_entity)).tocsr()
    print("begin caculate valid and train connection")
    valid_edge2train_edge = edge2entity_valid * edge2entity_train.T 

    print(valid_edge2train_edge.shape)
    share_entity = valid_edge2train_edge.tocoo()

    share_entity_row = share_entity.row 
    share_entity_col = share_entity.col 


    print(np.max(share_entity_row))
    print(np.max(share_entity_col))


    # for valid_edge in range(neg_begin, neg_end):
    #     valid_edge = valid_edge - n_ent_rel # 从 0 开始
        
    #     for train_edge in range(n_ent_rel, max_train_edge):
    #         train_edge = train_edge - n_ent_rel
    #         if not val_edge2ent[valid_edge].isdisjoint(edge2ent[train_edge]):
    #             node1_e.append(train_edge)
    #             node2_e.append(valid_edge)
    

    edge_type_node = np.zeros_like(share_entity_row)
    edge_type_node = torch.LongTensor(edge_type_node)
    edge_index_row = torch.LongTensor(share_entity_row)
    edge_index_col = torch.LongTensor(share_entity_col)

    new_edge_index = torch.stack([
        edge_index_col + n_ent_rel,
       edge_index_row + neg_begin
    ])
    new_edge_index = torch.cat((edge_index, new_edge_index),dim=-1)
    new_edge_type = torch.cat((edge_type, edge_type_node),dim=-1)
    return new_edge_index, new_edge_type


def merge_valid2train(edge_index, edge_type, edgeInfo, n_ent_rel, max_train_edge, max_edge):
    node1_e = []
    node2_e = []

    val_edge2ent,val_edge2rel = edgeInfo["val_edge2ent"], edgeInfo["val_edge2rel"]
    edge2ent, edge2rel = edgeInfo["edge2ent"], edgeInfo["edge2rel"]

    for valid_edge in range(max_train_edge, max_edge):
        valid_edge = valid_edge - n_ent_rel # 从 0 开始
        
        for train_edge in range(n_ent_rel, max_train_edge):
            train_edge = train_edge - n_ent_rel
            if not val_edge2ent[valid_edge].isdisjoint(edge2ent[train_edge]):
                node1_e.append(train_edge)
                node2_e.append(valid_edge)

    edge_type_node = np.zeros_like(node1_e)
    edge_type_node = torch.LongTensor(edge_type_node)
    edge_index_row = torch.LongTensor(node1_e)
    edge_index_col = torch.LongTensor(node2_e)

    new_edge_index = torch.stack([
       edge_index_row + n_ent_rel,
       edge_index_col + n_ent_rel
    ])
    new_edge_index = torch.cat((edge_index, new_edge_index),dim=-1)
    new_edge_type = torch.cat((edge_type, edge_type_node),dim=-1)
    return new_edge_index, new_edge_type