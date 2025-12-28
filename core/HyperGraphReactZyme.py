import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from core.HyperCEBrenda import HyperCE
from loss import *
from torch.cuda import amp

import torch.utils.data
from transformers import T5Tokenizer

from loss import *



class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, sigmoid_last_layer=False):
        super(MLPModel, self).__init__()

        # construct layers
        layers = [torch.nn.Linear(input_dim, hidden_dim),
                  torch.nn.ReLU(),
                  torch.nn.Dropout(dropout),
                  torch.nn.Linear(hidden_dim, output_dim)]
        if sigmoid_last_layer:
            layers.append(torch.nn.Sigmoid())

        # construct model
        self.predictor = torch.nn.Sequential(*layers)

    def forward(self, X):
        X = self.predictor(X)
        return X
class CrossAttention(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, output_dim):
        super(CrossAttention, self).__init__()
        
        self.out_dim = output_dim
        self.W_Q = nn.Linear(query_input_dim, output_dim)
        self.W_K = nn.Linear(key_input_dim, output_dim)
        self.W_V = nn.Linear(key_input_dim, output_dim)
        self.scale_val = self.out_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query_input, key_input, value_input, query_input_mask=None, key_input_mask=None):
        query = self.W_Q(query_input)
        key = self.W_K(key_input)
        value = self.W_V(value_input)

        attn_weights = torch.matmul(query, key.transpose(1, 2)) / self.scale_val
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, value)
        
        return output
    
class HyperGraphV3(Module):
    def __init__(self, hyperkgeConfig=None,n_node=0,n_hyper_edge=0,e_num=100,graph_info=None,config=None,NodeGnnDataset=None,clDataset=None):
        super(HyperGraphV3, self).__init__()

        self.hyperkgeConfig = hyperkgeConfig
        self.encoder = HyperCE(hyperkgeConfig,n_node,n_hyper_edge,e_num,graph_info)
        hidden_dim = hyperkgeConfig.embedding_dim

        self.entity_dim = hyperkgeConfig.embedding_dim
      
        self.c_num = graph_info["c_num"]
        self.e_num = graph_info["e_num"]


        self.rel_emb = nn.Embedding(self.e_num, 1280)
        self.node_emb = nn.Embedding(self.c_num, 1024)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.NodeGnnDataset= NodeGnnDataset
        self.clDataset = clDataset

        self.relation_lin = torch.nn.Sequential(
            torch.nn.Linear(1280, 1024),
            torch.nn.ReLU()
        )
        self.loss_funcation = nn.BCEWithLogitsLoss()
        mol_input_dim = 1024
        dropout = 0.5
        hidden_dim = 128
        self.lin_mol_embed = nn.Sequential(
                                    nn.Linear(mol_input_dim*2, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, hidden_dim, bias=False),
                                     nn.BatchNorm1d(hidden_dim),
                                    )
        seq_input_dim = 1280
        self.lin_seq_embed = nn.Sequential(
                                    nn.Linear(seq_input_dim, 512, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(512),
                                    nn.SiLU(),
                                    nn.Linear(512, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, hidden_dim, bias=False),
                                    nn.BatchNorm1d(hidden_dim),
                                    )
        self.hidden_dim = hidden_dim
        output_dim=64
        self.lin_out = nn.Sequential(
                                    nn.Linear(2*hidden_dim, hidden_dim, bias=False),
                                    # nn.BatchNorm1d(hidden_dim),
                                    nn.Dropout(dropout),
                                    nn.SiLU(),
                                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                                    # nn.BatchNorm1d(hidden_dim),
                                    nn.Dropout(dropout),
                                    nn.SiLU(),
                                    nn.Linear(hidden_dim, output_dim, bias=False),
                                    # nn.BatchNorm1d(output_dim),
                                    nn.Dropout(dropout),
                                    nn.SiLU(),
                                    nn.Linear(output_dim, 16, bias=False),
                                    # nn.BatchNorm1d(16),
                                    nn.Dropout(dropout),
                                    nn.Linear(16, 1, bias=False)
                                    )
        self.cross_attn_seq = CrossAttention(
                                query_input_dim=hidden_dim,
                                key_input_dim=hidden_dim,
                                output_dim=hidden_dim,
                            )
        
        self.cross_attn_mol = CrossAttention(
                                query_input_dim=hidden_dim,
                                key_input_dim=hidden_dim,
                                output_dim=hidden_dim,
                            )
        label = [0.0 for i in range(401)]
        label[0]=1.0
        self.label = torch.tensor(label)

    def init_embedding(self, c_embedding, e_embedding):
        """初始化Embedding层权重"""
        assert c_embedding.shape == (self.node_emb.num_embeddings, 
                                       self.node_emb.embedding_dim)
        assert e_embedding.shape == (self.rel_emb.num_embeddings, 
                                       self.rel_emb.embedding_dim)
        self.node_emb.weight.data.copy_(c_embedding)
        self.rel_emb.weight.data.copy_(e_embedding)
        # 可选：冻结Embedding层
        self.node_emb.weight.requires_grad = False
        self.rel_emb.weight.requires_grad = False


    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
   

    def reg_l2(self):
        return torch.mean(torch.norm(self.node_emb.weight,dim=-1))

    def get_base_emb(self, nids):
        node = self.node_emb(nids.cuda())
        return node

    def single_emb(self, data):
        n_id, adjs, split_idx = data
        n_id = n_id[split_idx:]
        x = self.get_base_emb(n_id)
        hyper_edge_emb = self.encoder(n_id,x, adjs,split_idx, True)
        return hyper_edge_emb

    def sample_noise(self, x):
        noise = torch.randn_like(x).to(x.device) * self.noise_sigma
        return noise

    def full_score(self, head_emb, relation, tail_emb,add_noise=False):
        
        if relation is not None:
            relation_emb = self.rel_emb(relation.cuda())
        else:
            relation_emb = None
        relation_emb = relation_emb
        return self.mlp_score(head_emb, relation_emb, tail_emb)

    def mlp_score(self, head_emb, relation, tail_emb,add_noise=False):

        b_size = head_emb.size(0)
        mol_embedded = self.lin_mol_embed(torch.cat([head_emb, tail_emb], dim=-1).reshape(b_size, -1)) 
        seq_embedded = self.lin_seq_embed(relation.reshape(-1,relation.size(-1)))
        mol_embedded = mol_embedded.reshape(b_size, -1, self.hidden_dim)
        seq_embedded = seq_embedded.reshape(b_size, -1,  self.hidden_dim)
        mol_embedded = mol_embedded.repeat([1, seq_embedded.shape[1],1])
        _mol_embedded = self.cross_attn_mol(mol_embedded, seq_embedded, seq_embedded) #(B,H)
        _seq_embedded = self.cross_attn_seq(seq_embedded, mol_embedded, mol_embedded) #(B,H)
        outputs = self.lin_out(torch.cat([_mol_embedded, _seq_embedded], dim=-1).reshape(-1, 2*self.hidden_dim)).reshape(b_size, -1)
        return outputs

    def realtion_predict(self, head, relation, tail):
        head = head.reshape(head.shape[0],-1)
        tail = tail.reshape(tail.shape[0],-1)
        if head.shape[0] != tail.shape[0]: return None
        emb = torch.cat([head, tail],dim=1)
        emb = self.dropout(emb)
        score = self.ce_predictor(emb)
        return score
    
    def paire_score(self, head, relation, tail):
        re_head, re_tail= torch.chunk(relation, 2, dim=-1)
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        score = head * re_head - tail * re_tail
        score = - torch.norm(score, p=1, dim=2)
        return score

    def realtion_predict_all(self, head, relation, tail):
        head = head.repeat(1,relation.shape[1],1)
        tail = tail.repeat(1,relation.shape[1],1)
        emb = torch.cat([head, tail,relation],dim=-1)

        q = self.dropout(torch.relu(self.q_weight(emb)))
        k = self.dropout(torch.relu(self.k_weight(emb)))
        v = self.dropout(torch.relu(self.v_weight(emb)))

        attention_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.entity_dim)
        # print(attention_weights.shape)
        attention_scores = torch.matmul(attention_weights, v)
        emb = self.dropout(attention_scores)
        # print(attention_scores.shape)
        score = self.ce_predictor(attention_scores)
        score = score.squeeze(-1)
        return score

    def realtion_predict_mul(self, head, relation, tail):
        emb = head * tail * relation
        emb = self.dropout(emb)
        score = self.ce_predictor(emb)
        score = score.squeeze(-1)
        return score

    def distmult_score(self, head, relation, tail):
        score = head * tail * relation
        return torch.sigmoid(torch.sum(score,dim=-1))

    def complex_score(self, head, relation, tail):
        head_re, head_im = head.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)
        relation_re, relation_im = relation.chunk(2, -1)   # (batch,1,dim)
        tail_re, tail_im = tail.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)

        score_re = head_re * relation_re - head_im * relation_im
        score_im = head_re * relation_im + head_im * relation_re 
        result = score_re * tail_re + score_im * tail_im
        score = torch.sum(result,dim=-1)
        return score

    def tucker_score(self, head, relation, tail):
        batch_size = head.shape[0]
        
        head = head.reshape(-1, self.entity_dim)
        x = self.bn0(head)
        x = self.input_dropout(x)
        x = x.reshape(batch_size, -1, self.entity_dim)

        # 核心张量与关系做x2 乘积
        tail = tail.reshape(-1, self.entity_dim)
        W_mat = torch.mm(tail, self.W.reshape(self.entity_dim, -1))
        W_mat = W_mat.reshape(-1, self.entity_dim, self.entity_dim)
        W_mat = self.hidden_dropout1(W_mat)                     # shape = (batch_size, e_dim, e_dim)
        x = torch.bmm(x, W_mat)                                 # shape = (batch_size, n, e_dim)

        x = x.reshape(-1, self.entity_dim)      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)                             # shape = (batch_size*n, e_dim)

        # 然后根据tail的形状进行计算: (batch_size, n, e_dim) or （1 , n_entity, e_dim)
        if relation.shape[0] == batch_size:
            x = x.reshape(batch_size, -1, self.entity_dim) # shape = (batch_size, n, e_dim)
            x = torch.bmm(x,relation.permute(0,2,1)) # result(batch_size, n, 1)
        else:
            tail = tail.reshape(-1,self.entity_dim)
            x = torch.mm(x, relation.permute(1,0))
        if len(x.shape) > 2:
            x = torch.squeeze(x)
        x = torch.sigmoid(x)
        return x
    
    @staticmethod
    def cl_score(x,y,weight,temperature=0.5):
        if x.shape != y.shape:
            return None
        """
        计算对比损失
        :param x: Tensor, shape (batch_size, dim)
        :param y: Tensor, shape (batch_size, dim)
        :param temperature: 温度参数，用于缩放相似度
        :return: 对比损失值
        """
        batch_size = x.shape[0]
        
        # 计算余弦相似度
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        similarity_matrix = torch.matmul(x, y.T) / temperature # N*N的相似矩阵
        # 生成标签
        labels = torch.arange(batch_size).to(x.device)

        softmax_scores = F.log_softmax(similarity_matrix, dim=1)
       
        loss = -softmax_scores[torch.arange(batch_size), labels] 
        if weight != None:
            weight = weight.to(x.device)
            loss = loss * weight
        loss = loss.mean()
        return loss

    @staticmethod
    def train_step(model,optimizer,data, config=None, scaler=None):
        optimizer.zero_grad(set_to_none=True)
        model.train()

        triples,labels, head_out,tail_out = data
        relation = triples[:,1]

        head_out, cl_head = head_out
        tail_out, cl_tail = tail_out
        with torch.amp.autocast('cuda'):
            # base loss
            head_emb = model.single_emb(head_out)
            tail_emb = model.single_emb(tail_out)
            score = model.full_score(head_emb, relation, tail_emb)
            labels = labels.unsqueeze(1).cuda()
            loss = model.loss_funcation(score,labels)
            logs = {    
                "loss": loss.item(),
            }
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return logs
