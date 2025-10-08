import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from core.HyperCEBrenda import HyperCE
from loss import *
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
    
class HyperGraphV3(Module):
    def __init__(self, hyperkgeConfig=None,n_node=0,n_hyper_edge=0,e_num=100,graph_info=None,config=None,NodeGnnDataset=None,clDataset=None):
        super(HyperGraphV3, self).__init__()

        self.hyperkgeConfig = hyperkgeConfig
        self.encoder = HyperCE(hyperkgeConfig,n_node,n_hyper_edge,e_num,graph_info)
        hidden_dim = hyperkgeConfig.embedding_dim

        self.entity_dim = hyperkgeConfig.embedding_dim
      
        self.c_num = graph_info["c_num"]
        self.e_num = graph_info["e_num"]


        self.rel_emb = nn.Embedding(self.e_num, hidden_dim*2)
        self.node_emb = nn.Embedding(self.c_num, hidden_dim)

        init_range =  6.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.rel_emb.weight, -init_range, init_range)

        self.dropout = torch.nn.Dropout(p=0.5)

        # self.node_encoder = GNN(
        #     num_layer=config["node_gnn_layer"],
        #     emb_dim=hidden_dim,
        #     gnn_type='gin',
        #     drop_ratio=config["node_gnn_dropout"],
        #     JK='last',
        # )
        self.mse_loss = nn.MSELoss()
        # self.node_encoder = TrfmSeq2seq(len(NodeGnnDataset.vocab),config["transformer_dim"], len(NodeGnnDataset.vocab), config["transformer_layer"]).cuda()
        # self.tokenizer = T5Tokenizer.from_pretrained("/home/skl/yl/ce_project/relation_cl/core/mollm/pretrain_model/MoleculeCaption/molt5-base-smiles2caption/", model_max_length=512)
        # self.node_encoder = GinDecoder(has_graph=False, MoMuK=False, model_size="base", use_3d=True)
        # for name, parameter in self.node_encoder.named_parameters():
        #     parameter.requires_grad = False
        
        self.NodeGnnDataset= NodeGnnDataset
        self.clDataset = clDataset
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim*2, self.e_num),
            torch.nn.Sigmoid()
        )
        self.loss_funcation = nn.CrossEntropyLoss()
        self.W  = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.entity_dim, self.entity_dim,self.entity_dim)),dtype=torch.float))

        self.input_dropout = torch.nn.Dropout(0.8)
        self.hidden_dropout1 = torch.nn.Dropout(0.8)
        self.hidden_dropout2 = torch.nn.Dropout(0.8)

        self.bn0 = torch.nn.BatchNorm1d(self.entity_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.entity_dim)

        # self.box = BoxLevel(5406,1, self.entity_dim)

        self.q_weight = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim * 3, hyperkgeConfig.embedding_dim),
            torch.nn.ReLU()
        )
        self.k_weight = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim * 3, hyperkgeConfig.embedding_dim),
            torch.nn.ReLU()
        )
        self.v_weight = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim * 3, hyperkgeConfig.embedding_dim),
            torch.nn.ReLU()
        )

        label = [0.0 for i in range(401)]
        label[0]=1.0
        self.label = torch.tensor(label)

    def init_node_embedding(self):
        # embeddings = []
        # transfor_embedding = []
        # for i in range(self.c_num):
        #     embeddings.append(self.NodeGnnDataset.get(i))
          
        #     if len(embeddings) == 32 or i == self.c_num-1:
        #         smiles_tokens_ = self.tokenizer(embeddings, padding=True, truncation=True, return_tensors="pt")
        #         smiles_tokens = smiles_tokens_['input_ids'].cuda()
        #         src_padding_mask = smiles_tokens_['attention_mask'].cuda()  # encoder input mask
        #         node = self.node_encoder.get_smiles_emb(smiles_tokens,src_padding_mask)
        #         node = torch.mean(node, dim=1)
        #         transfor_embedding.append(node)
        #         embeddings = []
        # embeddings = torch.cat(transfor_embedding, dim=0)
        # self.node_emb = embeddings
        # self.node_emb =  torch.FloatTensor(torch.randn(self.c_num,self.entity_dim))
        # print("init node_emb over: %s" % str(self.node_emb.shape))
        pass


    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
   

    def reg_l2(self):
        return torch.mean(torch.norm(self.node_emb.weight,dim=-1))
        # reg_loss = 0
        # for param in self.parameters():
        #     reg_loss += torch.norm(param, p=2)  # 使用L2正则化
        # return reg_loss
        # return torch.mean(torch.norm())

    def get_base_emb(self, nids):
        # batch = []
        # for i in range(len(nids)):
        #     batch.append(self.NodeGnnDataset.get(nids[i]))
        # smiles_tokens_ = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        # smiles_tokens = smiles_tokens_['input_ids'].cuda()
        # src_padding_mask = smiles_tokens_['attention_mask'].cuda()  # encoder input mask
        # node = self.node_encoder.get_smiles_emb(smiles_tokens,src_padding_mask)
        # node = torch.mean(node, dim=1)
        nids = nids.cuda()
        node = self.node_emb(nids)
        # node = self.node_emb[nids]

        # print(node.shape)
        # batch = torch.stack(batch,dim=0)
        # batch = batch.cuda()
        # batch_node = self.node_encoder.encode(batch)
        return node

    def single_emb(self, data,add_noise=False):
        n_id, adjs, split_idx = data
        n_id = n_id[split_idx:]
        x = self.get_base_emb(n_id)
        if add_noise:
            x_noise = self.sample_noise(x)
            x = x + x_noise
        # n_id = n_id.cuda()
        # x = self.node_emb(n_id[split_idx:])
        hyper_edge_emb = self.encoder(n_id,x, adjs,split_idx, True)
        return hyper_edge_emb

    def sample_noise(self, x):
        noise = torch.randn_like(x).to(x.device) * self.noise_sigma
        return noise

    def full_score(self, head_emb, relation, tail_emb,add_noise=False):
        
        if relation is not None:
            relation_emb = self.rel_emb(relation)
            if len(relation_emb.shape) == 2:
                relation_emb = relation_emb.unsqueeze(1)
        else:
            relation_emb = None

        if len(head_emb.shape) == 2:
            head_emb = head_emb.unsqueeze(1)

        if len(tail_emb.shape) == 2:
            tail_emb = tail_emb.unsqueeze(1)
        if add_noise:
            head_noise = self.sample_noise(head_emb)
            tail_noise = self.sample_noise(tail_emb)
            head_emb = head_emb + head_noise
            tail_emb = tail_emb + tail_noise
        return self.realtion_predict(head_emb, relation_emb, tail_emb)

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
    def train_step(model,optimizer,data,loss_funcation, config=None,subClassOf=None, typeOf= None, subLoss=None, typeLoss=None,cl_dataset=None):
        optimizer.zero_grad()
        model.train()

        head,relation,tail,negative_sample,head_out,tail_out = data
        relation = relation.cuda()

        head_out, cl_head = head_out
        tail_out, cl_tail = tail_out
        
        # base loss
        head_emb = model.single_emb(head_out)
        tail_emb = model.single_emb(tail_out)
        pos_score = model.full_score(head_emb, relation, tail_emb)

        # pos_score = pos_score.unsqueeze(-1)
        # lable shapre = len(head)* (1 + len(negative_sample))
        # lable = torch.zeros(len(head), 1 + len(negative_sample[0]))
        # lable[:, 0] = 1 
        # lable = lable.cuda()

        # negative_sample = negative_sample.cuda()
        # neg_score = model.full_score(head_emb, negative_sample, tail_emb)

        # score = torch.cat([pos_score,neg_score],dim=-1)
        # loss = model.loss_funcation(score,lable)

        loss = model.loss_funcation(pos_score,relation)
        # loss = model.loss_funcation(pos_score,neg_score)
        logs = {    
            "loss": loss.item(),
        }
        if config["add_entity_noise"]:
            head_emb_noise = model.single_emb(head_out,add_noise=True)
            tail_emb_noise = model.single_emb(tail_out,add_noise=True)
            noise_score = model.full_score(head_emb_noise, relation, tail_emb_noise, add_noise=False)
            mse_loss = model.mse_loss(pos_score, noise_score)
            noise_loss = model.loss_funcation(noise_score,relation)

            loss = loss + mse_loss*config["noise_weight"] + noise_loss
            logs["mse_loss"] = mse_loss.item()*config["noise_weight"]
           
        if config["add_edge_noise"]:
            noise_score = model.full_score(head_emb, relation, tail_emb, add_noise=True)
            mse_loss = model.mse_loss(pos_score, noise_score)
            noise_loss = model.loss_funcation(noise_score,relation)

            loss = loss + mse_loss*config["noise_weight"] + noise_loss
            logs["mse_loss"] = mse_loss.item()*config["noise_weight"] 
            logs["noise_loss"] = noise_loss.item()

        add_cl = False
        if config["add_edge_cl"]:
            base,pos,weight,base_out,pos_out = next(cl_dataset)
            base_emb = model.single_emb(base_out)
            pos_emb = model.single_emb(pos_out)
            cl_loss = HyperGraphV3.cl_score(base_emb,pos_emb,weight, config["cl_temp"])
            loss += config["cl_weight"]*cl_loss
            logs["cl_loss"] = cl_loss.item() * config["cl_weight"]

        if config["reg_weight"] != 0.0:
            reg = model.reg_l2()
            logs["reg"] = reg * config["reg_weight"]
            loss += reg * config["reg_weight"]
        loss.backward()
        optimizer.step()
        return logs
