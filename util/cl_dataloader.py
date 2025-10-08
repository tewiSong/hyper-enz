from concurrent.futures import thread
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from collections import defaultdict

# 按照伯努利的方法进行负采样
class NewOne2OneDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, init_value=-1,n_size=100, random=True, triple2Edge=None):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.init_value = init_value
        self.triple_set = set(triples)
        self.n_size = n_size
        self.positiveEdge = False
        if self.positiveEdge:  
            self.toal_size = self.n_size
        else:
            self.toal_size = self.n_size+1

        self.random = random
        self.pofHead = NewOne2OneDataset.count_relation_frequency(self.triple_set,nentity,nrelation)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)
        self.head_num = self.tail_num = nentity
        self.true_value = self.get_true_tails()
        if triple2Edge != None:
            self.triple2edge = triple2Edge
        else:
            self.triple2edge = defaultdict(int)
        
   
    def __len__(self):
        return len(self.triples)
        
    def get_true_tails(self):
        true_value =  torch.zeros(self.n_size+1) 
        nn.init.constant_(true_value, self.init_value)
        true_value[0] = 1
        return true_value 

    def __getitem__(self, idx):
        head,relation,tail =self.triples[idx]
        edge_id = self.triple2edge[(head,relation,tail)]

        sample_r = torch.LongTensor([relation]).expand(self.toal_size)
        pr = self.pofHead[relation]
        rand_value = np.random.randint(np.iinfo(np.int32).max) % 1000
        n_list=[]
        if rand_value > pr:
            sample_h = torch.LongTensor([head]).expand(self.toal_size)
            if not self.positiveEdge:
                n_list = [[tail]]
        else:
            sample_t = torch.LongTensor([tail]).expand(self.toal_size)
            if not self.positiveEdge:
                n_list = [[head]]
        n_size = 0
        replace_index = 0
        while n_size < self.n_size:
            rdm_words = np.random.randint(0, self.nentity, self.n_size*2)
            if rand_value > pr:
                mask = np.in1d(
                    rdm_words, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else: # 如果小于则替换头实体
                mask = np.in1d(
                    rdm_words, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            rdm_words = rdm_words[mask] # filter true triples
            n_list.append(rdm_words)
            n_size += rdm_words.size
        
        negative_sample = np.concatenate(n_list)[:self.toal_size]

        if rand_value > pr:
            sample_t =  torch.LongTensor(negative_sample)
        else:
            sample_h = torch.LongTensor(negative_sample)

        if self.random:
            shuffle_idx = torch.randperm(sample_h.nelement())
            return sample_h[shuffle_idx],sample_r[shuffle_idx], sample_t[shuffle_idx], self.true_value[shuffle_idx],torch.LongTensor([edge_id]),torch.LongTensor([head,relation,tail])
        else:
            return sample_h ,sample_r, sample_t, self.true_value, torch.LongTensor([edge_id]),torch.LongTensor([head,relation,tail])
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        value = torch.stack([_[3] for _ in data], dim=0)
        edge_id = torch.stack([_[4] for _ in data], dim=0)
        true_id = torch.stack([_[5] for _ in data], dim=0)
        return h,r,t, value,edge_id,true_id
    @staticmethod
    def count_relation_frequency(triples, rel_id_begin,nrelation):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        tr2h = {}
        hr2t = {}

        for head, relation, tail in triples:

            if relation not in tr2h.keys():
                tr2h[relation] = {}
            if relation not in hr2t.keys():
                hr2t[relation] = {}
            
            if head not in hr2t[relation].keys():
                hr2t[relation][head] = 1
            else:
                hr2t[relation][head] += 1

            if tail not in tr2h[relation].keys():
                tr2h[relation][tail] = 1
            else:
                tr2h[relation][tail] += 1
        
        hrpt = {}  # 记录关系中每个头实体的平均尾实体数量
        trph = {}  # 记录关系中每个尾实体的平均头实体数量
        pofHead = {}
        for rid in range(rel_id_begin,nrelation+rel_id_begin):
            if rid not in hr2t.keys(): continue
            if rid not in tr2h.keys(): continue

            t_total =np.sum([hr2t[rid][key] for key in hr2t[rid].keys()])
            h_total = len(list(hr2t[rid].keys()))
            hrpt[rid] = float(t_total)/h_total  # 平均每个头实体 对应多少个尾实体

            right_count =np.sum([tr2h[rid][key] for key in tr2h[rid].keys() ])
            left_count = len(list(tr2h[rid].keys()))
            trph[rid] = float(right_count)/left_count

            # 如果这个值越大，就说明越应该替换头实体
            pofHead[rid] = 1000* hrpt[rid]/(hrpt[rid]+trph[rid])
           
        return pofHead

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

class RelationTestDataset(Dataset):
    def __init__(self, triples, nentity, nrelation,all_triples, init_value=-1, n_size=100, random=False,triple2Edge=None):

        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.init_value = init_value
        self.triple_set = set(triples)
        self.n_size = n_size
        self.positiveEdge = False
        self.toal_size = self.n_size
        self.random = random
        self.true_value = self.get_true_tails()
        self.all_triples = set(all_triples)

        self.true_rel = RelationTestDataset.get_true_rel(self.triples)
        self.ht2edge_id = triple2Edge
   
    def __len__(self):
        return len(self.triples)
        
    def get_true_tails(self):
        true_value =  torch.zeros(self.n_size+1) 
        nn.init.constant_(true_value, self.init_value)
        true_value[0] = 1
        return true_value 

    def __getitem__(self, idx):

        head,relation,tail =self.triples[idx]
        tmp = [(0, rand_r) if (head, rand_r, tail) not in self.all_triples
                   else (-1, rand_r) for rand_r in range(self.nentity+1, self.nentity+self.nrelation)]
        tmp[relation-self.nentity] = (0, relation)

        tmp  = torch.LongTensor(tmp)
        sample_r =   tmp[...,1]
        filter_bias = tmp[...,0].float()
        true_r = torch.LongTensor([relation])
        return torch.LongTensor([self.ht2edge_id[(head,tail)]]) ,sample_r,true_r, filter_bias 
    
    @staticmethod
    def collate_fn(data):
        edge_id = torch.stack([_[0] for _ in data], dim=0)
        sample_r = torch.stack([_[1] for _ in data], dim=0)
        true_r = torch.stack([_[2] for _ in data], dim=0)
        bias = torch.stack([_[3] for _ in data], dim=0)

        return edge_id, sample_r, true_r, bias
    
    @staticmethod
    def get_true_rel(triples):
        ht2r = defaultdict(list)
        for h,r,t in triples:
            ht2r[(h,t)].append(r)
        return ht2r


class RelationPredictDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, init_value=-1, n_size=100, random=True,triple2Edge=None):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.init_value = init_value
        self.triple_set = set(triples)
        self.n_size = n_size
        self.positiveEdge = False
        self.toal_size = self.n_size+1
        self.random = random
        self.true_value = self.get_true_tails()

        self.true_rel = RelationPredictDataset.get_true_rel(self.triples)
        self.ht2edge_id = triple2Edge
   
    def __len__(self):
        return len(self.triples)
        
    def get_true_tails(self):
        true_value =  torch.zeros(self.n_size+1) 
        nn.init.constant_(true_value, self.init_value)
        true_value[0] = 1
        return true_value 

    def __getitem__(self, idx):

        head,relation,tail =self.triples[idx]
        n_list = [[relation]]

        n_size = 0
        replace_index = 0
        while n_size < self.n_size:
            rdm_words = np.random.randint(self.nentity+1, self.nentity+self.nrelation, self.n_size*2)
            mask = np.in1d(
                    rdm_words, 
                    self.true_rel[(head, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            rdm_words = rdm_words[mask] # filter true triples
            n_list.append(rdm_words)
            n_size += rdm_words.size
        
        negative_sample = np.concatenate(n_list)[:self.toal_size]
        negative_sample = torch.LongTensor(negative_sample)
        if self.random:
            shuffle_idx = torch.randperm(self.toal_size)
            return negative_sample[shuffle_idx], self.true_value[shuffle_idx], torch.LongTensor([self.ht2edge_id[(head,tail)]]),torch.LongTensor([head,relation,tail])
        else:
            return negative_sample[shuffle_idx], self.true_value, torch.LongTensor(self.ht2edge_id[(head,tail)]),torch.LongTensor([head,relation,tail])
    
    @staticmethod
    def collate_fn(data):
        r = torch.stack([_[0] for _ in data], dim=0)
        value = torch.stack([_[1] for _ in data], dim=0)
        edge_id = torch.stack([_[2] for _ in data], dim=0)
        true_id = torch.stack([_[3] for _ in data], dim=0)

        return None,r,None, value,edge_id,true_id
    
    @staticmethod
    def get_true_rel(triples):
        ht2r = defaultdict(list)
        for h,r,t in triples:
            ht2r[(h,t)].append(r)
        return ht2r

  

class CLDataset(Dataset):
    
    def __init__(self, triples, nentity, nrelation, p_size, n_size, po_dict,funct_idset,asy_idset,po_type='conf',
    relation_p=None,asyDict=None, funcDict=None,triple2edge=None):

        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
      
        self.triple_set = set(triples)
        self.n_size = n_size
        self.p_size = p_size 

        self.po_dict = po_dict
        self.funct_idset = funct_idset
        self.asy_idset = asy_idset

        self.po_type = po_type
        self.triple2edge= triple2edge
        if self.po_type == 'Vio':
            self.asyDict = asyDict
            self.funcDict = funcDict
            self.n_size = n_size 
            self.p_size = p_size  
            self.get_sample = self.build_vioKg
        else:
            self.get_sample = self.build_confKg
        if relation_p != None:
            self.pofHead = relation_p
        else:
            self.pofHead = CLDataset.count_relation_frequency(self.triple_set,nentity,nrelation)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)
        self.head_num = self.tail_num = nentity
    
        print("Create Dataset")
        print("Type: %s", po_type)
        if po_dict != None:
            print("size of Po_dict: %d" % len(po_dict.keys()))



    def __len__(self):
        return len(self.triples)
        
    def getPos_fromTriple(self,h,r,t):
        # 如果是conv ，直接生成的正样本
        p_list = self.po_dict[(h,r,t)]
        rand = np.random.randint(0,len(p_list),self.p_size)
        return torch.LongTensor(p_list[rand])
  
    
    def getPos_fromRelation(self,r,po_dict, self_p_size):
        p_list = po_dict[r]
        rand = np.random.randint(0,len(p_list),self_p_size)
        return  torch.LongTensor(p_list[rand])

    # 根据当前三元组，随机进行负采样
    def get_random_n(self,head,relation,tail, self_n_size):

        sample_r = torch.LongTensor([relation]).expand(self_n_size)
        pr = self.pofHead[relation]
        rand_value = np.random.randint(np.iinfo(np.int32).max) % 1000
        n_list = []
        if rand_value > pr:
            sample_h = torch.LongTensor([head]).expand(self_n_size)
        else:
            sample_t = torch.LongTensor([tail]).expand(self_n_size)
        n_size = 0
        replace_index = 0
        while n_size < self_n_size:
            rdm_words = np.random.randint(0, self.nentity, self_n_size*2)
            if rand_value > pr:
                mask = np.in1d(
                    rdm_words, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else: # 如果小于则替换头实体
                mask = np.in1d(
                    rdm_words, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            rdm_words = rdm_words[mask] # filter true triples
            n_list.append(rdm_words)
            n_size += rdm_words.size
        
        negative_sample = np.concatenate(n_list)[:self_n_size]
        if rand_value > pr:
            sample_t =  torch.LongTensor(negative_sample)
        else:
            sample_h = torch.LongTensor(negative_sample)

        sample = torch.stack([sample_h ,sample_r, sample_t], dim=1)
        return sample


    def build_confKg(self,head, relation, tail):
        pos = self.getPos_fromTriple(head, relation, tail)
        neg = self.get_random_n(head, relation, tail,self.n_size)
        return pos, neg

    def build_vioKg(self,  head, relation,tail):
        pos_func = torch.LongTensor([])
        neg_func = torch.LongTensor([])
        pos_asy =  torch.LongTensor([])
        neg_asy = torch.LongTensor([])

        if relation in self.funct_idset and relation in self.asy_idset:
            self_p_size = self.p_size //2 
            self_n_size = self.n_size //2 
        else:
            self_p_size = self.p_size  
            self_n_size = self.n_size 

        if relation in self.funct_idset:
            pos_func = self.getPos_fromRelation(relation, self.funcDict,self_p_size)
            neg_func = self.get_random_n(head, relation, tail,self_n_size)
            
        if relation in self.asy_idset:
            pos_asy = self.getPos_fromRelation(relation, self.asyDict,self_p_size)
            neg_asy = torch.zeros(pos_asy.shape,dtype=torch.long)
            neg_asy[0:,0] = pos_asy[0:,2]
            neg_asy[0:,1] = pos_asy[0:,1]
            neg_asy[0:,2] = pos_asy[0:,0]

        return torch.cat([pos_func,pos_asy]), torch.cat([neg_func,neg_asy])

    def __getitem__(self, idx):

        triple = self.triples[idx]
        edge_id = self.triple2edge[triple]
        pos,neg = self.get_sample(triple[0],triple[1],triple[2])
        return torch.LongTensor(triple), pos, neg, torch.LongTensor([edge_id])
    
    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        pos = torch.stack([_[1] for _ in data], dim=0)
        neg = torch.stack([_[2] for _ in data], dim=0)
        edge_id = torch.stack([_[3] for _ in data], dim=0)
        return triple, pos, neg,edge_id


    @staticmethod
    def count_relation_frequency(triples,rel_id_begin, nrelation):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        tr2h = {}

        hr2t = {}
        for head, relation, tail in triples:
            if relation not in tr2h.keys():
                tr2h[relation] = {}
            if relation not in hr2t.keys():
                hr2t[relation] = {}
            
            if head not in hr2t[relation].keys():
                hr2t[relation][head] = 1
            else:
                hr2t[relation][head] += 1

            if tail not in tr2h[relation].keys():
                tr2h[relation][tail] = 1
            else:
                tr2h[relation][tail] += 1
        
        hrpt = {}  # 记录关系中每个头实体的平均尾实体数量
        trph = {}  # 记录关系中每个尾实体的平均头实体数量
        pofHead = {}
        for rid in range(rel_id_begin,nrelation+rel_id_begin):  
            t_total =np.sum([hr2t[rid][key] for key in hr2t[rid].keys()])
            h_total = len(list(hr2t[rid].keys()))
            hrpt[rid] = float(t_total)/h_total

            right_count =np.sum([tr2h[rid][key] for key in tr2h[rid].keys() ])
            left_count = len(list(tr2h[rid].keys()))
            trph[rid] = float(right_count)/left_count
            pofHead[rid] = 1000* hrpt[rid]/(hrpt[rid]+trph[rid])
        return pofHead

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail
class OgOne2OneDataset(Dataset):
    def __init__(self, dp, nentity, nrelation, n_size=100,Datatype='Conf',pos_rat=1,relation_p=None,triple2edge=None):
        self.nentity = nentity
        self.nrelation = nrelation
        self.Datatype = Datatype
        # 假设数据集都能够看到全部的
        if Datatype == 'Conf':
            self.pos_conf_triples,self.pos_sys_triples,self.pos_trans_triples = dp.getConSetData()
            trans_num =  int(pos_rat*len(self.pos_trans_triples))
            self.triples = self.pos_trans_triples[:trans_num] + self.pos_sys_triples
            self.triple_set = set(dp.trained)
            self.true_value = self.get_true_tails(-1,n_size+1)
        elif Datatype == 'Func':
            self.vio_triples, self.asy_triples,self.functional_triples = dp.getVioData()
            self.triples = self.functional_triples 
            self.triple_set = set(dp.trained)
            self.fun_rel_id = set(dp.fun_rel_idset)
            self.true_value = self.get_true_tails(-1.2,n_size+1)
        elif Datatype == 'Asy':
            self.vio_triples, self.asy_triples,self.functional_triples = dp.getVioData()
            self.triple_set = set(dp.trained)
            self.triples =self.asy_triples
            self.true_value = self.get_true_tails(-1.2,2)
        else:
            raise ValueError('Do not support thie data type Value: %s' % Datatype)
     
        self.n_size = n_size
        if relation_p != None:
            self.pofHead = relation_p
        else:
            self.pofHead = OgOne2OneDataset.count_relation_frequency(self.triple_set,0,nrelation)
        
        self.triple2edge=triple2edge
            
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)
        self.head_num = self.tail_num = nentity
    def __len__(self):
        return len(self.triples)
        
    def get_true_tails(self,init_value, size):
        true_value =  torch.zeros(size) 
        nn.init.constant_(true_value, init_value)
        true_value[0] = 1
        return true_value 

    def asysItem(self, idx):
        head,relation,tail =self.triples[idx]
        samples_h =torch.LongTensor( [head, tail])
        samples_t = torch.LongTensor([tail, head])
        samples_r = torch.LongTensor([relation,relation])
        shuffle_idx = torch.randperm(samples_h.nelement())
        return samples_h[shuffle_idx],samples_r[shuffle_idx], samples_t[shuffle_idx], self.true_value[shuffle_idx]
        

    def confAndFuncationalItem(self, idx):
        head,relation,tail =self.triples[idx]
        sample_r = torch.LongTensor([relation]).expand(self.n_size+1)
        pr = self.pofHead[relation]
        rand_value = np.random.randint(np.iinfo(np.int32).max) % 1000
        
        if rand_value > pr:
            sample_h = torch.LongTensor([head]).expand(self.n_size+1)
            n_list = [[tail]]
        else:
            sample_t = torch.LongTensor([tail]).expand(self.n_size+1)
            n_list = [[head]]
        n_size = 0
        replace_index = 0
        while n_size < self.n_size:
            rdm_words = np.random.randint(0, self.nentity, self.n_size*2)
            if rand_value > pr:
                mask = np.in1d(
                    rdm_words, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else: # 如果小于则替换头实体
                mask = np.in1d(
                    rdm_words, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            rdm_words = rdm_words[mask] # filter true triples
            n_list.append(rdm_words)
            n_size += rdm_words.size
        
        negative_sample = np.concatenate(n_list)[:self.n_size+1]

        if rand_value > pr:
            sample_t =  torch.LongTensor(negative_sample)
        else:
            sample_h = torch.LongTensor(negative_sample)

        shuffle_idx = torch.randperm(sample_h.nelement())

        return sample_h[shuffle_idx],sample_r[shuffle_idx], sample_t[shuffle_idx], self.true_value[shuffle_idx]
    
    def __getitem__(self, idx):
        if self.Datatype != 'Asy':
            return self.confAndFuncationalItem(idx)
        return self.asysItem(idx)
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        value = torch.stack([_[3] for _ in data], dim=0)
        return h,r,t, value
    @staticmethod
    def count_relation_frequency(triples,rel_id_begin, nrelation):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        tr2h = {}

        hr2t = {}
        for head, relation, tail in triples:
            if relation not in tr2h.keys():
                tr2h[relation] = {}
            if relation not in hr2t.keys():
                hr2t[relation] = {}
                
            
            if head not in hr2t[relation].keys():
                hr2t[relation][head] = 1
            else:
                hr2t[relation][head] += 1

            if tail not in tr2h[relation].keys():
                tr2h[relation][tail] = 1
            else:
                tr2h[relation][tail] += 1
        
        hrpt = {}  # 记录关系中每个头实体的平均尾实体数量
        trph = {}  # 记录关系中每个尾实体的平均头实体数量
        pofHead = {}
        for rid in range(rel_id_begin,nrelation+rel_id_begin):  
            t_total =np.sum([hr2t[rid][key] for key in hr2t[rid].keys()])
            h_total = len(list(hr2t[rid].keys()))
            hrpt[rid] = float(t_total)/h_total

            right_count =np.sum([tr2h[rid][key] for key in tr2h[rid].keys() ])
            left_count = len(list(tr2h[rid].keys()))
            trph[rid] = float(right_count)/left_count
            pofHead[rid] = 1000* hrpt[rid]/(hrpt[rid]+trph[rid])
           
        return pofHead

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail
class OGOneToOneTestDataset(Dataset):
    def __init__(self, triples, all_true_triples,nentity, nrelation, mode='hr_t'):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.triple_set = set(all_true_triples)
        self.mode = mode

        if mode == 'hr_t':
            self.replace_index = 2
        elif mode == 'h_rt':
            self.replace_index = 0

    def __len__(self):
        return len(self.triples)
        

    def __getitem__(self, idx):
        head,relation,tail =self.triples[idx]
        samples = torch.zeros(self.nentity, 3,dtype=torch.long)
        samples = samples + torch.LongTensor((self.triples[idx])) 
        if self.mode == 'h_rt':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
          
            tmp[head] = (0, head)
        elif self.mode == 'hr_t':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)

        tmp  = torch.LongTensor(tmp)
        samples[...,self.replace_index] = tmp[...,1]
        filter_bias = tmp[...,0].float()
        postive_sampe = torch.LongTensor(self.triples[idx])
        return postive_sampe,samples,filter_bias,self.mode

    @staticmethod
    def collate_fn(data):
        positive = torch.stack([_[0] for _ in data], dim=0)
        samples = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive, samples, filter_bias, mode