#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cmd import IDENTCHARS
from concurrent.futures import thread

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os

class TestReactionRelationDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
    

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, tail, relation = self.triples[idx]

        tmp = [(0, rand_r) if (head, tail, rand_r) not in self.triple_set
                   else (-1, relation) for rand_r in range(self.nrelation)]
        
        tmp[relation] = (0, relation)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        head = torch.LongTensor(head)
        tail = torch.LongTensor(tail)
        relation = torch.LongTensor([relation])

        return head, tail, relation, negative_sample, filter_bias
    
    @staticmethod
    def collate_fn(data):
        head_list = [_[0] for _ in data]
        head = torch.cat(head_list)
        tail_list = [_[1] for _ in data]
        tail = torch.cat(tail_list)

        head_index = []
        for i, tensor in enumerate(head_list):
            head_index.append(torch.full((len(tensor),1), i, dtype=torch.long))
        head_index = torch.cat(head_index,dim=0)

        tail_index = []
        for i, tensor in enumerate(tail_list):
            tail_index.append(torch.full((len(tensor),1), i, dtype=torch.long))
        tail_index = torch.cat(tail_index,dim=0)

        relation = torch.stack([_[2] for _ in data], dim=0)
        negative_sample = torch.stack([_[3] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        return head, tail, relation, negative_sample, filter_bias, head_index, tail_index

class NagativeReactionRelationSampleDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size,):

        self.len = len(triples)
        self.triples = triples
        self.triple_set = (triples)

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.true_realtion = self.get_true_head_and_tail(self.triple_set)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, tail, relation = positive_sample
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(0, self.nrelation,size=self.negative_sample_size*2)
            mask = np.in1d(
                    negative_sample, 
                    self.true_realtion[(head, tail)], 
                    assume_unique=True, 
                    invert=True
            )
            
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        head = torch.LongTensor(head)
        tail = torch.LongTensor(tail)
        relation = torch.LongTensor([relation])

        return head, tail, relation, negative_sample
    
    @staticmethod
    def collate_fn(data):
        # 
        head_list = [_[0] for _ in data]
        head = torch.cat(head_list)
        tail_list = [_[1] for _ in data]
        tail = torch.cat(tail_list)

        head_index = []
        for i, tensor in enumerate(head_list):
            head_index.append(torch.full((len(tensor),1), i, dtype=torch.long))
        head_index = torch.cat(head_index,dim=0)

        tail_index = []
        for i, tensor in enumerate(tail_list):
            tail_index.append(torch.full((len(tensor),1), i, dtype=torch.long))
        tail_index = torch.cat(tail_index,dim=0)

        relation = torch.stack([_[2] for _ in data], dim=0)
        negative_sample = torch.stack([_[3] for _ in data], dim=0)

        return head,tail, relation, negative_sample, head_index, tail_index
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_relation = {}
        for left, right, e in triples:
            if (left, right) not in true_relation:
                true_relation[(left, right)] = []
            true_relation[(left, right)].append(e)
        
        for left, right in true_relation:
            true_relation[(left, right)] = np.array(list(set(true_relation[(left, right)])))
        
        return true_relation

class NagativeReactionSampleDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size,mode):

        self.len = len(triples)
        self.triples = triples
        self.triple_set = np.array(triples)

        equa_set = set()
        for left,right,e in triples:
            equa_set.add(left)
            equa_set.add(right)
        self.equa_set = np.array(list(equa_set))
        self.mode = mode

        self.equa2id = {
            self.equa_set[i]:i for i in range(len(self.equa_set))
        }

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.true_realtion = self.get_true_head_and_tail(self.triple_set)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, tail, relation = positive_sample
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(0, len(self.equa_set),size=self.negative_sample_size*2)
            if self.mode == "hr_t":
                mask = np.in1d(
                        negative_sample, 
                        [self.equa2id[tail]], 
                        assume_unique=True, 
                        invert=True
                )
            else:
                mask = np.in1d(
                        negative_sample, 
                        [self.equa2id[head]], 
                        assume_unique=True, 
                        invert=True
                )
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        
        negative_sample = self.equa_set[negative_sample]

        negative_sample_list = []
        for negative in negative_sample:
            negative_sample_list.append(torch.LongTensor(list(negative)))
        
        # negative_sample = torch.LongTensor(negative_sample)
        head = torch.LongTensor(head)
        tail = torch.LongTensor(tail)
        relation = torch.LongTensor([relation])

        return head, tail, relation, negative_sample_list,self.mode
    
    @staticmethod
    def collate_fn(data):
        # 
        head_list = [_[0] for _ in data]
        head = torch.cat(head_list)
        tail_list = [_[1] for _ in data]
        tail = torch.cat(tail_list)

        head_index = []
        for i, tensor in enumerate(head_list):
            head_index.append(torch.full((len(tensor),1), i, dtype=torch.long))
        head_index = torch.cat(head_index,dim=0)

        tail_index = []
        for i, tensor in enumerate(tail_list):
            tail_index.append(torch.full((len(tensor),1), i, dtype=torch.long))
        tail_index = torch.cat(tail_index,dim=0)

        relation = torch.stack([_[2] for _ in data], dim=0)


        negative_sample_list  = [_[3] for _ in data]
        negative_sample_list = list(np.concatenate(negative_sample_list))
    
        negative_sample = torch.cat(list(negative_sample_list))
        neg_index = []
        for i, tensor in enumerate(negative_sample_list):
            neg_index.append(torch.full((len(tensor),1), i, dtype=torch.long))
        neg_index = torch.cat(neg_index,dim=0)

        mode = data[0][4]
        return head,tail, relation, negative_sample, head_index, tail_index,neg_index,mode
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_relation = {}
        for left, right, e in triples:
            if (left, right) not in true_relation:
                true_relation[(left, right)] = []
            true_relation[(left, right)].append(e)
        
        for left, right in true_relation:
            true_relation[(left, right)] = np.array(list(set(true_relation[(left, right)])))
        
        return true_relation



class TestRelationDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
    

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        tmp = [(0, rand_r) if (head, rand_r, tail) not in self.triple_set
                   else (-1, relation) for rand_r in range(self.nrelation)]
        
        tmp[relation] = (0, relation)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))
        return positive_sample, negative_sample, filter_bias
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        return positive_sample, negative_sample, filter_bias


class NagativeRelationSampleDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode=None,head_num=None,tail_num=None,all_traied=None):

        self.len = len(triples)
        self.triples = triples
        if all_traied != None:
            self.triple_set = (all_traied)
        else:
            self.triple_set = (triples)

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.true_realtion = self.get_true_head_and_tail(self.triple_set)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(0, self.nrelation,size=self.negative_sample_size*2)
            mask = np.in1d(
                    negative_sample, 
                    self.true_realtion[(head, tail)], 
                    assume_unique=True, 
                    invert=True
            )
            
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        return positive_sample, negative_sample
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_relation = {}
       

        for head, relation, tail in triples:
            if (head, tail) not in true_relation:
                true_relation[(head, tail)] = []

            true_relation[(head, tail)].append(relation)
           
        for head, tail in true_relation:
            true_relation[(head, tail)] = np.array(list(set(true_relation[(head, tail)])))
                      
        return true_relation


class NagativeSampleDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode,head_num=None,tail_num=None,all_traied=None):

        self.len = len(triples)
        self.triples = triples
        if all_traied != None:
            self.triple_set = (all_traied)
        else:
            self.triple_set = (triples)

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(self.triple_set)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)

        if not head_num is None and not tail_num is None:
            self.head_num = head_num
            self.tail_num = tail_num
        elif head_num is None and tail_num is None:
            self.head_num = self.tail_num = self.nentity
        else:
            print("ERROR")
        

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            
            if self.mode == 'h_rt':
                negative_sample = np.random.randint(self.head_num, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'hr_t':
                negative_sample = np.random.randint(self.tail_num, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
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

class OneToNDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, mode='hr_all', entity2type=None, typeDiff=None,en2typeid=None):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.data_full= self.build_dataset(triples)
        self.data =list(self.data_full.keys())
        self.mode = mode
        # 初始化entity2type
        if  entity2type != None:
            self.entity2tye = [i for i in range(nentity)]
            typeid = 0
            for type in entity2type.keys():
                for index in entity2type[type]:
                    self.entity2tye[index] = typeid
                typeid += 1
            self.entity2tye = np.array(self.entity2tye)
            self.en2typeid = np.array(en2typeid).ravel()
            self.typeDiff = 1.5 - np.array(typeDiff)
            print(self.typeDiff.shape)
            self.all_typeId = self.en2typeid[[i for i in range(self.nentity)]]

    def build_dataset(self,triples):
        data = {}
        for h, r,t in triples:
            if not (h,r) not in data:
                data[(h,r)].append(t)
            else:
                data[(h,r)] = [t]
        return data
        
    def __len__(self):
        return len(self.data)
        
    def get_true_tails(self,h_and_rs):
        true_tails =  torch.zeros(self.nentity)  # 不同的会有不同的要求
        for t in self.data_full[h_and_rs]:
            true_tails[t] = 1
        # truth_typeId = self.en2typeid[self.data_full[h_and_rs]]
        # value =torch.FloatTensor(self.typeDiff[truth_typeId[0]][...,self.all_typeId])
        return true_tails #,value

    def __getitem__(self, idx):
        h_and_rs =self.data[idx]
        true_tails = self.get_true_tails(h_and_rs)
        h_and_rs = torch.tensor(h_and_rs)
       # return h_and_rs, true_tails, value, self.mode
        return h_and_rs, true_tails, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        # subsample_weight = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][2]
        # return positive_sample, negative_sample, subsample_weight, mode
        return positive_sample, negative_sample,  mode

class SimpleTripleDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation):
        self.nentity = nentity
        self.nrelation = nrelation
        self.all_true_triples = all_true_triples
        self.triples = triples
        self.data_full= self.build_dataset(all_true_triples)
        self.data =list(self.data_full.keys())

    def build_dataset(self,triples):
        data = {}
        for h, r,t in triples:
            if not (h,r) not in data:
                data[(h,r)].append(t)
            else:
                data[(h,r)] = [t]
        return data
        
    def __len__(self):
        return len(self.triples)
    
    def get_true_tails(self,h_and_rs):
        true_tails =  torch.zeros(self.nentity)
        for t  in self.data_full[h_and_rs]:
            true_tails[t] = 1
        return true_tails

    def __getitem__(self, idx):
        triple = self.triples[idx]
        h_and_rs = (triple[0],triple[1])
        true_tail = triple[2]
        filter = self.get_true_tails(h_and_rs)
        h_and_rs = torch.tensor(h_and_rs)
        true_tail = torch.tensor(true_tail)
        return h_and_rs, true_tail, filter

class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode,head_num=None,tail_num=None):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        if not head_num is None and not tail_num is None:
            self.head_num = head_num
            self.tail_num = tail_num
        elif head_num is None and tail_num is None:
            self.head_num = self.tail_num = self.nentity
        else:
            print("ERROR")

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        if self.mode == 'h_rt':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.head_num)]
          
            tmp[head] = (0, head)
        elif self.mode == 'hr_t':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.tail_num)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

# 包含多个triples 组合的数据集，用于关系分类测试
class MulTestDataset(Dataset):

    def __init__(self, triples, all_true_triples, nentity, nrelation, mode="hr_t"):
        self.len = len(triples[0])

        self.triples_num = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_samples = []
        negative_samples = []
        filter_biass = []
        for i in range(self.triples_num):
            head, relation, tail = self.triples[i][idx]
            if self.mode == 'h_rt':
                tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                    else (-1, head) for rand_head in range(self.nentity)]
                tmp[head] = (0, head)
            elif self.mode == 'hr_t':
                tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                    else (-1, tail) for rand_tail in range(self.nentity)]
                tmp[tail] = (0, tail)
            else:
                raise ValueError('negative batch mode %s not supported' % self.mode)
            tmp = torch.LongTensor(tmp)            
            filter_bias = tmp[:, 0].float()
            negative_sample = tmp[:, 1]
            positive_sample = torch.LongTensor((head, relation, tail))
            positive_samples.append(positive_sample)
            negative_samples.append(negative_sample)
            filter_biass.append(filter_bias)
        positive_samples = torch.stack(list(positive_samples), dim=0)
        negative_samples = torch.stack(list(negative_samples), dim=0)
        filter_biass = torch.stack(list(filter_biass), dim=0)
        return  positive_samples, negative_samples, filter_biass, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

# 针对两个数据集的构造迭代器
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.len = len(dataloader_head) + len(dataloader_tail)
        self.step = 0
        self.epoch = 0
        
    def __next__(self):
        self.step += 1
        if(self.step //self.len) > self.epoch:
            self.epoch += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class MultiShotItertor(object):
    def __init__(self, a,b,c,d):
        self.a = self.one_shot_iterator(a)
        self.b = self.one_shot_iterator(b)
        self.c = self.one_shot_iterator(c)
        self.d = self.one_shot_iterator(d)
        self.step = 0

    def __next__(self):
        
        if self.step % 4 == 0:
            data = next(self.a)
        elif self.step % 4 == 1:
            data = next(self.b)
        elif self.step % 4 == 2:
            data = next(self.c)
        elif self.step % 4 == 3:
            data = next(self.d)
        self.step += 1
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
# 针对一个数据集构造迭代器
class OneShotIterator(object):
    def __init__(self, dataloader):
        self.dataloader = self.one_shot_iterator(dataloader)
 
    def __next__(self):
        data = next(self.dataloader)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
