
import torch
import numpy as np
import os
import json

import torch.nn.functional as F
from util.dataloader import TestDataset,TestReactionRelationDataset
import logging
from util.dataloader import MulTestDataset
from torch.utils.data import DataLoader
from util.tools import logset

class ModelUtil(object):
    @staticmethod
    def save_best_model(metrics,best_metrics,model, optimizer,save_variable_list,args):
        if metrics['MR'] < best_metrics['MR']:
            best_metrics['MR'] = metrics['MR']
            logging.info('Save models for best MR until now')
            ModelUtil.save_model(model, optimizer, save_variable_list, args,type='MR')
        if metrics['MRR'] > best_metrics['MRR']:
            best_metrics['MRR'] = metrics['MRR']
            logging.info('Save models for best MRR until now')
            ModelUtil.save_model(model, optimizer, save_variable_list, args,type='MRR')
        if metrics['HITS@1'] > best_metrics['HITS@1']:
            best_metrics['HITS@1'] = metrics['HITS@1']
            logging.info('Save models for best Hit1 until now')
            ModelUtil.save_model(model, optimizer, save_variable_list, args, type='hit1')
        if metrics['HITS@3'] > best_metrics['HITS@3']:
            best_metrics['HITS@3'] = metrics['HITS@3']
            logging.info('Save models for best Hit3 until now')
            ModelUtil.save_model(model, optimizer, save_variable_list, args,type='hit3')
        if metrics['HITS@10'] > best_metrics['HITS@10']:
            best_metrics['HITS@10'] = metrics['HITS@10']
            logging.info('Save models for best Hit10 until now')
            ModelUtil.save_model(model, optimizer, save_variable_list, args, type='hit10')

    @staticmethod
    def init_model(model, optimizer,path):
        checkpoint = torch.load(os.path.join(path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        current_learning_rate = checkpoint['current_learning_rate']
        warm_up_steps = checkpoint['warm_up_steps']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @staticmethod
    def save_model(model, optimizer, save_variable_list=None, args=None, type=None, path=None):
        '''W
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        '''
        if args != None:
            root_path = args.save_path
        else:
            root_path = path

        if type is not None:
            root_path = os.path.join(root_path,type)
            if  not os.path.exists(root_path):
                os.makedirs(root_path)
        if args != None:
            argparse_dict = vars(args)
            with open(os.path.join(root_path, 'config.json'), 'w') as fjson:
                json.dump(argparse_dict, fjson)
        print("save dict: ", root_path)
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(root_path, 'checkpoint')
        )

       
class ReactionTrainer(object):
    # train_type:  NagativeSample and 1toN
    def __init__(self, data, train_iterator, model, optimizer, loss_function, args, lr_scheduler=None,logging=None,train_type='NagativeSample',entity_iter=None):
        self.args = args
        self.train_type = train_type
        self.model = model
        self.optimizer = optimizer
        self.step = 0
        self.data = data
        self.max_step = args.max_steps
        self.train_iterator = train_iterator
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.entity_iter=entity_iter
        self.logging = logging
        self.best_metrics = {'MRR':0.0, 'MR':1000000000, 'HITS@1':0.0,'HITS@3':0.0,'HITS@10':0.0}

    def _init_model(self):
        if self.args.init_checkpoint:
            checkpoint = torch.load(os.path.join(self.args.init_checkpoint, 'checkpoint'))
            self.init_step = checkpoint['step']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_learning_rate = checkpoint['current_learning_rate']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.init_step = 0

    def logging_model_info(self):
        pass

    def logging_traing_info(self):
        logging = self.logging
        logging.info('Start Training...')
        logging.info('init_step = %d' % self.step)
        logging.info('max_step = %d' % self.max_step)
        logging.info('batch_size = %d' % self.args.batch_size)
        logging.info('negative_adversarial_sampling = %d' % self.args.negative_adversarial_sampling)
        logging.info('hidden_dim = %d' % self.args.hidden_dim)
        logging.info('gamma = %f' % self.args.gamma)
        logging.info('negative_adversarial_sampling = %s' % str(self.args.negative_adversarial_sampling))
        if self.args.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % self.args.adversarial_temperature)

    def train_model_(self):
        ReactionTrainer.train_model(data=self.data,train_iterator=self.train_iterator,
                    model=self.model,optimizer=self.optimizer,loss_function=self.loss_function,
                    max_step=self.max_step, init_step=self.step,
                    args=self.args,best_metrics=self.best_metrics,
                    lr_scheduler=self.lr_scheduler,train_type=self.train_type,entity_iter=self.entity_iter)

    # 训练负采样的数据集: 数据集产生正样本和负样本, 损失基于正负样本的计算
    @staticmethod
    def train_step(model, optimizer, train_iterator,loss_function, args,entity_iter=None):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        head,tail, relation, negative_sample, head_index, tail_index = next(train_iterator)

        if args.cuda:
            head = head.cuda()
            tail = tail.cuda()
            relation = relation.cuda()
            head_index = head_index.cuda()
            tail_index = tail_index.cuda()
            negative_sample = negative_sample.cuda()
       
        negative_score = model.full_score(head,head_index,tail,tail_index,negative_sample)
        positive_score= model.full_score(head,head_index,tail,tail_index,relation)
        loss = loss_function(positive_score, negative_score)
        log = {}
        log["rel_loss"] = loss.item()
        
        if entity_iter != None:

            head,tail, relation, negative_sample, head_index, tail_index,neg_index,mode = next(entity_iter)

            if args.cuda:
                head = head.cuda()
                tail = tail.cuda()
                relation = relation.cuda()
                head_index = head_index.cuda()
                tail_index = tail_index.cuda()
                negative_sample = negative_sample.cuda()
                neg_index = neg_index.cuda()
            
            positive_score= model.full_score(head,head_index,tail,tail_index,relation)
            if mode == "hr_t":
                negative_score = model.full_score(head,head_index,negative_sample,neg_index,relation,mode)
            else:
                negative_score = model.full_score(negative_sample,neg_index,tail,tail_index,relation,mode)
            
            entity_loss = loss_function(positive_score, negative_score)
            loss += entity_loss
            log = {
            "entity loss":entity_loss.item(),
            }

        regularization = 0
        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.weight.data.norm(p = 3)**3 + 
                model.relation_embedding.weight.data.norm(p = 3).norm(p = 3)**3
            )
            loss += regularization
        loss.backward()
        optimizer.step()
        log["loss"] = loss.item()
        log["regularization"] = regularization
      
        return log
  
    @staticmethod
    def train_model(data, train_iterator, model, optimizer,loss_function, max_step, init_step,args,best_metrics,lr_scheduler=None, train_type="NagativeSample",entity_iter=None):
        training_logs = []

        if train_type == 'NagativeSample':
            TRAIN_STEP = ReactionTrainer.train_step
        else:
            TRAIN_STEP = ReactionTrainer.train_step_1

        for step in range(init_step, max_step):
            log = TRAIN_STEP(model, optimizer,train_iterator,loss_function,args,entity_iter)
            training_logs.append(log)
            # if lr_scheduler != None:
            lr_scheduler.step()
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                }
                ModelUtil.save_model(model,optimizer,save_variable_list,args)
            if  step % args.valid_steps == 0 and step != 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                }
                metrics = ModelTester.test_step(model, data["valid"], data["all_true_triples"], args)
                logset.log_metrics('Valid', step, metrics)
                ModelUtil.save_best_model(metrics,best_metrics, model,optimizer,save_variable_list,args)
            
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                logset.log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': max_step, 
            'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
            # todo 其他需要保存的参数
        }
        ModelUtil.save_model(model,optimizer,save_variable_list,args)

# class Trainer(object):
#     # train_type:  NagativeSample and 1toN
#     def __init__(self, data, train_iterator, model, optimizer, loss_function, args, lr_scheduler=None,logging=None,train_type='NagativeSample'):
#         self.args = args
#         self.train_type = train_type
#         self.model = model
#         self.optimizer = optimizer
#         # 
#         self.step = 0
#         # self.step = args.init_step
#         self.data = data
#         self.max_step = args.max_steps
#         self.train_iterator = train_iterator
#         self.lr_scheduler = lr_scheduler
#         self.loss_function = loss_function
#         self.logging = logging
#         self.best_metrics = {'MRR':0.0, 'MR':1000000000, 'HITS@1':0.0,'HITS@3':0.0,'HITS@10':0.0}

#     def _init_model(self):
#         if self.args.init_checkpoint:
#             checkpoint = torch.load(os.path.join(self.args.init_checkpoint, 'checkpoint'))
#             self.init_step = checkpoint['step']
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.current_learning_rate = checkpoint['current_learning_rate']
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         else:
#             self.init_step = 0

#     def logging_model_info(self):
#         pass

#     def logging_traing_info(self):
#         logging = self.logging
#         logging.info('Start Training...')
#         logging.info('init_step = %d' % self.step)
#         logging.info('max_step = %d' % self.max_step)
#         logging.info('batch_size = %d' % self.args.batch_size)
#         logging.info('negative_adversarial_sampling = %d' % self.args.negative_adversarial_sampling)
#         logging.info('hidden_dim = %d' % self.args.hidden_dim)
#         logging.info('gamma = %f' % self.args.gamma)
#         logging.info('negative_adversarial_sampling = %s' % str(self.args.negative_adversarial_sampling))
#         if self.args.negative_adversarial_sampling:
#             logging.info('adversarial_temperature = %f' % self.args.adversarial_temperature)

#     def train_model_(self):
#         Trainer.train_model(data=self.data,train_iterator=self.train_iterator,
#                     model=self.model,optimizer=self.optimizer,loss_function=self.loss_function,
#                     max_step=self.max_step, init_step=self.step,
#                     args=self.args,best_metrics=self.best_metrics,
#                     lr_scheduler=self.lr_scheduler,train_type=self.train_type)

#     # 数据集产生 ground truth：损失基于ground truth 和 预测结果计算
#     @staticmethod
#     def train_step_1(model, optimizer, train_iterator,loss_function,args):
#         model.train()
#         optimizer.zero_grad()
#         h_and_rs, ground_truth,value, mode = next(train_iterator)

#         if args.cuda:
#             h_and_rs = h_and_rs.cuda()
#             ground_truth = ground_truth.cuda()
#             value = value.cuda()
#         ground_truth.detach()   
#         value.detach()
#         ground_truth = ground_truth 
#         if args.label_smoothing != 0.0:
            
#             ground_truth = ((1.0-args.label_smoothing)*ground_truth) + (1.0/ground_truth.size(1))  
            
#         if mode[0] == 'hr_all':
#             h = h_and_rs[:,0]
#             r = h_and_rs[:,1]
#             pre_score = model(h,r, None, mode=mode[0])
#         else:
#             t = h_and_rs[:,0]
#             r = h_and_rs[:,1]
#             pre_score = model(None,r, t, mode=mode[0])
#         pre_score = torch.sigmoid(pre_score*value)
      
#         loss = loss_function(pre_score,ground_truth)
#         loss.backward()
#         optimizer.step()
#         log = {
#             'loss': loss.item()
#         }
#         return log

#     # 训练负采样的数据集: 数据集产生正样本和负样本, 损失基于正负样本的计算
#     @staticmethod
#     def train_step(model, optimizer, train_iterator,loss_function, args):
#         '''
#         A single train step. Apply back-propation and return the loss
#         '''

#         model.train()
#         optimizer.zero_grad()
#         positive_sample,negative_sample = next(train_iterator)

#         if args.cuda:
#             positive_sample = positive_sample.cuda()
#             negative_sample = negative_sample.cuda()
       
#         h = positive_sample[:,0]
#         r = positive_sample[:,1]
#         t = positive_sample[:,2]
#         negative_score = model(h,negative_sample,t)
#         positive_score= model(h,r,t)
#         loss = loss_function(positive_score, negative_score)

#         regularization = 0
#         if args.regularization != 0.0:
#             regularization = args.regularization * (
#                 model.entity_embedding.weight.data.norm(p = 3)**3 + 
#                 model.relation_embedding.weight.data.norm(p = 3).norm(p = 3)**3
#             )
#             loss += regularization
#         loss.backward()
#         optimizer.step()

#         log = {
#             'regularization':regularization,
#             'loss': loss.item()
#         }
#         return log
#     # @staticmethod
#     # def train_step(model, optimizer, train_iterator,loss_function, args):
#     #     '''
#     #     A single train step. Apply back-propation and return the loss
#     #     '''

#     #     model.train()
#     #     optimizer.zero_grad()
#     #     positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)

#     #     if args.cuda:
#     #         positive_sample = positive_sample.cuda()
#     #         negative_sample = negative_sample.cuda()
#     #         subsampling_weight = subsampling_weight.cuda()
       
#     #     h = positive_sample[:,0]
#     #     r = positive_sample[:,1]
#     #     t = positive_sample[:,2]
#     #     if mode == 'hr_t':
#     #         negative_score = model(h,r, negative_sample, mode=mode)
#     #     else:
#     #         negative_score = model(negative_sample,r,t, mode=mode)

#     #     positive_score= model(h,r,t)
#     #     loss = loss_function(positive_score, negative_score,subsampling_weight)

#     #     regularization = 0
#     #     if args.regularization != 0.0:
#     #         regularization = args.regularization * (
#     #             model.entity_embedding.weight.data.norm(p = 3)**3 + 
#     #             model.relation_embedding.weight.data.norm(p = 3).norm(p = 3)**3
#     #         )
#     #         loss += regularization
#     #     loss.backward()
#     #     optimizer.step()

#     #     log = {
#     #         'regularization':regularization,
#     #         'loss': loss.item()
#     #     }
#     #     return log

#     @staticmethod
#     def train_model(data, train_iterator, model, optimizer,loss_function, max_step, init_step,args,best_metrics,lr_scheduler=None, train_type="NagativeSample"):
#         training_logs = []

#         if train_type == 'NagativeSample':
#             TRAIN_STEP = Trainer.train_step
#         else:
#             TRAIN_STEP = Trainer.train_step_1

#         for step in range(init_step, max_step):
#             log = TRAIN_STEP(model, optimizer,train_iterator,loss_function,args)
#             training_logs.append(log)
#             # if lr_scheduler != None:
#             lr_scheduler.step()
#             if step % args.save_checkpoint_steps == 0:
#                 save_variable_list = {
#                     'step': step, 
#                     'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
#                 }
#                 ModelUtil.save_model(model,optimizer,save_variable_list,args)
#             if  step % args.valid_steps == 0 and step !=0:
#                 save_variable_list = {
#                     'step': step, 
#                     'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
#                 }
#                 metrics = ModelTester.test_step(model, data.valid, data.all_true_triples, args)
#                 logset.log_metrics('Valid', step, metrics)
#                 ModelUtil.save_best_model(metrics,best_metrics, model,optimizer,save_variable_list,args)
            
#             if step % args.log_steps == 0:
#                 metrics = {}
#                 for metric in training_logs[0].keys():
#                     metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
#                 logset.log_metrics('Training average', step, metrics)
#                 training_logs = []

#         save_variable_list = {
#             'step': max_step, 
#             'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
#             # todo 其他需要保存的参数
#         }
#         ModelUtil.save_model(model,optimizer,save_variable_list,args)

class ModelTester(object):
    def __init__(self,model):
        self.model = model
    
    @staticmethod
    def class_test(model, tiple_list,all_true_triples,relation_types,args):
        for i in range(len(relation_types)):
            logging.info("Begin test relation %s " % relation_types[i])
            args.test_relation=relation_types[i]
            metrics =  ModelTester.class_test_step(model, tiple_list[i], all_true_triples, args,test_relation=relation_types[i]) 
            logset.log_metrics('Test', 0, metrics)

    @staticmethod
    def class_test_step(model, test_triples, all_true_triples, relation, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        size_triples = len(test_triples)
        test_dataloader_head= DataLoader(
                MulTestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'h_rt'
                ), 
                batch_size=4,
                num_workers=1
            )
        test_dataloader_tail = DataLoader(
                MulTestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'hr_t'
                ), 
                batch_size=4,
                num_workers=1
            )
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        test_num = 0
        step_hit10 = 0
        step_hit1 = 0
        step_hit3 = 0
        hit_false = 50
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_samples, negative_samples, filter_biass, mode in test_dataset:
                    step +=1
                    if step % 1000 == 0:
                        logging.info("ClassTest %d / %d" % (step, total_steps))
                    if args.cuda:
                        positive_samples = positive_samples.cuda()
                        negative_samples = negative_samples.cuda()
                        filter_biass = filter_biass.cuda()
                    positive_samples = list(positive_samples.split(1,1))
                    negative_samples = list(negative_samples.split(1,1))
                    filter_biass = list(filter_biass.split(1,1))
                    ranks=[[] for i in range(size_triples)]
                    mode = mode[0]
                    for triple_sort in range(size_triples):
                        positive_sample = torch.squeeze(positive_samples[triple_sort],dim=1) 
                        negative_sample = torch.squeeze(negative_samples[triple_sort],dim=1)
                        filter_bias = torch.squeeze(filter_biass[triple_sort],dim=1)
                        batch_size = positive_sample.size(0)
                        h = positive_sample[:,0]
                        r = positive_sample[:,1]
                        t = positive_sample[:,2]
                        if mode == 'hr_t':
                            score = model(h,r, negative_sample, mode=mode)
                        else:
                            score = model(negative_sample,r,t, mode=mode)
                        score += filter_bias
                        argsort = torch.argsort(score, dim = 1, descending=True)
                        if mode == 'h_rt':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'hr_t':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)
                        for i in range(batch_size):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1
                            ranking = 1 + ranking.item()    
                            ranks[triple_sort].append(ranking)

                    for ks in range(len(ranks[0])):
                        ranking = None
                        is_hit10 = 0.0
                        is_hit3 = 0.0
                        is_hit1 = 0.0
                        test_num += 1
                        if relation == 'symmetry' or relation == 'inverse':
                            ranking = ranks[1][ks]
                            if ranks[0][ks] > 10 :         
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking <= 10 else 0.0
                                if ranks[0][ks] > 3 :           # 
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking <= 3.0 else 0.0
                                    if ranks[0][ks] >1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking <=1 else 0.0
                        elif relation == 'asymmetry':
                            ranking = ranks[1][ks] 
                            if ranks[0][ks] > 10 :   
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking > hit_false else 0.0 
                                if ranks[0][ks] > 3 :            
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking > hit_false else 0.0
                                    if ranks[0][ks] >1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking > hit_false else 0.0                      
                        elif relation == 'transitive' or relation == 'composition':
                            ranking = ranks[2][ks]
                            if ranks[0][ks] > 10 or ranks[1][ks] > 10:     
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking <= 10 else 0.0
                                if ranks[0][ks] > 3 or ranks[1][ks] > 3:           # 
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking <= 3.0 else 0.0
                                    if ranks[0][ks] >1 or ranks[1][ks] > 1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking <=1 else 0.0
                        logs.append({
                                'HITS@1': is_hit1,
                                'HITS@3': is_hit3,
                                'HITS@10': is_hit10,
                        })                   

        metrics={}
        if len(logs) == 0:
            return metrics
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])
        metrics['HITS@1'] =  metrics['HITS@1']/(test_num)
        metrics['HITS@3'] = metrics['HITS@3']/(test_num)
        metrics['HITS@10'] = metrics['HITS@10']/(test_num)
        metrics['step_hit1'] = step_hit1
        metrics['step_hit3'] = step_hit3
        metrics['step_hit10'] = step_hit10
        metrics['total_num'] = test_num
        return metrics
    
    # # 可以输出测试的loss，但是目前仅支持Pair Loss.
    # @staticmethod
    # def test_step(model, test_triples, all_true_triples, args, loss_function=None):
    #     '''
    #     Evaluate the model on test or valid datasets
    #     '''
    #     model.eval()
    #     test_dataloader_head = DataLoader(
    #         TestDataset(
    #             test_triples, 
    #             all_true_triples, 
    #             args.nentity, 
    #             args.nrelation, 
    #             'h_rt'
    #         ), 
    #         batch_size=args.test_batch_size,
    #         num_workers=1, 
    #         collate_fn=TestDataset.collate_fn
    #     )
    #     test_dataloader_tail = DataLoader(
    #         TestDataset(
    #             test_triples, 
    #             all_true_triples, 
    #             args.nentity, 
    #             args.nrelation, 
    #             'hr_t'
    #         ), 
    #         batch_size=args.test_batch_size,
    #         num_workers=1, 
    #         collate_fn=TestDataset.collate_fn
    #     )
    #     test_dataset_list = [test_dataloader_head, test_dataloader_tail]
    #     logs = []
    #     step = 0
    #     total_steps = sum([len(dataset) for dataset in test_dataset_list])
    #     with torch.no_grad():
    #         for test_dataset in test_dataset_list:
    #             for positive_sample, negative_sample, filter_bias, mode in test_dataset:
    #                 batch_size = positive_sample.size(0)
    #                 if args.cuda:
    #                     positive_sample = positive_sample.cuda()
    #                     negative_sample = negative_sample.cuda()
    #                     filter_bias = filter_bias.cuda()
                        
    #                 h = positive_sample[:,0]
    #                 r = positive_sample[:,1]
    #                 t = positive_sample[:,2] 
    #                 if mode == 'hr_t':
    #                     negative_score = model(h,r, negative_sample, mode=mode)
    #                 else:
    #                     negative_score = model(negative_sample,r,t, mode=mode)
                    
    #                 if loss_function != None:
    #                     positive_score = model(h,r, t)
    #                     loss = loss_function(positive_score,negative_score)
    #                     logging.info("loss of test is %f" % loss)

    #                 score = negative_score + filter_bias
    #                 argsort = torch.argsort(score, dim = 1, descending=True)
    #                 if mode == 'h_rt':
    #                     positive_arg = h
    #                 elif mode == 'hr_t':
    #                     positive_arg = t
    #                 else:
    #                     raise ValueError('mode %s not supported' % mode)
                    
    #                 for i in range(batch_size):
    #                     ranking = (argsort[i, :] == positive_arg[i]).nonzero()
    #                     assert ranking.size(0) == 1
    #                     ranking = 1 + ranking.item()
    #                     logs.append({
    #                         'MRR': 1.0/ranking,
    #                         'MR': float(ranking),
    #                         'HITS@1': 1.0 if ranking <= 1 else 0.0,
    #                         'HITS@3': 1.0 if ranking <= 3 else 0.0,
    #                         'HITS@10': 1.0 if ranking <= 10 else 0.0,
    #                     })
    #                 if step % args.test_log_steps == 0:
    #                     logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
    #                 step += 1
    #     metrics = {}
    #     for metric in logs[0].keys():
    #         metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    #     return metrics


    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, loss_function=None):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        test_dataloader_head = DataLoader(
            TestReactionRelationDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
            ), 
            batch_size=args.test_batch_size,
            num_workers=1, 
            collate_fn=TestReactionRelationDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head]
        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for head, tail, relation, negative_sample, filter_bias, head_index, tail_index in test_dataset:
                    batch_size = relation.size(0)
                    if args.cuda:
                        head = head.cuda()
                        tail = tail.cuda()
                        relation = relation.cuda()
                        head_index = head_index.cuda()
                        tail_index = tail_index.cuda()

                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                  
                    negative_score = model.full_score(head, head_index,tail, tail_index, negative_sample)
                    score = negative_score + filter_bias
                    argsort = torch.argsort(score, dim = 1, descending=True)
                    positive_arg = relation
                    
                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })
                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                    step += 1
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        return metrics

