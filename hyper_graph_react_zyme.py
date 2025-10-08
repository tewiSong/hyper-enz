from collections import defaultdict
from curses import init_pair
import time
from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP
import os
import numpy as np
from scipy.sparse import csr_matrix
import yaml 
import math
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,MultiStepLR
import torch.nn as nn
from torch.cuda import amp
from loss import NSSAL
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
from util.generate_hyper_graph import *
import logging
from torch.optim.lr_scheduler import StepLR
from torchstat import stat
import argparse
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from util_react.react_zyme_datasets import *
import random
from core.HyperGraphReactZyme import HyperGraphV3
from core.HyperCEBrenda import HyperKGEConfig 

import pickle

def cosine_annealing(epoch, num_epochs, start_lr, end_lr):
    import math
    cos_val = math.cos(math.pi * epoch / num_epochs)
    return end_lr + (start_lr - end_lr) * 0.5 * (1 + cos_val)

def get_noise_sigma(step, max_step,config):
    noise_sigma = config["noise_sigma"]
    noise_sigma = noise_sigma * cosine_annealing(step, max_step, 0, 1)
    return noise_sigma

def get_entity_noise_sigma(weight, step, max_step):
    _3sigma_percent = 0.9973
    noise_sigma = float(torch.topk(weight.reshape(-1).abs(), int(weight.shape[0]*weight.shape[1] * _3sigma_percent), largest=False)[0][-1]) / 3 * 0.1
    noise_sigma = noise_sigma * cosine_annealing(step, max_step, 0, 1)
    return noise_sigma

def logging_log(step, logs,writer):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        writer.add_scalar(metric, metrics[metric], global_step=step, walltime=None)
    logset.log_metrics('Training average', step, metrics)

def set_config(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--valid', action='store_true', help='valid model')
    parser.add_argument('--debug', action='store_true', help='valid model')

    
    parser.add_argument('--max_step', type=int,default=200001, help='最大的训练step')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_step", type=int, default=10000)
    parser.add_argument("--neg_size",type=int, default=256)
    parser.add_argument("--gamma", type=float, default=20)
    parser.add_argument("--adversial_temp", type=float, default=0.5)

    parser.add_argument("--dim", type=int, default=200)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--warm_up_step", type=int, default=50000)

    parser.add_argument("--loss_function", type=str)

    # HAKE 模型的混合权重
    parser.add_argument("--mode_weight",type=float,default=0.5)
    parser.add_argument("--phase_weight",type=float,default=0.5)

    parser.add_argument("--g_type",type=int,default=5)
    parser.add_argument("--g_level",type=int,default=5)

    parser.add_argument("--model",type=str)
    parser.add_argument("--init",type=str)
    parser.add_argument("--configName",type=str)

    parser.add_argument("--g_mode",type=float,default=0.5)
    parser.add_argument("--g_phase",type=float,default=0.5)

    # RotPro 约束参数配置
    parser.add_argument("--gamma_m",type=float,default=0.000001)
    parser.add_argument("--alpha",type=float,default=0.0005)
    parser.add_argument("--beta",type=float,default=1.5)
    parser.add_argument("--train_pr_prop",type=float,default=1)
    parser.add_argument("--loss_weight",type=float,default=1)

    # 选择数据集
    parser.add_argument("--level",type=str,default='ins')
    parser.add_argument("--data_reverse",action='store_true')

    return parser.parse_args(args)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def topk_accuracy(logits, labels, k=1):
    asrt = torch.argsort(logits, dim=1, descending=True)
    if (logits == 0).all(dim=-1).sum():
        rand_perm = torch.stack([torch.randperm(logits.size(1)) for _ in range(logits.size(0))])
        indices = torch.where((logits == 0).all(dim=-1) == 1)[0]
        asrt[indices] = rand_perm[indices]
    
    ranking = torch.empty(logits.shape[0], logits.shape[1], dtype = torch.long).scatter_ (1, asrt, torch.arange(logits.shape[1]).repeat(logits.shape[0], 1))
    ranking = (ranking + 1).to(labels.device)
    mean_rank = (ranking * labels.float()).sum(dim=-1) / (labels.sum(dim=-1)) # (num_seq)
    #mean_rank = (ranking * labels).sum(-1) / (((labels.argsort(dim=-1, descending=True) + 1) * labels).sum(-1))
    mean_rank = mean_rank.mean(dim=0)
    #mrr = (1.0 / ranking * labels.float()).sum(dim=-1) / ((1.0 / (labels.argsort(dim=-1, descending=True) + 1) * labels).sum(-1) + 1e-9)
    mrr = (1.0 / ranking * labels.float()).sum(dim=-1) / (labels.sum(dim=-1)) # (num_seq)
    mrr = mrr.mean(dim=0)
    
    top_accs = []
    top_accs2 = []
    for k in [1, 2, 3, 4, 5, 10, 20, 50]:
        top_acc = ((ranking <= k) * labels.float()).sum(dim=-1) / k
        top_acc = top_acc.mean(dim=0)    
        top_accs.append(top_acc)

        top_acc2 = (((ranking <= k) * labels.float()).sum(dim=-1) > 0).float()
        top_acc2 = top_acc2.mean(dim=0)
        top_accs2.append(top_acc2)
        
    return top_accs[0], top_accs[1], top_accs[2], top_accs[3], top_accs[4], top_accs[5], top_accs[6], top_accs[7], top_accs2[0], top_accs2[1], top_accs2[2], top_accs2[3], top_accs2[4], top_accs2[5], top_accs2[6], top_accs2[7], mean_rank, mrr



def test_step_function(model, test_dataset, args, loss_function=None):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()

        logs = []
        step = 0
        total_steps = len(test_dataset)
        pre_list = []
        index_list = []
        batch_seq = 2048
        with torch.no_grad():
            for triples,labels, head_out,tail_out in test_dataset:
                batch_size = torch.sum(labels).numpy().astype(int)
                n_size = int(len(labels)//batch_size)

                head_emb = model.single_emb(head_out)
                tail_emb = model.single_emb(tail_out)

                # if args["cuda"]:
                #     head = head.cuda()
                #     tail = tail.cuda()
                #     relation = relation.cuda()
                #     negative_sample = negative_sample.cuda()
                #     filter_bias = filter_bias.cuda()
                
                # all_scores = []
                # for i in range(n_size//batch_seq+1):
                #     negative_score = model.full_score(head_emb, negative_sample[:,i*batch_seq:(i+1)*batch_seq],tail_emb)
                #     negative_score = negative_score.view(batch_size, -1)
                #     all_scores.append(negative_score)
                #     torch.cuda.empty_cache()
                # if negative_score == None: continue
                score = model.full_score(head_emb, triples[:,1],tail_emb)
                # all_scores = torch.cat(all_scores, dim=1)
                # score = all_scores + filter_bias
                score = score.reshape(2,-1)
                pre_list.append(score.cpu())
                # index_list.append(relation.cpu())
                argsort = torch.argsort(score, dim = 1, descending=True)
                for i in range(2):
                    # ranking = (argsort[i, :] == relation[i]).nonzero()
                    ranking = (argsort[i, :] == 1).nonzero()
                    assert ranking.size(0) == 1
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })
                if step % 1000 == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                step += 1
                # if step >= 10: break
        metrics = {}

        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        torch.cuda.empty_cache()
        metrics["total step"] = total_steps
        preds = torch.cat(pre_list, dim=0)
        labels = torch.zeros_like(preds)
        for i in range(len(index_list)):
            labels[i][index_list[i]] = 1
        return metrics,preds,labels

if __name__=="__main__":
    # 读取4个数据集
    setup_seed(20)
    args = set_config()
    with open('./config/hypergraph_brenda_07_all_data.yml','r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        baseConfig = config['baseConfig']
        modelConfig = config[args.configName]

    cuda = baseConfig['cuda']
    # cuda = False
    init_step  = 0
    save_steps = baseConfig['save_step']
    n_size = modelConfig['n_size']
    log_steps = 100
    root_path = os.path.join("./models/",args.save_path)
    
    args.save_path = root_path
    args.loss_weight = modelConfig['reg_weight']
    args.batch_size = modelConfig['batch_size']
    init_path = args.init
    max_step   = modelConfig['max_step']
    batch_size = args.batch_size
    test_step = modelConfig['test_step']
    dim = modelConfig['dim']
    lr = modelConfig['lr']
    decay = modelConfig['decay']
    args.data_reverse = modelConfig['data_reverse']
    torch.seed
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(os.path.join(root_path,'log')):
        os.makedirs(os.path.join(root_path,'log'))
    writer = SummaryWriter(os.path.join(root_path,'log'))
    if args.train:
        logset.set_logger(root_path,"train.log")
    else:
        logset.set_logger(root_path,'test.log')
    
    # 读取数据集
    logging.info('Model: %s begining load data' % modelConfig['name'])
    train_dataset,valid_dataset,test_dataset,graph_info,train_info,smileGraphDataset, clDataset,train_test,cl_dataset = build_graph_sampler(modelConfig)
    logging.info('build trainning dataset....')

    base_loss_funcation = NSSAL(gamma=modelConfig["gamma"], plus_gamma=False)
    sub_loss_function = NSSAL(gamma=modelConfig["gamma_s"], plus_gamma=True)
    type_loss_function = NSSAL(gamma=modelConfig["gamma_t"], plus_gamma=True)


    hyperConfig = HyperKGEConfig()
    hyperConfig.embedding_dim = modelConfig['dim']
    hyperConfig.conv_args_conv_dropout_rate = modelConfig['dropout']
    hyperConfig.gamma = modelConfig['gamma']
    n_node = graph_info['base_node_num']
    help_data = None

    model = HyperGraphV3(hyperkgeConfig=hyperConfig,n_node=n_node, n_hyper_edge=graph_info["max_edge_id"]-n_node,e_num=graph_info['e_num'],graph_info=graph_info,config=modelConfig,NodeGnnDataset=smileGraphDataset,clDataset=clDataset)
    c_embedding_table = torch.load("./pre_handle_data/react_zyme/c_emb_table.pt")
    e_embedding_table = torch.load("./pre_handle_data/react_zyme/e_emb_table.pt")
    model.init_embedding(c_embedding_table,e_embedding_table)
    if cuda:
        model = model.cuda()

    # for name,param in model.named_parameters():
        # print(name)
    params = [param for name,param in model.named_parameters() ]
    
    # 给优化器设置正则
    optimizer = torch.optim.Adam([
        {
         'params':filter(lambda p: p.requires_grad , params), 
         'lr': lr
        }
        ], lr=lr
    )
    result = get_parameter_number(model)
    logging.info("模型总大小为：%s" % str(result["Total"]))
    # 如果-有保-存模-型则，读取-模型,进行-测试
    init_path = None
    if init_path != None:   
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        init_step = checkpoint['step']

    logging.info('Model: %s' % modelConfig['name'])
    logging.info('Instance nentity: %s' % graph_info['c_num'])
    logging.info('Instance nrelatidataset. %s' % graph_info['e_num'])
    logging.info('max step: %s' % max_step)
    logging.info('init step: %s' % init_step)
    logging.info('lr: %s' % lr)

    # 设置学习率更新策略
    lr_scheduler = MultiStepLR(optimizer,milestones=[00], gamma=decay)
    logAll = []
   
    stepW = 0
    bestModel = {
        "MRR":0,
        "MR":1000000000,
        "HITS@1":0,
        "HITS@3":0,
        "HITS@10":0,
        "Mix":0,
    }
    baselog = []
    conf_cllog = []
    vio_cllog = []
    scaler = torch.amp.GradScaler('cuda')
    if args.train :
        logging.info('beging trainning')
        for step in range(init_step, max_step):
            begin_time = time.time()
            if step % 10 == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,'ConfigName':args.configName
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

            if step % test_step == 0 and step != 0:
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,'ConfigName':args.configName
                }
                logging.info('Valid InstanceOf at step: %d' % step)
                metrics,preds,labels = test_step_function(model, valid_dataset,modelConfig)
                metrics["Mix"] = (metrics["HITS@1"] + metrics["HITS@3"] + metrics["HITS@10"]) / 3
                for key in metrics:
                    writer.add_scalar(key, metrics[key], global_step=step, walltime=None)
                logset.log_metrics('Valid ', step, metrics)
                ModelUtil.save_best_model(metrics=metrics,best_metrics=bestModel,model=model,optimizer=optimizer,save_variable_list=save_variable_list,args=args)
            train_time_handle = []
            train_begin = time.time()
            logging.info("one epoch train step: %d" % (len(train_dataset)))
            sub_step = 0

            for data in train_dataset:
                log = HyperGraphV3.train_step(model=model,optimizer=optimizer,data=data,config=modelConfig,scaler=scaler)
                baselog.append(log)
                writer.add_scalar('Loss/train', log['loss'], step*len(train_dataset) + sub_step)
                sub_step += 1

            if step % 5 == 0:
                try:
                    logging_log(step, baselog, writer)
                    baselog=[]
                    time_used = time.time() - begin_time
                    begin_time = time.time()
                    logging.info("epoch %d used time: %.3f" % (step, time_used))
                except Exception as e:
                    logging.info("epoch %d test error: %s" % (str(e)))
            lr_scheduler.step()
        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":max_step,'ConfigName':args.configName
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)
        
        init_path = os.path.join(root_path,"hit10")
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)

        logging.info('Test InstanceOf at step: %d' % checkpoint['step'])
        metrics,preds,labels = test_step_function(model, test_dataset,modelConfig)
        logset.log_metrics('Test ',checkpoint['step'], metrics)
        
    else:
        logging.info('Test InstanceOf at step: %d' % checkpoint['step'])
        metrics,preds,labels = test_step_function(model, train_test,modelConfig)
        logset.log_metrics('Test ',checkpoint['step'], metrics)

        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc, top10_acc, top20_acc, top50_acc, top1_acc2, top2_acc2, top3_acc2, top4_acc2, top5_acc2, top10_acc2, top20_acc2, top50_acc2, mean_rank, mrr = topk_accuracy(preds.detach().cpu(), labels.detach().cpu())
        print(f'Pred Top1 Acc-N: {top1_acc:.4f}, Top2 Acc-N: {top2_acc:.4f}, Top3 Acc-N: {top3_acc:.4f}, Top4 Acc-N: {top4_acc:.4f}, Top5 Acc-N: {top5_acc:.4f}, Top10 Acc-N: {top10_acc:.4f}, Top20 Acc-N: {top20_acc:.4f},  Top50 Acc-N: {top50_acc:.4f}, Top1 Acc: {top1_acc2:.4f}, Top2 Acc: {top2_acc2:.4f}, Top3 Acc: {top3_acc2:.4f}, Top4 Acc: {top4_acc2:.4f}, Top5 Acc: {top5_acc2:.4f}, Top10 Acc: {top10_acc2:.4f}, Top20 Acc: {top20_acc2:.4f},  Top50 Acc: {top50_acc2:.4f}, Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}')
        
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc, top10_acc, top20_acc, top50_acc, top1_acc2, top2_acc2, top3_acc2, top4_acc2, top5_acc2, top10_acc2, top20_acc2, top50_acc2, mean_rank, mrr = topk_accuracy(labels.detach().cpu(), labels.detach().cpu())
        print(f'Data Top1 Acc-N: {top1_acc:.4f}, Top2 Acc-N: {top2_acc:.4f}, Top3 Acc-N: {top3_acc:.4f}, Top4 Acc-N: {top4_acc:.4f}, Top5 Acc-N: {top5_acc:.4f}, Top10 Acc-N: {top10_acc:.4f}, Top20 Acc-N: {top20_acc:.4f},  Top50 Acc-N: {top50_acc:.4f}, Top1 Acc: {top1_acc2:.4f}, Top2 Acc: {top2_acc2:.4f}, Top3 Acc: {top3_acc2:.4f}, Top4 Acc: {top4_acc2:.4f}, Top5 Acc: {top5_acc2:.4f}, Top10 Acc: {top10_acc2:.4f}, Top20 Acc: {top20_acc2:.4f},  Top50 Acc: {top50_acc2:.4f}, Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}')
        
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc, top10_acc, top20_acc, top50_acc, top1_acc2, top2_acc2, top3_acc2, top4_acc2, top5_acc2, top10_acc2, top20_acc2, top50_acc2, mean_rank, mrr = topk_accuracy(preds.transpose(0,1).detach().cpu(), labels.transpose(0,1).detach().cpu())
        print(f'Pred Transpose Top1 Acc-N: {top1_acc:.4f}, Top2 Acc-N: {top2_acc:.4f}, Top3 Acc-N: {top3_acc:.4f}, Top4 Acc-N: {top4_acc:.4f}, Top5 Acc-N: {top5_acc:.4f}, Top10 Acc-N: {top10_acc:.4f}, Top20 Acc-N: {top20_acc:.4f},  Top50 Acc-N: {top50_acc:.4f}, Top1 Acc: {top1_acc2:.4f}, Top2 Acc: {top2_acc2:.4f}, Top3 Acc: {top3_acc2:.4f}, Top4 Acc: {top4_acc2:.4f}, Top5 Acc: {top5_acc2:.4f}, Top10 Acc: {top10_acc2:.4f}, Top20 Acc: {top20_acc2:.4f},  Top50 Acc: {top50_acc2:.4f}, Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}')
        
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc, top10_acc, top20_acc, top50_acc, top1_acc2, top2_acc2, top3_acc2, top4_acc2, top5_acc2, top10_acc2, top20_acc2, top50_acc2, mean_rank, mrr = topk_accuracy(labels.transpose(0,1).detach().cpu(), labels.transpose(0,1).detach().cpu())
        print(f'Data Transpose Top1 Acc-N: {top1_acc:.4f}, Top2 Acc-N: {top2_acc:.4f}, Top3 Acc-N: {top3_acc:.4f}, Top4 Acc-N: {top4_acc:.4f}, Top5 Acc-N: {top5_acc:.4f}, Top10 Acc-N: {top10_acc:.4f}, Top20 Acc-N: {top20_acc:.4f},  Top50 Acc-N: {top50_acc:.4f}, Top1 Acc: {top1_acc2:.4f}, Top2 Acc: {top2_acc2:.4f}, Top3 Acc: {top3_acc2:.4f}, Top4 Acc: {top4_acc2:.4f}, Top5 Acc: {top5_acc2:.4f}, Top10 Acc: {top10_acc2:.4f}, Top20 Acc: {top20_acc2:.4f},  Top50 Acc: {top50_acc2:.4f}, Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}')