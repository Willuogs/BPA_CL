#!/usr/bin/env python
from __future__ import print_function

import argparse
from argparse import ArgumentParser
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict,defaultdict
import traceback
from model.baseline import TextCLIP
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import warnings
warnings.filterwarnings("ignore", message=".*gather.*")

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from yaml import FullLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction
from tools import *
from Text_Prompt01 import *
from KLLoss import KLLoss
import resource
from visualize import record_skeleton, wrong_analyze

classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part(use_paraphrase=True,prob_t5 = 0.2)
text_list = text_prompt_openai_random(use_swap=True, use_paraphrase=True,prob_swap=0.3, prob_t5=0.2)



device = "cuda" if torch.cuda.is_available() else "cpu"

scaler = torch.cuda.amp.GradScaler()




rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):

    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser: ArgumentParser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')

    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visualize, debug & record
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=24,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--cl-mode', 
        choices=['ST-Multi-Level'], 
        default=None,
        help='mode of Contrastive Learning Loss')
    parser.add_argument(
        '--cl-version',
        choices=['V0', 'V1', 'V2', "NO FN", "NO FP", "NO FN & FP"],
        default='V0',
        help='different way to calculate the cl loss')
    parser.add_argument(
        '--pred_threshold', type=float, default=0.0, help='threshold to define the confident sample')
    parser.add_argument(
        '--use_p_map', type=str2bool, default=True,
                        help='whether to add (1 - p_{ik}) to constrain the auxiliary item')
    parser.add_argument(
        '--w-multi-cl-loss', type=float, default=[0.1, 0.2, 0.5, 1], nargs='+',
                        help='weight of multi-level cl loss')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer', 
        default='SGD', 
        help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument(
        '--lr-ratio',
        type=float,
        default=0.001,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-alpha', type=float, default=0.8)
    parser.add_argument('--te-lr-ratio', type=float, default=1)




    

    return parser

def triplet_loss(class_text_features, margin=0.3):
    loss = 0.0
    num_triplets = 0

    # **确保所有张量都在同一个设备**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for label, feats in class_text_features.items():
        # 确保 feats 是 Tensor，并放到 device
        feats = torch.stack(feats).to(device)  # [N, D]
        if len(feats) < 2:
            continue

        # 随机选择锚点和正样本
        anchor_idx = torch.randint(0, len(feats), (1,))
        positive_idx = (anchor_idx + 1) % len(feats)  # 避免自对比
        anchor = feats[anchor_idx]
        positive = feats[positive_idx]

        # **确保 anchor, positive 在同一设备**
        anchor = anchor.to(device)
        positive = positive.to(device)

        # 挖掘硬负样本（距离锚点最近的负类特征）
        negative_distances = []
        for other_label in class_text_features:
            if other_label == label:
                continue
            other_feats = torch.stack(class_text_features[other_label]).to(device)  # **确保 other_feats 也在 device**
            dist = F.pairwise_distance(anchor, other_feats, p=2)  # 计算欧式距离
            negative_distances.append(dist.min())  # 取负类中最近的

        if not negative_distances:  # 无负类样本（单类别情况）
            continue

        negative = torch.stack(negative_distances).min(dim=0)[0].view(1, -1).to(device)  # **确保 negative 也在 device**

        # 计算三元组损失
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss += F.relu(pos_dist - neg_dist + margin)
        num_triplets += 1

    return loss / num_triplets if num_triplets else torch.tensor(0.0, device=device, requires_grad=True)

def class_contrastive_loss(class_text_features, temperature=0.07):
    all_features = []
    all_labels = []

    # 合并所有特征和对应的标签
    for label, features in class_text_features.items():
        # 1. 先把由 numpy.ndarrays 组成的 list 拼成一个 ndarray
        features_np = np.array(features)  # 保证所有 feature 的形状一致，如 (N, D)

        # 2. 再将 ndarray 转成 Tensor
        features_tensor = torch.from_numpy(features_np).float()

        # 3. 收集到 all_features、all_labels 中
        all_features.append(features_tensor)
        all_labels.extend([label] * len(features))

    # 拼成一个大 Tensor
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    device = all_features.device
    all_labels = all_labels.contiguous().view(-1, 1)
    mask = torch.eq(all_labels, all_labels.T).float().to(device)

    similarity_matrix = F.cosine_similarity(all_features.unsqueeze(1), 
                                            all_features.unsqueeze(0),
                                            dim=2)
    # 移除自相似
    similarity_matrix = similarity_matrix - torch.eye(len(all_features)).to(device) * 1e9

    # 正样本相似度（同一类别）
    pos_sim = torch.sum(mask * similarity_matrix, dim=1) / (mask.sum(dim=1) + 1e-8)

    # 负样本相似度（不同类别）
    neg_sim = torch.sum((1 - mask) * similarity_matrix, dim=1) / ((1 - mask).sum(dim=1) + 1e-8)

    loss = -torch.mean(torch.log(torch.sigmoid(pos_sim - neg_sim)))
    return loss

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    #answer = input('delete it? y/n:')
                    answer = 'y'
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        #input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

        




        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                for name in self.arg.model_args['head']:
                    self.model_text_dict[name] = nn.DataParallel(
                        self.model_text_dict[name],
                        device_ids=self.arg.device,
                        output_device=self.output_device)
        


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args, cl_mode=self.arg.cl_mode,
                           multi_cl_weights=self.arg.w_multi_cl_loss, cl_version=self.arg.cl_version,
                           pred_threshold=self.arg.pred_threshold, use_p_map=self.arg.use_p_map)
        #print(self.model)
        self.loss_ce = nn.CrossEntropyLoss().cuda(output_device)
        self.loss = KLLoss().cuda(output_device)

        self.model_text_dict = nn.ModuleDict()

        for name in self.arg.model_args['head']:
            model_, preprocess = clip.load(name, device)
            # model_, preprocess = clip.load('ViT-L/14', device)
            del model_.visual
            model_text = TextCLIP(model_)
            model_text = model_text.cuda(self.output_device)
            self.model_text_dict[name] = model_text

        if self.arg.weights:
            self.global_step = 0
            try:
                self.global_step = int(arg.weights[:-3].split('-')[-1])
            except:
                pass

            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

            

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                [{'params': self.model.parameters(),'lr': self.arg.base_lr},
                {'params': self.model_text_dict.parameters(), 'lr': self.arg.base_lr*self.arg.te_lr_ratio}],
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        # train model with real data
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            

            # forward
            with torch.cuda.amp.autocast():
                output, cl_loss, feature_dict, logit_scale, part_feature_list  = self.model(data, label)
                label_g = gen_label(label)
                label = label.long().cuda(self.output_device)
                loss_te_list = []
                class_text_features = defaultdict(list)
                for ind in range(num_text_aug):
                    if ind > 0:
                        text_id = np.ones(len(label),dtype=np.int8) * ind
                        texts = torch.stack([text_dict[j][i,:] for i,j in zip(label,text_id)])
                        texts = texts.to(self.output_device)

                    else:

                        texts = list()
                        for i in range(len(label)):
                            text_len = len(text_list[label[i]])
                            text_id = np.random.randint(text_len,size=1)
                            text_item = text_list[label[i]][text_id.item()]
                            texts.append(text_item)

                        # 拼接所有文本并移到目标设备
                        texts = torch.cat(texts).to(self.output_device)


                    text_embedding = self.model_text_dict[self.arg.model_args['head'][0]](texts).float()

                    for lbl, feat in zip(label.cpu(), text_embedding.detach().cpu()):
                        class_text_features[lbl.item()].append(feat)

                    if ind == 0:
                        logits_per_image, logits_per_text = create_logits(feature_dict[self.arg.model_args['head'][0]],text_embedding,logit_scale[:,0].mean())

                        ground_truth = torch.tensor(label_g,dtype=feature_dict[self.arg.model_args['head'][0]].dtype,device=device)
                    else:
                        logits_per_image, logits_per_text = create_logits(part_feature_list[ind-1],text_embedding,logit_scale[:,ind].mean())

                        ground_truth = torch.tensor(label_g,dtype=part_feature_list[ind-1].dtype,device=device)


                    loss_imgs = self.loss(logits_per_image,ground_truth)
                    loss_texts = self.loss(logits_per_text,ground_truth)

                    loss_te_list.append((loss_imgs + loss_texts) / 2)

                loss_ce = self.loss_ce(output, label)
                #loss_text_constraint = class_contrastive_loss(class_text_features)
                loss_text_constraint = triplet_loss(class_text_features, margin=0.3)  # 新增边际参数

                if self.arg.cl_mode is not None:
                    self.train_writer.add_scalar('cl_loss', cl_loss.mean().data.item(), self.global_step)
                    #loss = loss_ce + self.arg.loss_alpha * sum(loss_te_list) / len(loss_te_list) + 0.1 * loss_text_constraint + 0.2 * cl_loss.mean() #原loss
                    loss = loss_ce + self.arg.loss_alpha * sum(loss_te_list) / len(loss_te_list) + 0.2 * loss_text_constraint + 0.3 * cl_loss.mean()  #三元组loss
                    

                else:
                    loss = loss_ce + self.arg.loss_alpha * sum(loss_te_list) / len(loss_te_list) + 0.1 * loss_text_constraint

            # backward
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(self.optimizer)
            scaler.update()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value),
                                                                                np.mean(acc_value) * 100))
        #self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        self.print_log('\tLearning Rate: {:.6f}'.format(self.lr))



        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights,
                       self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)



            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    # print(data.size())
                    b, _, _, _, _ = data.size()
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output, _, _, _, _ = self.model(data,label)
                    loss = self.loss_ce(output, label)

                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            self.print_log(f'Epoch number: {self.best_acc_epoch}')

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            wrong_analyze(wf, rf)
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            wrong_analyze(wf, rf)
            self.print_log('Done.\n')


if __name__ == '__main__':
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    parser = get_parser()

    # load arg from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    try:
        processor.start()
    except:
        processor.print_log(str(traceback.format_exc()), False)