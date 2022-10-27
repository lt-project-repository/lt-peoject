"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from logger import Logger
import time
import numpy as np
import warnings
import pdb
# import higher
import json

import matplotlib.pyplot as plt


# from sklearn import manifold
# from tsnecuda import TSNE
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class model():

    def __init__(self, config, data, test=False, meta_sample=False, learner=None):

        self.meta_sample = meta_sample

        # init meta learner and meta set
        if self.meta_sample:
            assert learner is not None
            self.learner = learner
            self.meta_data = iter(data['meta'])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        # self.gpu_id = self.config['gpu_id']
        # pdb.set_trace()
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.num_classes = self.training_opt['num_classes']
        self.batch_size = self.training_opt['batch_size']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config['shuffle'] if 'shuffle' in config else False

        # Compute epochs from iterations
        if self.training_opt.get('num_iterations', False):
            self.training_opt['num_epochs'] = math.ceil(self.training_opt['num_iterations'] / len(self.data['train']))
        if self.config.get('warmup_iterations', False):
            self.config['warmup_epochs'] = math.ceil(self.config['warmup_iterations'] / len(self.data['train']))

        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])

        # Initialize model
        self.init_models()
        # pdb.set_trace()

        # Load pre-trained model parameters
        if 'model_dir' in self.config and self.config['model_dir'] is not None:
            self.load_model(self.config['model_dir'])

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps 
            # for each epoch based on actual number of training data instead of 
            # oversampled data number 
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            # pdb.set_trace()
            self.epoch_steps = int(self.training_data_num \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()
            # pdb.set_trace()
            if self.memory['init_centroids']:
                self.criterions['FeatureLoss'].centroids.data = \
                    self.centroids_cal(self.data['train_plain'])

            # Set up log file
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(self.config)
        else:
            if 'KNNClassifier' in self.config['networks']['classifier']['def_file']:
                self.load_model()
                if not self.networks['classifier'].initialized:
                    cfeats = self.get_knncentroids()
                    print('===> Saving features to %s' %
                          os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'))
                    with open(os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'), 'wb') as f:
                        pickle.dump(cfeats, f)
                    self.networks['classifier'].update(cfeats)
            self.log_file = None

        freq_path = self.training_opt['freq_path']
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        self.sample_per_class = torch.tensor(freq)

        self.init_stat()

        self.feature_norm_list_per_class = [[] for i in range(self.num_classes)]

        # self.feature_list_per_class = [[] for i in range(self.num_classes) ]

        # self.neg_ratio_09 = []
        # self.neg_ratio_18 = []
        # self.neg_ratio_19 = []
        # self.neg_ratio_37 = []

    def init_stat(self):
        ### 我们需要统计哪些内容呢？
        ### 1. 每个类别的分类器，来自于不同类别样本的梯度统计(pos neg)；
        ### 2. 每个类别的所有样本，来自于不同类别分类器的梯度的统计(pos neg)。
        ### 上面提及的梯度，都是logit.grad层面的梯度，并不是分类器或者样本feature的直接梯度

        ## 声明变量，用来计算每个类别的分类器的梯度统计

        # 统计每个类别neg pos grad 来自于不同类别的分类
        # size = (C, C)。dim_0 代表着每个类别的分类器，dim_1代表着该分类器来自于不同类别的样本的梯度
        self.accum_cls_fine_grad = torch.zeros((self.num_classes, self.num_classes), device=self.device)
        self.accum_cls_pos_fine_grad = torch.zeros((self.num_classes, self.num_classes), device=self.device)
        self.accum_cls_neg_fine_grad = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        # size = (C), 记录每个类别的分类器，在logit层面受到的pos和neg 梯度，累计整个训练过程
        self.accum_cls_pos_grad = torch.zeros(self.num_classes, device=self.device)
        self.accum_cls_neg_grad = torch.zeros(self.num_classes, device=self.device)
        # 统计每个类别的分类器，logit层面pos grad 梯度的比值
        self.cls_pos_neg_ratio = torch.ones(self.num_classes, device=self.device)

        ## 声明变量，用来统计每个类别的所有样本累积在一起受到的梯度

        # size = (C, C)。dim_0 代表着每个类别的样本，dim_1代表着该分类器来自于不同类别的样本的梯度
        self.accum_sample_pos_fine_grad = torch.zeros((self.num_classes, self.num_classes), device=self.device)
        self.accum_sample_neg_fine_grad = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        # size = (C), 统计每个类别的样本feature，在logit层面受到的pos和neg 梯度，累计整个训练过程
        self.accum_sample_pos_grad = torch.zeros(self.num_classes, device=self.device)
        self.accum_sample_neg_grad = torch.zeros(self.num_classes, device=self.device)
        # 统计每个类别的样本，logit层面pos grad 梯度的比值
        self.sample_pos_neg_ratio = torch.ones(self.num_classes, device=self.device)

        self.past_weight = copy.deepcopy(self.networks['classifier'].module.fc.weight)

    def init_stat_epoch(self):

        # TODO

        # size = (C), 记录每个类别的样本feature，在logit层面接受到的pos和neg 梯度，累计整个训练过程
        # self.accum_sample_pos_grad = torch.zeros(self.num_classes, device=self.device)
        # self.accum_sample_neg_grad = torch.zeros(self.num_classes, device=self.device)
        # # 统计每个类别的feature，logit层面pos grad 梯度的比值
        # self.sample_pos_neg_ratio = torch.ones(self.num_classes,device=self.device)

        # size = (C), 记录每个类别的样本feature，在logit层面接受到的pos和neg 梯度，累计整个epoch过程
        self.accum_epoch_sample_pos_grad = torch.zeros(self.num_classes, device=self.device)
        self.accum_epoch_sample_neg_grad = torch.zeros(self.num_classes, device=self.device)
        # 统计每个类别的feature，logit层面pos grad 梯度的比值,仅仅每个epoch
        self.sample_pos_neg_ratio_epoch = torch.ones(self.num_classes, device=self.device)

        # 统计每个类别neg pos grad 来自于不同类别的分类，pos grad由于只能来自于pos，所以和self.accum_cls_pos_grad是一样的，仅仅epoch
        self.accum_pos_logit_grad_epoch = torch.zeros(self.num_classes, device=self.device)
        self.accum_neg_logit_grad_epoch = torch.zeros((self.num_classes, self.num_classes), device=self.device)

    def init_models(self, optimizer=True):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        if self.meta_sample:
            # init meta optimizer
            self.optimizer_meta = torch.optim.Adam(self.learner.parameters(),
                                                   lr=self.training_opt['sampler'].get('lr', 0.01))

        print("Using", torch.cuda.device_count(), "GPUs.")

        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            # model_args = list(val['params'].values())
            # model_args.append(self.test_mode)
            model_args = val['params']
            model_args.update({'test': self.test_mode})

            self.networks[key] = source_import(def_file).create_model(**model_args)
            if 'KNNClassifier' in type(self.networks[key]).__name__:
                # Put the KNN classifier on one single GPU
                self.networks[key] = self.networks[key].cuda()
            else:
                self.networks[key] = nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'selfatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False
                    # print('  | ', param_name, param.requires_grad)

            if self.meta_sample and key != 'classifier':
                # avoid adding classifier parameters to the optimizer,
                # otherwise error will be raised when computing higher gradients
                continue

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                 'lr': optim_params['lr'],
                                                 'momentum': optim_params['momentum'],
                                                 'weight_decay': optim_params['weight_decay']})
            # if key == 'classifier':
            #     self.model_fc_optim_params = {'params': self.networks[key].parameters(),
            #                                     'lr': optim_params['lr'],
            #                                     'momentum': optim_params['momentum'],
            #                                     'weight_decay': optim_params['weight_decay']}
            # if key == 'feat_model':
            #     self.model_cnn_optim_params = {'params': self.networks[key].parameters(),
            #                                     'lr': optim_params['lr'],
            #                                     'momentum': optim_params['momentum'],
            #                                     'weight_decay': optim_params['weight_decay']}
            # pdb.set_trace()

    def init_criterions(self):
        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = list(val['loss_params'].values())
            # pdb.set_trace()

            self.criterions[key] = source_import(def_file).create_loss(*loss_args).cuda()
            self.criterion_weights[key] = val['weight']

            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                 'lr': optim_params['lr'],
                                 'momentum': optim_params['momentum'],
                                 'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        if self.config['coslr']:
            print("===> Using coslr eta_min={}".format(self.config['endlr']))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config['endlr'])
        elif self.config['coslrwarmup']:
            print("===> Using coslrwarmup eta_min={}, warmup_epochs={}".format(
                self.config['endlr'], self.config['warmup_epochs']))
            scheduler = CosineAnnealingLRWarmup(
                optimizer=optimizer,
                T_max=self.training_opt['num_epochs'],
                eta_min=self.config['endlr'],
                warmup_epochs=self.config['warmup_epochs'],
                base_lr=self.config['base_lr'],
                warmup_lr=self.config['warmup_lr']
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.scheduler_params['step_size'],
                                                  gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def batch_forward(self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
        """
        This is a general single batch running function.

        在这个上面的改动就是 self.logits.retain_grad()
        """

        # Calculate Features
        self.features, self.feature_maps = self.networks['feat_model'](inputs)

        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to 
            if phase != 'test':
                if centroids and 'FeatureLoss' in self.criterions.keys():
                    self.centroids = self.criterions['FeatureLoss'].centroids.data
                    torch.cat([self.centroids] * self.num_gpus)
                else:
                    self.centroids = None

            if self.centroids is not None:
                centroids_ = torch.cat([self.centroids] * self.num_gpus)
            else:
                centroids_ = self.centroids

            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.networks['classifier'](self.features,
                                                                                  self.accum_sample_pos_grad + 1e-3,
                                                                                  centroids_)

            # pdb.set_trace()

            # 是non-leaves node 可求grad
            if phase == 'train':
                self.logits.retain_grad()

            # self.logit = self.logit_ * weight

    def collect_grad(self, logits_grad, labels):

        ### 将logit_grad 的pos neg grad 拆分开
        # B * C
        logits_grad = torch.abs(logits_grad)

        # target = self.logits.new_zeros(self.batch_size, self.num_classes)
        # target[torch.arange(self.batch_size), labels] = 1 #TODO

        # B * C
        # pos_logit_grad = logits_grad * target
        # neg_logit_grad = logits_grad * (1 - target)

        # pdb.set_trace()

        for idx in range(self.batch_size):
            self.accum_cls_fine_grad[:, labels[idx]] += logits_grad[idx]
            self.accum_neg_logit_grad_epoch[labels[idx], :] += logits_grad[idx]

        eye_mask = torch.eye(self.num_classes, device=self.device)

        self.accum_cls_pos_fine_grad += self.accum_cls_fine_grad * eye_mask
        self.accum_cls_neg_fine_grad += self.accum_cls_fine_grad * (1 - eye_mask)

        # pdb.set_trace()

        self.accum_cls_pos_grad = self.accum_cls_pos_fine_grad.sum(dim=1)
        self.accum_cls_neg_grad = self.accum_cls_neg_fine_grad.sum(dim=1)

        self.accum_sample_pos_grad = self.accum_cls_pos_fine_grad.sum(dim=0)
        self.accum_sample_neg_grad = self.accum_cls_neg_fine_grad.sum(dim=0)

        self.cls_pos_neg_ratio = (self.accum_cls_pos_grad + 1e-10) / (self.accum_cls_neg_grad + 1e-10)

        # #######
        # # 统计每个类别的样本来自于不同分类器的梯度
        # self.accum_sample_pos_fine_grad
        # self.accum_sample_cls_fine, size = (C,C), dim_0 是每个类别的sample， dim_1是每个类别的分类器
        # for idx in range(self.batch_size):
        #     self.accum_sample_cls_fine[labels[idx],:] += logits_grad[idx]

    def get_weight_for_forward_logit(self):

        # print(self.pos_neg_ratio)

        pos_weights = 1. / self.pos_neg_ratio
        neg_weights = torch.ones(self.num_classes, device=self.device)

        pos_weights = pos_weights.view(1, -1).expand(self.batch_size, self.num_classes)
        neg_weights = neg_weights.view(1, -1).expand(self.batch_size, self.num_classes)
        return pos_weights, neg_weights

    def get_weighted_logit_grad(self, logits_grad, labels):
        # 在这个方法里，可以尝试不同的直接给梯度的加权方法
        # self.sample_per_class
        # B * C

        def neg_pos_bal(logits_grad, labels):
            # 每个class的分类器，接受到的pos neg grad是均衡的
            weighted_logits_grad = torch.ones((self.batch_size, self.num_classes), device=self.device)

            for idx in range(self.num_classes):
                # self.cls_pos_neg_ratio[idx]
                # labels == idx
                logits_grad_cls = logits_grad[:, idx]
                weight_cls = torch.ones(self.batch_size, device=self.device)
                weight_cls[labels != idx] = self.cls_pos_neg_ratio[idx]  # TODO
                # weight_cls[labels == idx]
                # weight[:,idx] = weight_cls
                # if self.epoch == 3 and idx in [1,50,99]:
                #     pdb.set_trace()
                weighted_logits_grad[:, idx] = torch.mul(weight_cls, logits_grad_cls)
            # return logits_grad
            return weighted_logits_grad

        def neg_pos_bal_norm(logits_grad, labels):
            # 每个class的分类器，接受到的pos neg grad是均衡的，同时，weighted logit grad在dim C保持norm不变
            weighted_logits_grad = torch.ones((self.batch_size, self.num_classes), device=self.device)

            for idx in range(self.num_classes):
                # self.cls_pos_neg_ratio[idx]
                # labels == idx
                logits_grad_cls = logits_grad[:, idx]
                weight_cls = torch.ones(self.batch_size, device=self.device)
                weight_cls[labels != idx] = self.cls_pos_neg_ratio[idx]
                # weight_cls[labels == idx]
                # weight[:,idx] = weight_cls
                # if self.epoch == 3 and idx in [1,50,99]:
                #     pdb.set_trace()
                # TODO 有问题。如果整个batch全是neg sample，那么norm一致，就会取消加权效果
                # logits_grad[:,idx] = torch.mul(weight_cls, logits_grad_cls) / torch.mul(weight_cls, logits_grad_cls).norm() * logits_grad_cls.norm()
                weighted_logits_grad[:, idx] = torch.mul(weight_cls, logits_grad_cls) / torch.mul(weight_cls,
                                                                                                  logits_grad_cls).norm() * logits_grad_cls.norm()
            # return logits_grad
            return weighted_logits_grad

        # logits_grad = copy.deepcopy(logits_grad)
        logits_grad_ori = logits_grad  # TODO
        logits_grad = torch.abs(logits_grad)

        if self.training_opt['backward_weight_logits'] == 'none':
            # 普通的训练方法
            return logits_grad_ori

        elif self.training_opt['backward_weight_logits'] == 'neg_pos_bal':

            ####  这个策略是，平衡每个fc weight grad的neg grad 和 pos grad，而不是每个样本的neg grad和neg grad平衡
            ####  直接输出weighted logit.grad
            return neg_pos_bal(logits_grad_ori, labels)

        elif self.training_opt['backward_weight_logits'] == 'neg_pos_bal_norm':

            return neg_pos_bal_norm(logits_grad_ori, labels)


        elif self.training_opt['backward_weight_logits'] == 'balanced_freq':
            # 在logits的梯度上，
            logits_grad = torch.abs(logits_grad)

            target = self.logits.new_zeros(self.batch_size, self.num_classes)
            target[torch.arange(self.batch_size), labels] = 1  # TODO
            neg_target = 1 - target
            spc = self.sample_per_class.type_as(target)
            neg_weight = neg_target * spc.unsqueeze(0).expand(self.batch_size, -1)
            for batch_idx in range(self.batch_size):
                neg_weight[batch_idx] = neg_weight[batch_idx] / spc[labels[batch_idx]]

            neg_grad = (logits_grad * neg_weight).sum(dim=1)
            pos_grad = (logits_grad * target).sum(dim=1)
            # pos_weight = target

            neg_grad_sum = neg_grad.unsqueeze(1).expand(-1, self.num_classes)
            pos_grad_sum = pos_grad.unsqueeze(1).expand(-1, self.num_classes)
            neg_pos_ratio = neg_grad_sum / pos_grad_sum
            pos_weight = target * neg_pos_ratio
            weight = pos_weight + neg_weight

            # logits_g_w = logits_grad * weight
            # logits_g_w_neg = logits_g_w * neg_target
            # logits_g_w_pos = logits_g_w * target

            # pdb.set_trace()

            '''
            weight = torch.ones((self.batch_size, self.num_classes),device=self.device)
            spc = self.sample_per_class.type_as(weight)
            weight = weight * spc.unsqueeze(0).expand(weight.shape[0], -1)

            for batch_idx in range(self.batch_size):
                weight[batch_idx] = weight[batch_idx] / spc[labels[batch_idx]] 
            # pdb.set_trace()
            '''
            return weight

    def batch_backward(self, labels):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        # self.logits.retain_grad()
        self.loss.backward()

        # if self.epoch in [100]:

        #     pdb.set_trace()

        weighted_logits_grad = self.get_weighted_logit_grad(self.logits_for_backward.grad, labels)
        # sum_of_logits_grad 
        # weight_for_logits_grad = torch.log(weight_for_logits_grad+1)
        # logits_grad_with_weight = self.logits_for_backward.grad * weight_for_logits_grad
        # logits_grad_with_weight = logits_grad_with_weight / logits_grad_with_weight.norm() * self.logits_for_backward.grad.norm()

        # TODO 应该是收集加权前的，还是加权后的

        self.collect_grad(weighted_logits_grad, labels)

        # print(self.logits_for_backward.grad.norm() / logits_grad_with_weight.norm())

        #  logits 的grad
        # if self.epoch == 2:
        #     pdb.set_trace()
        self.logits.backward(weighted_logits_grad)
        # self.logits_for_backward.grad 

        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):
        # 这个batch loss里面，使用logits_for_backward提到原始logits进行loss的计算。
        self.loss = 0

        self.logits_for_backward = self.logits.detach()
        self.logits_for_backward.requires_grad = True

        # pdb.set_trace()

        if self.training_opt['weight_logits'] == True:
            pos_weight, neg_weight = self.get_weight_for_forward_logit()
            target = self.logits.new_zeros((self.batch_size, self.num_classes), device=self.device)
            target[torch.arange(self.batch_size), labels] = 1
            weight_for_logits = pos_weight * target + neg_weight * (1 - target)
        else:
            weight_for_logits = torch.ones((self.batch_size, self.num_classes), device=self.device)

        # pdb.set_trace()

        # First, apply performance loss
        if 'PerformanceLoss' in self.criterions.keys():
            #### 权重加在exp上
            self.loss_perf = self.criterions['PerformanceLoss'](self.logits_for_backward, labels)

            self.loss_perf *= self.criterion_weights['PerformanceLoss']
            self.loss += self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat

    '''
    def batch_loss(self, labels):
        self.loss = 0

        
        if self.training_opt['weight_logits'] == True:
            pos_weight, neg_weight = self.get_weight_for_forward_logit()
            target = self.logits.new_zeros((self.batch_size, self.num_classes),device=self.device)
            target[torch.arange(self.batch_size), labels] = 1
            weight_for_logits = pos_weight * target + neg_weight * (1 - target)
        else:
            weight_for_logits = torch.ones((self.batch_size, self.num_classes), device=self.device)

        # pdb.set_trace()

        # First, apply performance loss
        if 'PerformanceLoss' in self.criterions.keys():

            #### 权重加在exp上
            self.loss_perf = self.criterions['PerformanceLoss'](self.logits + torch.log(weight_for_logits), labels)

            self.loss_perf *=  self.criterion_weights['PerformanceLoss']
            self.loss += self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat
    '''

    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def meta_forward(self, inputs, labels, verbose=False):
        # take a meta step in the inner loop
        '''
        self.learner.train()
        self.model_optimizer.zero_grad()
        self.optimizer_meta.zero_grad()
        with higher.innerloop_ctx(self.networks['classifier'], self.model_optimizer) as (fmodel, diffopt):
            # obtain the surrogate model
            features, _ = self.networks['feat_model'](inputs)
            train_outputs, _ = fmodel(features.detach())
            loss = self.criterions['PerformanceLoss'](train_outputs, labels, reduction='none')
            loss = self.learner.forward_loss(loss)
            diffopt.step(loss)

            # use the surrogate model to update sample rate
            val_inputs, val_targets, _ = next(self.meta_data)
            val_inputs = val_inputs.cuda()
            val_targets = val_targets.cuda()
            features, _ = self.networks['feat_model'](val_inputs)
            val_outputs, _ = fmodel(features.detach())
            val_loss = F.cross_entropy(val_outputs, val_targets, reduction='mean')
            val_loss.backward()
            self.optimizer_meta.step()

        self.learner.eval()

        if verbose:
            # log the sample rates
            num_classes = self.learner.num_classes
            prob = self.learner.fc[0].weight.sigmoid().squeeze(0)
            print_str = ['Unnormalized Sample Prob:']
            interval = 1 if num_classes < 10 else num_classes // 10
            for i in range(0, num_classes, interval):
                print_str.append('class{}={:.3f},'.format(i, prob[i].item()))
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            print_str.append('\nMax Mem: {:.0f}M'.format(max_mem_mb))
            print_write(print_str, self.log_file)
        '''
        pass

    def collect_feature_norm(self, features, labels):
        # self.feature_norm = torch.zeros(self,num_classes,device=self.device)

        for idx in range(len(labels)):
            self.feature_norm_list_per_class[labels[idx]].append(features[idx].norm(dim=0))
            # pdb.set_trace()

    def collect_feature(self, features, labels, phase):
        if phase == 'train':
            for idx in range(len(labels)):
                self.feature_list.append(features[idx])
                self.feature_list_label.append(labels[idx].item())
        elif phase == 'val':
            for idx in range(len(labels)):
                self.feature_list_eval.append(features[idx])
                self.feature_list_label_eval.append(labels[idx].item())

    def collect_cossim_fc_weight_time(self):
        weight = self.networks['classifier'].module.fc.weight
        weight_cos_sim = []
        # if self.epoch < 15:
        #     pdb.set_trace()
        for idx in range(self.num_classes):
            weight_cos_sim.append(F.cosine_similarity(weight[idx, :], self.past_weight[idx, :], dim=0).item())
        # self.past_weight = copy.deepcopy(weight)
        return weight_cos_sim

    def collect_feature_fc_weight(self):
        weight = self.networks['classifier'].module.fc.weight
        for idx in range(self.num_classes):
            self.feature_list.append(weight[idx])
            self.feature_list_label.append(self.num_classes + idx)

            self.feature_list_eval.append(weight[idx])
            self.feature_list_label_eval.append(self.num_classes + idx)

    def init_tsne_feature_list(self):
        self.feature_list = []
        self.feature_list_label = []

        self.feature_list_eval = []
        self.feature_list_label_eval = []

    # def get_feature_inter_intra_class_dist(self):

    def cal_feature_norm_per_class(self):
        self.norm_per_class = []

        for lst in self.feature_norm_list_per_class:
            self.norm_per_class.append(sum(lst) / (len(lst) + 1e-10))

        self.norm_per_class = torch.tensor(self.norm_per_class)
        # pdb.set_trace()

    # 所有在每个epoch训练和eval后的处理，都可以放在这个方法中
    def callback_after_epoch(self, epoch):

        # if epoch % 10 == 1:
        if True:
            # self.accum_cls_pos_grad = self.accum_cls_pos_fine_grad.sum(dim=1)
            # self.accum_cls_neg_grad = self.accum_cls_neg_fine_grad.sum(dim=1)

            # self.accum_sample_pos_grad = self.accum_cls_pos_fine_grad.sum(dim=0)
            # self.accum_sample_neg_grad

            print_str = 'accum_cls_pos_grad: ' + str(self.accum_cls_pos_grad.cpu().numpy())
            print_write(print_str, self.log_file)

            print_str = 'accum_cls_neg_grad: ' + str(self.accum_cls_neg_grad.cpu().numpy())
            print_write(print_str, self.log_file)

            print_str = 'accum_sample_pos_grad: ' + str(self.accum_sample_pos_grad.cpu().numpy())
            print_write(print_str, self.log_file)

            print_str = 'accum_sample_neg_grad: ' + str(self.accum_sample_neg_grad.cpu().numpy())
            print_write(print_str, self.log_file)

            print_str = 'accum_neg_logit_grad_epoch' + str(self.accum_neg_logit_grad_epoch)
            print_write(print_str, self.log_file)

            # print_str = 'cls_pos_neg_ratio: ' + str(self.cls_pos_neg_ratio.cpu().numpy())
            # print_write(print_str, self.log_file)

            # print_str = 'cos sim of fc weight: \n' + str(self.collect_cossim_fc_weight_time())
            # print_write(print_str, self.log_file)

        pass

    def train(self):
        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        print_write(['Do shuffle??? --- ', self.do_shuffle], self.log_file)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt['num_epochs']
        # pdb.set_trace()

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            self.epoch = epoch

            # if self.epoch in [10,20,50,100,500]:
            #     pdb.set_trace()

            self.init_stat_epoch()

            for model in self.networks.values():
                model.train()

            torch.cuda.empty_cache()

            # pdb.set_trace()

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train() 
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # Iterate over dataset
            total_preds = []
            total_labels = []

            self.init_tsne_feature_list()

            self.feature_norm_list_per_class = [[] for i in range(self.num_classes)]
            self.feature_list_per_class = [[] for i in range(self.num_classes)]

            self.accum_pos_grad = 0
            self.accum_neg_grad = 0

            self.pos_neg_ratio = torch.ones(self.num_classes, device=self.device)

            for step, (inputs, labels, indexes) in enumerate(self.data['train']):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                    if self.meta_sample:
                        # do inner loop
                        self.meta_forward(inputs, labels, verbose=step % self.training_opt['display_step'] == 0)

                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels,
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    self.collect_feature_norm(self.features, labels)
                    self.collect_feature(self.features, labels, phase='train')
                    self.batch_loss(labels)
                    self.batch_backward(labels)

                    # Tracking predictions
                    _, preds = torch.max(self.logits, 1)
                    total_preds.append(torch2numpy(preds))
                    total_labels.append(torch2numpy(labels))

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:
                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item() \
                            if 'PerformanceLoss' in self.criterions else None
                        minibatch_loss_total = self.loss.item()
                        minibatch_acc = mic_acc_cal(preds, labels)

                        print_str = ['Epoch: [%d/%d]'
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d'
                                     % (step),
                                     'Minibatch_loss_feature: %.3f'
                                     % (minibatch_loss_feat) if minibatch_loss_feat else '',
                                     'Minibatch_loss_performance: %.3f'
                                     % (minibatch_loss_perf) if minibatch_loss_perf else '',
                                     'Minibatch_accuracy_micro: %.3f'
                                     % (minibatch_acc)]
                        print_write(print_str, self.log_file)

                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            'CE': minibatch_loss_perf,
                            'feat': minibatch_loss_feat
                        }

                        self.logger.log_loss(loss_info)

                # Update priority weights if using PrioritizedSampler
                # if self.training_opt['sampler'] and \
                #    self.training_opt['sampler']['type'] == 'PrioritizedSampler':
                if hasattr(self.data['train'].sampler, 'update_weights'):
                    if hasattr(self.data['train'].sampler, 'ptype'):
                        ptype = self.data['train'].sampler.ptype
                    else:
                        ptype = 'score'
                    ws = get_priority(ptype, self.logits.detach(), labels)
                    # ws = logits2score(self.logits.detach(), labels)
                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt['sampler']['type'] == 'ClassPrioritySampler':
                        inlist.append(labels.cpu().numpy())
                    self.data['train'].sampler.update_weights(*inlist)
                    # self.data['train'].sampler.update_weights(indexes.cpu().numpy(), ws)

            if hasattr(self.data['train'].sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            if hasattr(self.data['train'].sampler, 'reset_weights'):
                self.data['train'].sampler.reset_weights(epoch)

            self.callback_after_epoch(epoch)

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)

            # 每个epoch 的train和val结束后，收集 classifier的 weight
            # self.collect_feature_fc_weight()
            # self.plot_tsne(epoch)

            # Reset class weights for sampling if pri_mode is valid
            if hasattr(self.data['train'].sampler, 'reset_priority'):
                ws = get_priority(self.data['train'].sampler.ptype,
                                  self.total_logits.detach(),
                                  self.total_labels)
                self.data['train'].sampler.reset_priority(ws, self.total_labels.cpu().numpy())

            # Log results
            self.logger.log_acc(rsls)

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                best_centroids = self.centroids
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

            feat_saved_dir = self.training_opt['log_dir'] + '/saved_feats'
            if not os.path.exists(feat_saved_dir):
                os.makedirs(feat_saved_dir)
            feat_saved_path = os.path.join(feat_saved_dir,
                                           str(self.epoch) + '_feats.pth')
            # TODO
            # torch.save({'epoch':self.epoch,
            # 'feats':self.feats_all,
            # 'labels':self.labels_all}, feat_saved_path)

            print('===> Saving checkpoint')
            self.save_latest(epoch)

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)

        # Test on the test set
        self.reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        print('Done')

    def eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)

        # Calculate normal prediction accuracy
        rsl = {'train_all': 0., 'train_many': 0., 'train_median': 0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = \
                list(map(np.concatenate, [mixup_preds * 2, mixup_labels1 + mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1 - mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = ['\n Training acc Top1: %.3f \n' % (rsl['train_all']),
                     'Many_top1: %.3f' % (rsl['train_many']),
                     'Median_top1: %.3f' % (rsl['train_median']),
                     'Low_top1: %.3f' % (rsl['train_low']),
                     '\n']
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase='val', openset=False, save_feat=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f'
                  % self.training_opt['open_threshold'])

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        get_feat_only = save_feat
        self.feats_all, self.labels_all, self.idxs_all, self.logits_all = [], [], [], []
        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels,
                                   centroids=self.memory['centroids'],
                                   phase=phase)

                # 收集每个epoch中，val样本的feature
                self.collect_feature(self.features, labels, phase=phase)

                if not get_feat_only:
                    self.total_logits = torch.cat((self.total_logits, self.logits))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths))

                # pdb.set_trace()
                # if get_feat_only:
                self.logits_all.append(self.logits)
                self.feats_all.append(self.features)
                self.labels_all.append(labels)
                self.idxs_all.append(paths)

        # self.cal_intra_inter_cos_sim(feats_all, labels_all)

        if get_feat_only:
            typ = 'feat'
            if phase == 'train_plain':
                name = 'train{}_all.pkl'.format(typ)
            elif phase == 'test':
                name = 'test{}_all.pkl'.format(typ)
            elif phase == 'val':
                name = 'val{}_all.pkl'.format(typ)

            fname = os.path.join(self.training_opt['log_dir'], name)
            print('===> Saving feats to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                    'feats': np.concatenate(self.feats_all),
                    'labels': np.concatenate(self.labels_all),
                    'idxs': np.concatenate(self.idxs_all),
                },
                    f, protocol=4)
            return

            # probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)
        if self.training_opt['norm_logits'] == True:
            probs_logits = F.softmax(self.total_logits.detach(), dim=1)
            class_probs_norm = probs_logits.norm(dim=0, p=1)
            class_probs_norm = class_probs_norm.unsqueeze(0).expand(self.total_logits.shape[0], -1)
            new_logits = probs_logits / torch.pow(class_probs_norm, self.training_opt['pow_for_logits'])
            # pdb.set_trace()
        else:
            new_logits = self.total_logits.detach()
            # pdb.set_trace()

        probs, preds = F.softmax(new_logits, dim=1).max(dim=1)

        if self.epoch % 20 == 1:
            print_str = 'sum of logits: \n' + str(F.softmax(self.total_logits.detach(), dim=1).sum(dim=0))
            print_write(print_str, self.log_file)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                           self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1 = mic_acc_cal(preds[self.total_labels != -1],
                                             self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds[self.total_labels != -1],
                                 self.total_labels[self.total_labels != -1],
                                 self.data['train'],
                                 acc_per_cls=True)
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s'
                     % phase,
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f'
                     % self.eval_acc_mic_top1,
                     '\n',
                     'Averaged F-measure: %.3f'
                     % self.eval_f_measure,
                     '\n',
                     'Many_shot_accuracy_top1: %.3f'
                     % self.many_acc_top1,
                     'Median_shot_accuracy_top1: %.3f'
                     % self.median_acc_top1,
                     'Low_shot_accuracy_top1: %.3f'
                     % self.low_acc_top1,
                     '\n']

        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}

        if phase == 'val':
            print_write(print_str, self.log_file)
        else:
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)

        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.cls_accs, f)
        return rsl

    def centroids_cal(self, data, save_all=False):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all, idxs_all = [], [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(data):
                inputs, labels = inputs.cuda(), labels.cuda()

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]
                # Save features if requried
                if save_all:
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(idxs.numpy())

        if save_all:
            fname = os.path.join(self.training_opt['log_dir'], 'feats_all.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all)},
                            f)
        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def get_knncentroids(self):
        datakey = 'train_plain'
        assert datakey in self.data

        print('===> Calculating KNN centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all = [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(self.data[datakey]):
                inputs, labels = inputs.cuda(), labels.cuda()

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)

                feats_all.append(self.features.cpu().numpy())
                labels_all.append(labels.cpu().numpy())

        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all)

        featmean = feats.mean(axis=0)

        def get_centroids(feats_, labels_):
            centroids = []
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_ == i], axis=0))
            return np.stack(centroids)

        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)

        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,
                'cl2ncs': cl2n_centers}

    def reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_model(self, model_dir=None):
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')

        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))

        # pdb.set_trace()  
        checkpoint = torch.load(model_dir)

        # model_state = checkpoint['state_dict_best']
        weights = checkpoint['state_dict_model']
        # weights_classifier = checkpoint['state_dict_classifier']

        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None

        fenduan = self.training_opt.get('fenduan', False)

        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None

        for key, model in self.networks.items():
            if key == 'classifier':
                continue
            # weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # weights = {k: checkpoint[k] for k in checkpoint if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)

    def save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights
        }

        model_dir = os.path.join(self.training_opt['log_dir'],
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)

    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):

        model_states = {'epoch': epoch,
                        'best_epoch': best_epoch,
                        'state_dict_best': best_model_weights,
                        'best_acc': best_acc,
                        'centroids': centroids}

        model_dir = os.path.join(self.training_opt['log_dir'],
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)

    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'],
                                'logits_%s' % ('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename,
                 logits=self.total_logits.detach().cpu().numpy(),
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)
