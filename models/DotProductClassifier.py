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
import torch
import torch.nn as nn
from utils import *
from os import path
import json


class DotProduct_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048, freq_path=None, margin_cls=False, margin_mode='freq', lamda_1=1.0, lamda_2=1.0, *args):
        super(DotProduct_Classifier, self).__init__()
        self.freq_path = freq_path
        self.margin_cls = margin_cls
        self.margin_mode = margin_mode
        if self.freq_path != None and self.freq_path != "none":
            # pdb.set_trace()
            with open(self.freq_path, 'r') as fd:
                # pdb.set_trace()
                freq = json.load(fd)
            freq = torch.tensor(freq).float()
            self.sample_per_class = freq
            self.sample_per_class = self.sample_per_class / torch.sum(self.sample_per_class)
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.fc = nn.Linear(feat_dim, num_classes)
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2

    def forward(self, x, pos_grad, accum_sample_neg_grad, labels, *args):
        # if self.margin_mode == 'freq':
        #     margin = self.sample_per_class
        if self.margin_mode == 'pos_grad':
            # pdb.set_trace()
            margin = pos_grad

        if self.training and self.margin_cls:
            margin = margin.type_as(x)
            margin = margin.expand(x.shape[0], -1)
            # print(pos_grad.shape)
            # print(margin.shape)
            # print(self.fc(x).shape)
            x = self.fc(x) + self.lamda_1 * torch.log(margin)
            
            accum_sample_neg_grad_by_labels = accum_sample_neg_grad[labels]
            x = x - self.lamda_2 * torch.log(accum_sample_neg_grad_by_labels.unsqueeze(1) + 1e-3)
        else:
            x = self.fc(x)
        return x, None


def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False,
                 freq_path=None, margin_cls=False, margin_mode='freq', lamda_1=1.0, lamda_2=1.0, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim, freq_path, margin_cls, margin_mode, lamda_1, lamda_2)

    if not test:
        if stage1_weights:
            assert (dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            if log_dir is not None:
                subdir = log_dir.strip('/').split('/')[-1]
                subdir = subdir.replace('stage2', 'stage1')
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
                # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading classifier weights from %s' % weight_dir)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
