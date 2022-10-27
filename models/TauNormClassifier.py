"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from utils import *
from os import path


class DotProduct_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048, freq_path=None, margin_cls=False, margin_mode='freq', *args):
        super(DotProduct_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.freq_path = freq_path
        self.margin_cls = margin_cls
        self.margin_mode = margin_mode
        self.fc = nn.Linear(feat_dim, num_classes)
        self.scales = Parameter(torch.ones(num_classes))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False

    def forward(self, x, pos_grad, *args):
        if self.margin_mode == 'pos_grad':
            # pdb.set_trace()
            margin = pos_grad
        if self.training and self.margin_cls:
            margin = margin.type_as(x)
            margin = margin.expand(x.shape[0], -1)
            # print(pos_grad.shape)
            # print(margin.shape)
            # print(self.fc(x).shape)
            x = self.fc(x) + torch.log(margin)
        else:
            x = self.fc(x)
        x *= self.scales
        return x, None


def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False,
                 freq_path=None, margin_cls=False, margin_mode='freq', *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim, freq_path, margin_cls, margin_mode)

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
