import torch
import os
import pdb
import torch.nn.functional as F


# import


def gen_label_feat_dict(feats_all, labels_all, num_classes=100):
    label_feat_dict = {}
    for idx in range(num_classes):
        label_feat_dict[idx] = []

    for idx in range(len(feats_all)):
        feat = feats_all[idx]
        label = labels_all[idx]
        # pdb.set_trace()
        label_feat_dict[label.item()].append(feat)
    return label_feat_dict


def cal_intra_cos_sim(label_feat_dict, num_classes=100):
    intra_class_cos_sim = []
    for idx in range(num_classes):
        feat_class_list = label_feat_dict[idx]
        cos_sim_class_list = []

        for i_idx in range(len(feat_class_list) - 1):
            for j_idx in range(i_idx + 1, len(feat_class_list), 1):
                cos_sim_class_list.append(F.cosine_similarity(feat_class_list[i_idx], feat_class_list[j_idx], dim=0))

        # for i_idx in range(len(feat_class_list)):
        #     cos_sim_class_list.append(F.cosine_similarity(feat_class_list[i_idx], center_each_class[idx], dim=0))
        intra_class_cos_sim.append(sum(cos_sim_class_list).item() / len(cos_sim_class_list))

    # print('intra_class_dim: ')
    # print(intra_class_sim)
    return intra_class_cos_sim


def cal_inter_cos_sim(label_feat_dict, num_classes=100):
    cate_cossim_matrix = torch.zeros((num_classes, num_classes), device='cuda')
    for cate_i in range(num_classes - 1):
        # if cate_i == 1:
        #     break
        for cate_j in range(cate_i + 1, num_classes, 1):

            print(str(cate_i) + '  ' + str(cate_j))

            cossim_i_j_list = []
            feat_list_i = label_feat_dict[cate_i]
            feat_list_j = label_feat_dict[cate_j]
            for feat_i in feat_list_i:
                for feat_j in feat_list_j:
                    cossim_i_j_list.append(F.cosine_similarity(feat_i, feat_j, dim=0))
            cate_cossim_matrix[cate_i][cate_j] += sum(cossim_i_j_list) / len(cossim_i_j_list)
            cate_cossim_matrix[cate_j][cate_i] += sum(cossim_i_j_list) / len(cossim_i_j_list)

    inter_class_sim = torch.sum(cate_cossim_matrix, dim=0) / num_classes
    return inter_class_sim


def cal_intra_inter_l2_dist(label_feat_dict, num_classes=100):
    # 计算每个类别的center
    center_each_class = []
    for idx in range(num_classes):
        feat_class_list = label_feat_dict[idx]
        center_this_class = 0
        for feat in feat_class_list:
            center_this_class += feat
        center_this_class = center_this_class / len(feat_class_list)
        center_each_class.append(center_this_class)

    # 计算 intra l2 距离
    intra_class_l2_dist = []
    for idx in range(num_classes):
        feat_class_list = label_feat_dict[idx]
        l2_dist_each_class_list = []

        for i_idx in range(len(feat_class_list)):
            l2_dist_each_class_list.append(torch.dist(feat_class_list[i_idx], center_each_class[idx], p=2))  # TODO

        # for i_idx in range(len(feat_class_list)):
        #     cos_sim_class_list.append(F.cosine_similarity(feat_class_list[i_idx], center_each_class[idx], dim=0))
        intra_class_l2_dist.append(sum(l2_dist_each_class_list).item() / len(l2_dist_each_class_list))

    # 计算 inter l2 距离
    cate_inter_l2_dist_matrix = torch.zeros((num_classes, num_classes), device='cuda')
    for cate_i in range(len(center_each_class) - 1):
        for cate_j in range(cate_i + 1, len(center_each_class), 1):
            dist = torch.dist(center_each_class[cate_i], center_each_class[cate_j], p=2)  # TODO
            cate_inter_l2_dist_matrix[cate_i][cate_j] = dist
            cate_inter_l2_dist_matrix[cate_j][cate_i] = dist
    inter_class_l2_dist = cate_inter_l2_dist_matrix.sum(dim=0) / num_classes

    return intra_class_l2_dist, inter_class_l2_dist


epochs = [2, 16, 33, 50, 66, 83, 100]
for epoch in epochs:
    # load_path = '/home2/qinwei/project/LT_Project/logs/CIFAR100_LT/models/resnet32_softmax_imba200_8/saved_feats/'+str(epoch)+'_feats.pth'
    # load_path = '/home2/qinwei/project/LT_Project/logs/CIFAR100_LT/models/resnet32_softmax_imba200_margin_cls_2/saved_feats/'+str(epoch)+'_feats.pth'
    load_path = '/home2/qinwei/project/LT_Project/logs/CIFAR100_LT/models/resnet32_softmax_imba1_collect_feat/saved_feats/' + str(
        epoch) + '_feats.pth'
    feats_dict = torch.load(load_path)

    print(load_path)

    epoch = feats_dict['epoch']
    feats_all = feats_dict['feats']
    labels_all = feats_dict['labels']

    feats_all = torch.cat(feats_all).cuda()  # TODO cuda； 设置显卡；
    labels_all = torch.cat(labels_all).cuda()

    label_feat_dict = gen_label_feat_dict(feats_all, labels_all)

    # inter_class_cos_sim = cal_inter_cos_sim(label_feat_dict)
    # print('inter cos sim:' + str(sum(inter_class_cos_sim).item()/len(inter_class_cos_sim)))

    # intra_class_cos_sim = cal_intra_cos_sim(label_feat_dict)
    # print('intra cos sim:' + str(sum(intra_class_cos_sim)/len(intra_class_cos_sim)))

    intra_class_l2_dist, inter_class_l2_dist = cal_intra_inter_l2_dist(label_feat_dict)
    print('intra l2:' + str(sum(intra_class_l2_dist) / len(intra_class_l2_dist)))
    print('inter l2:' + str(sum(inter_class_l2_dist).item() / len(inter_class_l2_dist)))

    # print(load_path)

# pdb.set_trace()
