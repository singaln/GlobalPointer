#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: SinGaln
# @time: 2022/1/11 10:21

import json
import torch
import random
import logging
import numpy as np


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    详情见: https://kexue.fm/archives/7359
    y_true和y_pred的shape一致，y_true的元素非0即1，
    1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    :param y_true: [batch_size, num_labels, seq_len, seq_len]
    :param y_pred: [batch_size, num_labels, seq_len, seq_len]
    :return: loss scores
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


class Metrics(object):
    """
    f1: 2 * precision * recall / precision + recall
    precision_score: pred_true / all_pred
    recall_score:
    """

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


# 初始化log的输出
def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def write(data_path, word2id, tag2id):
    with open(data_path + "/word2id.json", "w", encoding="utf-8") as f1, open(data_path + "/tag2id.json", "w",
                                                                              encoding="utf-8") as f2:
        token2id = json.dumps(word2id, ensure_ascii=False, indent=2)
        label2id = json.dumps(tag2id, ensure_ascii=False, indent=2)
        f1.write(token2id)
        f2.write(label2id)
