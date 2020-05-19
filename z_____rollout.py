# -*- coding:utf-8 -*-

import os
import random
import math
import copy

import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from helpers import *

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate):
        self.ori_model = model                #浅拷贝，类似copy.copy ,当model改变时ori_model改变
        self.own_model = copy.deepcopy(model) #深拷贝                当model改变时own_model不改变
        self.update_rate = update_rate

    def get_reward(self, x, discriminator, VOCAB_SIZE, cuda):
        """
        Args:
            x : (batch_size, seq_len) input data  #数据：单词id
            discriminator : discrimanator model
            Directly outputting the prob of one sequence (no rollout) 直接输出一个序列的prob
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # samples = self.own_model.sample(batch_size, seq_len, x)
        one_hot_samples = convert_to_one_hot(x, VOCAB_SIZE, cuda)   #x的one-hot形式，[b,seq_len,vocab_size]
        pred = discriminator(one_hot_samples)  # pred:[b,1],一个batch内第b个句子的真假概率
        pred = pred.cpu().data[:,1].numpy()
        
        return pred
    
    def get_reward_mc(self, x, num, discriminator):  #Monte Carlo？？？
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.own_model.sample(batch_size, seq_len, data)
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred
            # for the last token
            pred = discriminator(x)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]


