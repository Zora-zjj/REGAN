# -*- coding:utf-8 -*-

import os, sys
import random
import math

import tqdm
import numpy as np
import torch


class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)      #read_file后面函数，返回单词列表，不去重
        self.data_num = len(self.data_lis)             #单词总数，不去重
        self.indices = range(self.data_num)
        self.num_batches = int(math.floor(float(self.data_num) / self.batch_size))           #几个batch
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)   

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]                 # 某个batch内 index 范围
        d = [self.data_lis[i] for i in index]                                   # 某个batch内的单词序列，[batch_size,1]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))                      #将单词转为np数据？？？怎么转
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)    # cat 合并 (batch_size，2）[0,单词]
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)   #        （batch_size，2）[单词,0] 
        self.idx += self.batch_size        #每个batch的第一个idx号

        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')                           #.strip()移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            l = [int(s) for s in l]
            lis.append(l)
        return lis                                                #单词序列

class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data_file, fake_data_file, batch_size, seq_len):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        real_data_lis = self.read_real_file(real_data_file)
        fake_data_lis = self.read_fake_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        #self.data_num = sum(1 for _ in self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.floor(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size

        return data, label

    def read_real_file(self, data_file):   #真实数据
        char_to_ix = {
            'x': 0,
            '+': 1,
            '-': 2,
            '*': 3,
            '/': 4
            # '_': 5,
            # '\n': 6
        }
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = list(line)[:-1]
            l = [char_to_ix[s] for s in l]
            # weird fix: sometimes, the generated sequence has length 14 and not 15...
            if len(l) < self.seq_len:
                l.append(0)
            assert len(l) == self.seq_len
            lis.append(l)
        return lis

    def read_fake_file(self, data_file):  #虚假数据
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis
