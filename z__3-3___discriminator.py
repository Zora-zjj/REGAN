# -*- coding: utf-8 -*

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Discriminator(nn.Module):   #文本分类，输入caption  x: (batch_size , seq_len)，用CNN返回整个x的分类
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size , seq_len)
        """
        emb = self.emb(x).unsqueeze(1)       # [batch_size,1,seq_len,emb_dim]
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))
        return pred          # [batch_size,num_classes]

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


class LSTMDiscriminator(nn.Module):     #用lstm返回最后一步的分类，同上面class好像返回结果类型一样
    """
        Many to one LSTM
    """
    def __init__(self, num_classes, vocab_size, hidden_dim, use_cuda=False):
        super(LSTMDiscriminator, self).__init__()
        # self.emb = nn.Embedding(vocab_size, emb_dim)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()

    """
        x is output of Generator
        x dimensions: (batch_size, seq_len, vocab_size)

    """
    def forward(self, x):
        # input x is now [batch_size,seq_len,vocab_size]
        
        h0, c0 = self.init_hidden(x.size(0))
        output, (h, c) = self.lstm(x, (h0, c0))  # output dim: (batch_size, seq_length, hidden_dim)
        seq_len = output.size()[1]
        batch_size = output.size()[0]
        output = self.lin(output.contiguous())[: , -1 , : ] # only need last lstm block's output只需要最后一个lstm块的输出
        return self.softmax(output.contiguous()) # returning dim   [batch_size, 1, num_classes]

    def init_hidden(self, batch_size):
        # noise distribution fed to G
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    def init_parameters(self):
        for param in self.parameters():
            param.data.normal_(0, 0.02)



