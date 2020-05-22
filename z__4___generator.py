# -*- coding: utf-8 -*-

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):      #LSTM，输入是图片特征？？是caption把？？？，x：[b,seq_len],数据是id
    """Generator """
    def __init__(self, num_emb, emb_dim, hidden_dim, use_cuda):
        super(Generator, self).__init__()
        self.num_emb = num_emb    #vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.emb = nn.Embedding(num_emb, emb_dim)    
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.softmax = nn.LogSoftmax()    #softmax输出都是0-1之间的，因此logsofmax输出的是小于0的数
        self.init_params()                #init_params后面函数，参数初始化

    def forward(self, x):      #x是caption？？？
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
        emb = self.emb(x)      # emb:[b,seq_len,emb_size] 是输入
        h0, c0 = self.init_hidden(x.size(0))      #init_hidden后面函数， h和c初始化  x.size(0)是batch_size
        output, (h, c) = self.lstm(emb, (h0, c0))   # output：[b,seq_len,hidden_size]
        pred = self.softmax(self.lin(output.contiguous().view(-1, self.hidden_dim)))    #view只能用在contiguous的variable上，用contiguous()来返回一个contiguous copy
        return pred                               #生成预测值  #pred : [b*seq_len,num_emd] ,vocab_size

    def step(self, x, h, c):     # x是句子的第一个单词， step：根据上个单词，预测下一个单词，lstmcell
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        emb = self.emb(x)
        output, (h, c) = self.lstm(emb, (h, c))   # oi,hi,ci,输出 o(i+1),h(i+1),c(i+1)   # output：[b,1,hidden_size]
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)))
        return pred, h, c   #  pred:[b*1,num_emb]   h/c:[1, batch_size, hidden_dim]    #说明当batch_first=True时，只有x和output是b前，h和c都是后


    def init_hidden(self, batch_size):        #h和c初始化
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c
    
    def init_params(self):                    #参数初始化
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)  #将data用从均匀分布中抽样得到的值填充

    def sample(self, batch_size, seq_len, x=None, sampleFromZero=False):  # 输出output：[b,seq_len]，数据：单词id   #将step函数合并成一个句子
        res = []
        if x is None:
            sampleFromZero = True
        if sampleFromZero:
            x = Variable(torch.zeros((batch_size, 1)).long())      #初始x0为0 维度[batch,1]
        if self.use_cuda:
            x = x.cuda()
        h, c = self.init_hidden(batch_size)                 #初始h，c为0
        samples = []
        if sampleFromZero:     #当输入是None，自定义x0，else是输入是xt，不是None
            for i in range(seq_len):
                output, h, c = self.step(x, h, c)    #output：[b*1,num_emb]即vocab_size
                x = output.multinomial(1)            #同multinomial(output,n):对output每一行作n次取值，输出张量是对应的坐标，即单词的id
                samples.append(x)    #samples：[b,seq_len,vocab_size]
        else:
            given_len = x.size(1)           # given_len < seq_len      # 初始x为xt：[b,seq_len]
            lis = x.chunk(x.size(1), dim=1) # x.chunk(seq_len,1),在维度1上将x分成seq_len份   #lis[i]维度:[b,1]为上面的x0，x1，x2。。。
            for i in range(given_len):
                output, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = output.multinomial(1)
            for i in range(given_len, seq_len):  #given_len和seq_len不一样，这个for的作用是什么？？？
                samples.append(x)
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output   #输出output：[b,seq_len]，数据：单词id