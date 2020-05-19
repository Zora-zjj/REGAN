
import os, sys
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def convert_to_one_hot(data, vocab_size, cuda):        #返回samples：[b,seq_len,vocab_size],数据：0，1编码   #该代码太繁琐不好
    """
        data dims: (batch_size, seq_len)   #数据id？？？
        returns:(batch_size, seq_len, vocab_size)
    """
    batch_size = data.size(0)
    seq_len = data.size(1)

    samples = Variable(torch.Tensor(batch_size, seq_len, vocab_size))  #空数据，后面填
    one_hot = Variable(torch.zeros((batch_size, vocab_size)).long())   #数据为0

    if(cuda):
        # data = data.cuda()
        samples = samples.cuda()
        one_hot = one_hot.cuda()

    for i in range(batch_size):   
        x = data[i].view(-1,1)     # data[i]:[seq_len],表示第i个b的数据    # x:[seq_len,1]   #a：[2,3,4] ,则a.shape[0]为2，a.shape[0]为3
        one_hot = Variable(torch.zeros((seq_len, vocab_size)).long())
        if cuda:
            one_hot = one_hot.cuda()
            x = x.cuda()
        samples[i] = one_hot.scatter_(1, x, 1.0)   #input.scatter_(dim,index,src)将src中数据根据index中的索引按照dim的方向填进input中
                                                   #在第i个batch中,[seq_len,vocab_size],对应位置填上1.0
    return samples
