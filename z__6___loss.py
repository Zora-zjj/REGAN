'''
Loss functions.
'''

import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 

import utils


class NLLLoss(nn.Module):
    """Self-Defined NLLLoss Function

    Args:
        weight: Tensor (num_class, )      # weight:[C,1]    # num_class = C
    """
    def __init__(self, weight):
        super(NLLLoss, self).__init__()
        self.weight = weight

    def forward(self, prob, target):
        """
        Args:
            prob: (N, C)  #维度           # prob: [N, C]
            target : (N, )               # target : [N,1]
        """
        N = target.size(0)
        C = prob.size(1)
        weight = Variable(self.weight).view((1, -1))  # weight:[1,C]
        weight = weight.expand(N, C)                  # weight:[N, C]  
        if prob.is_cuda:
            weight = weight.cuda()
        prob = weight * prob                          # (N, C)*(N, C) ???能乘吗，但是维度确实如此

        one_hot = torch.zeros((N, C))                 # one_hot:[N, C]  
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)    #scatter_(input, dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中。
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()               # 对于target元素是1的，取对应索引的prob值
        loss = torch.masked_select(prob, one_hot)  # torch.masked_select(input, mask, out=None) → Tensor
                                                   # 根据mask中的二元值，取input中的指定项，将取值返回到一个新的1D张量
        return -torch.sum(loss)


class GANLoss(nn.Module):

    """Reward-Refined NLLLoss Function for adversial training of Generator"""   #奖励-改进NLLLoss函数，用于生成器的逆向训练

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward_reinforce(self, prob, target, reward, cuda=False):
        """
        Forward function used in the SeqGAN implementation. 
        Args:
            prob: (N, C), torch Variable                      # 与NLLLoss区别：没有prob = weight * prob ，没有weight
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward                                  # 与NLLLoss区别：加了reward
        loss =  -torch.sum(loss)

        return loss
    
    def forward_reward(self, i, samples, prob, rewards, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda=False):
        """
        Returns what is used to get the gradient contribution of the i-th term of the batch.返回用于获取批处理的第i项的梯度贡献的内容

        """ 
        conditional_proba = Variable(torch.zeros(BATCH_SIZE, VOCAB_SIZE))     # conditional_proba：[b,vocab_size]
        if cuda: 
            conditional_proba = conditional_proba.cuda()
        for j in range(BATCH_SIZE):
            conditional_proba[j, int(samples[j, i])] = 1
            conditional_proba[j, :] = - (rewards[j]/BATCH_SIZE * conditional_proba[j, :]) 

        return conditional_proba

    def forward_reward_grads(self, samples, prob, rewards, g, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda=False):
        """
        Returns a list of gradient contribution of every term in the batch.返回批处理中每个项的梯度贡献列表

        """
        conditional_proba = Variable(torch.zeros(BATCH_SIZE, g_sequence_len, VOCAB_SIZE))
        batch_grads = []
        if cuda:
            conditional_proba = conditional_proba.cuda()
        for j in range(BATCH_SIZE):
            for i in range(g_sequence_len):
                conditional_proba[j, i, int(samples[j, i])] = 1
            conditional_proba[j, :, :] = - (rewards[j] * conditional_proba[j, :, :])
        for j in range(BATCH_SIZE):
            j_grads = []
            # since we want to isolate each contribution, we have to zero the generator's gradients here. 
            g.zero_grad()
            prob[j, :, :].backward(conditional_proba[j, :, :], retain_graph=True)
            for p in g.parameters():
                j_grads.append(p.grad.clone())
            batch_grads.append(j_grads)

        return batch_grads

class VarianceLoss(nn.Module):

    """Loss for the control variate annex network"""     #控制变量的annex network的loss

    def __init__(self):
        super(VarianceLoss, self).__init__()

    def forward(self, grad, cuda = False):
        """
        Used to get the gradient of the variance.       #用于从方差得到梯度

        """
        bs = len(grad)
        ref = 0
        for j in range(bs):
            for i in range(len(grad[j])):
                ref += torch.sum(grad[j][i]**2).item()
        total_loss = np.array([ref/bs])
        total_loss = Variable(torch.Tensor(total_loss), requires_grad=True)
        if cuda:
            total_loss = total_loss.cuda()

        return total_loss

    def forward_variance(self, grad, cuda=False):
        """
        Used to get the variance of one single parameter.   #用于得到单一参数的方差
        In this case, we take look at the last layer, then take the variance of the first parameter of this last layer in main.py

        """                                                 #看最后一层，得到最后一层的第一个参数的方差in main.py
        bs = len(grad)
        n_layers = len(grad[0])
        square_term = torch.zeros((grad[0][n_layers-1].size()))
        normal_term = torch.zeros((grad[0][n_layers-1].size()))
        if cuda:
            square_term = square_term.cuda()
            normal_term = normal_term.cuda()
        for j in range(bs):
            square_term = torch.add(square_term, grad[j][n_layers-1]**2)    #x.add_(1)：对x的每个元素加1  ？？？？
            normal_term = torch.add(normal_term, grad[j][n_layers-1])
        square_term /= bs
        normal_term /= bs
        normal_term = normal_term ** 2

        return square_term - normal_term



