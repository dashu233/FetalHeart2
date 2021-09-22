import numpy as np
import torch
import torch.nn as nn
from inspect import signature
import logging
from config import cfg
from utils.log import setup_logger
from utils.pather import PathManager
import pickle


# data argumentation

logger = logging.getLogger("FetalHeart")

if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
    setup_logger()
@torch.no_grad()
class Normalizer():
    def __init__(self):
        self.method = cfg.get('Normalizer','method')
        if self.method == 'rescale':
            self.rescale = cfg.getfloat('Normalizer','rescale')
        if self.method == 'shift':
            self.mean = cfg.getfloat('Normalizer','mean')
            self.var = cfg.getfloat('Normalizer','var')

    def __call__(self,x):
        #print(self.method)
        if self.method == 'rescale':
            return x/self.rescale
        if self.method == 'shift':
            #print('mean:',self.mean)
            #print('var:',self.var)
            return (x-self.mean)/self.var


@torch.no_grad()
class DownSampler():
    def __init__(self):
        self.dw_len = cfg.getint('DownSampler','dw_len')
        self.method = cfg.get('DownSampler','method')
    def __call__(self, x:torch.tensor):
        # x: (batch, times)
        if self.method == 'skip':
            return x[:,::self.dw_len]
        elif self.method == 'average':
            return torch.nn.functional.avg_pool1d(x.view(x.size(0),1,-1),self.dw_len).view(x.size(0),-1)
        elif self.method == 'smooth':
            return torch.nn.functional.avg_pool1d(x.view(x.size(0), 1, -1), self.dw_len,stride=1).view(x.size(0), -1)

@torch.no_grad()
class RandZero():
    def __init__(self):
        self.prob = cfg.getfloat('RandZero','prob')
        self.zero_len = cfg.getint('RandZero','zero_len')
    def __call__(self, x:torch.Tensor):
        coin = torch.rand([1]).item()
        if coin < self.prob:
            batch_size = x.size(0)
            z_start = torch.randint(0, 4800-self.zero_len, [batch_size], device=x.device)
            for i in range(batch_size):
                x[i, z_start[i]:z_start[i] +self.zero_len] = torch.zeros([self.zero_len])
            return x
        else:
            return x

@torch.no_grad()
class MultiChannel():
    def __call__(self,x:torch.Tensor):
        # batch,1,len
        x = x.view(x.size(0), 1, -1)
        pad_x = x.new_zeros([x.size(0), 1, x.size(2) + 2])
        pad_x[:, :, 1:-1] = x
        res = x.new_zeros([x.size(0), 3, x.size(2)])
        res[:, 0, :] = x[:, 0, :]
        res[:, 1, :] = (pad_x[:, 0, :-2] - pad_x[:, 0, 2:]) / 2.0
        res[:, 2, :] = pad_x[:, 0, :-2] + pad_x[:, 0, 2:] - 2 * pad_x[:, 0, 1:-1]
        return res

@torch.no_grad()
class Mask():
    def __call__(self,x:torch.Tensor):
        x = x.view(x.size(0), 1, -1)
        res = x.new_ones([x.size(0), 2, x.size(2)])
        res[:,0,:] = x[:,0,:]
        l1,l2,l3 = torch.where(x<0)
        res[l1,1,l3] = 0
        return res

@torch.no_grad()
class Filler():
    def __init__(self,method = 'mean'):
        self.method = method

    def __call__(self, x:torch.Tensor):
        if self.method == 'mean':
            # fill 0 with mean of nonzero term
            for i, data_per_batch in enumerate(x):
                nonzero_indice = torch.where(data_per_batch>0.001)[0]
                #if len(nonzero_indice) == 0:
                    #print('fuckkkkkkk!')
                mean = data_per_batch[nonzero_indice].mean()
                x[i] = torch.where(data_per_batch > 0.001, data_per_batch, mean)
            return x

@torch.no_grad()
class Differentiator():
    # calculate the difference between t and t+1, keep first term as origin
    def __call__(self, x:torch.Tensor):
        try:
            assert x.size(1) > 1
        except AssertionError:
            logger.error('data_argumentation:Differentiator requests x.size(1)>1')
        x_ = x.new_zeros(x.size())
        x_[:, 1:] = x[:, :-1]
        x = x - x_
        return x


@torch.no_grad()
class Fetcher():
    def __init__(self):
        self.method = cfg.get('Fetcher','method')
        self.dim = cfg.getint('data', 'input_dim')
        try:
            assert self.dim > 1
        except AssertionError:
            logger.error('too small input_dim')

        self.seq_len = cfg.getint('data', 'seq_len')
    def __call__(self, x):
        if self.method == 'normal':
            '''
            just pick nearby BPM as feature
            can't behind Differentiator
            '''
            batch_size = x.size(0)
            try:
                assert self.seq_len * self.dim > x.size(1)
            except AssertionError:
                logger.error('data_argumentation: Fetcher request input lenth < seq_len*dim, '
                             'may be caused by too small cfg.data.input_dim or cfg.data.seq_len')
            x_ = x.new_zeros(batch_size,self.seq_len*self.dim)
            x_[:,:x.size(1)] = x
            return x_.view(batch_size,self.seq_len,self.dim).permute(1,0,2)

        if self.method == 'keep_first':
            '''
            covert single float sequence to features sequence
            a feature contains the near dim-1 floats and the last dim is the start float
            no intersection between different sequence
            '''

            batch_size = x.size(0)
            try:
                assert self.seq_len * (self.dim - 1) > x.size(1) - 1
            except AssertionError:
                logger.error('data_argumentation: Fetcher request input lenth < seq_len*(dim-1), '
                             'may be caused by too small cfg.data.input_dim or cfg.data.seq_len')

            time_width = self.dim - 1
            head = x[:, 0].view(-1)
            x_ = x.new_zeros([self.seq_len, batch_size, self.dim])
            # last feature is the start time BPM abs
            x_[:, :, time_width] = head.repeat([self.seq_len, 1])
            x_seq = x.new_zeros([batch_size, self.seq_len * time_width])
            # first time is not diff
            x_seq[:, :x.size(1) - 1] = x[:, 1:]
            x_[:, :, :time_width] = x_seq.view(batch_size, self.seq_len, time_width).permute(1, 0, 2)
            return x_
        if self.method == 'overlap_keep_first':

            batch_size = x.size(0)
            overlap_len = cfg.getint('Fetcher','overlap_len')
            try:
                assert self.seq_len * (self.dim - 1-overlap_len) + overlap_len > x.size(1) - 1
            except AssertionError:
                logger.error('data_argumentation: data_lenth is {}, but only use {} '.format(x.size(1) - 1,self.seq_len * (self.dim - 1-overlap_len) + overlap_len)+
                             'may be caused by too small cfg.data.input_dim, cfg.data.seq_len, or dw_len')
            time_width = self.dim - 1
            time_step = torch.linspace(0,time_width-1,time_width,device=x.device,dtype=torch.int64)
            seq_step = torch.linspace(0,(self.seq_len-1)*(time_width-overlap_len),self.seq_len,device=x.device,dtype=torch.int64)
            time_step = time_step.repeat(self.seq_len)
            seq_step = seq_step.repeat(time_width)
            full_step = time_step + seq_step
            full_step = full_step.view(1,-1).repeat([batch_size,1])
            pad_x = x.new_zeros([batch_size,self.seq_len*time_width])
            pad_x[:,:x.size(1)-1] = x[:,1:]
            patches = torch.gather(pad_x,dim=1,index=full_step)
            patches = patches.view(batch_size,self.seq_len,time_width).permute(1,0,2)

            head = x[:, 0].view(-1)
            x_ = x.new_zeros([self.seq_len, batch_size, self.dim])
            # last feature is the start time BPM abs
            x_[:, :, time_width] = head.repeat([self.seq_len, 1])
            x_[:, :, :time_width] = patches
            return x_


DATA_ARGUMENT_DICT = {'DownSampler':DownSampler,'Filler':Filler,'RandZero':RandZero,'Mask':Mask,
                      'Differentiator':Differentiator,'Fetcher':Fetcher,'Normalizer':Normalizer,
                      'MultiChannel':MultiChannel}

if __name__ == '__main__':
    pass