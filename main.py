import pickle
from train import train_eval_loop
import argparse
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import random
import torch.optim as optim
from model import BinaryPrediction,BinaryPredConv,ClipModel
from config import cfg
from utils.log import setup_logger
from utils.pather import PathManager
from dataset import BinaryFetalHeartDataset
from data_argumentation import DATA_ARGUMENT_DICT
import os
from train import train_eval_loop
import time

if __name__ == '__main__':

    # TODO: distribute training
    # set random seed
    seed = cfg.getint('other','seed')
    random.seed(seed)
    torch.manual_seed(seed)
    logger = setup_logger()
    output_dir = cfg.get('other','output_dir')
    PathManager.mkdirs(output_dir)
    cfg_name = output_dir+'/'+\
               time.asctime( time.localtime(time.time())).replace(' ','_').replace(':','_')+'.conf'
    with open(cfg_name,'w') as f:
        cfg.write(f)

    parser = argparse.ArgumentParser(description='FetalHeart')
    parser.add_argument('--gpu', type=int, default=0,
                        help='number of GPU device to use (default: 0)')
    args = parser.parse_args()
    model_type = cfg.get('model','type')
    if model_type == 'conv':
        model = BinaryPredConv()
    elif model_type == 'clip':
        model = ClipModel()
    else:
        model = BinaryPrediction()
    optimizer = optim.Adam(model.parameters(),
                           lr = cfg.getfloat('train','lr'),
                           betas=(cfg.getfloat('train','beta1'),cfg.getfloat('train','beta2')),
                           weight_decay=cfg.getfloat('train','weight_decay'))

    data_num = len(os.listdir('dataset'))
    train_num = cfg.getint('train','train_num')
    eval_num = cfg.getint('eval','eval_num')
    try:
        assert train_num+eval_num < data_num
    except AssertionError:
        logger.error('main:train_num + eval_num must smaller than data_num. check dataset or your cfg')
        quit()

    fulllist = [i for i in range(data_num)]
    random.shuffle(fulllist)
    train_list = fulllist[:train_num]
    eval_list = fulllist[train_num:train_num+eval_num]

    argument_list = cfg.getliststr('data','argument')
    transforms = [DATA_ARGUMENT_DICT[ag]() for ag in argument_list]
    train_dataset = \
        BinaryFetalHeartDataset(train_list,transforms)
    eval_dataset = \
        BinaryFetalHeartDataset(eval_list,transforms)

    train_dataloader = DataLoader(train_dataset,cfg.getint('train','batch_size'),
                                shuffle=True,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset,cfg.getint('eval','batch_size'),shuffle=False,drop_last=True)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=cfg.getlistint('train','steps'),
                                                       gamma = cfg.getfloat('train','lr_decay'))
    train_eval_loop(model,optimizer,lr_schedule,train_dataloader,eval_dataloader,'cpu',
                    cfg.getint('train','epoch'),verbose=True)