from config import cfg
import pickle
import numpy as np
import torch
from model import BinaryPredConv
from data_argumentation import DATA_ARGUMENT_DICT
import os
from torch.nn.parallel import DistributedDataParallel
from dataset import BinaryFetalHeartDataset
from train import eval
from torch.utils.data.dataloader import DataLoader


def main():
    Draw = False
    best_model_path = cfg.get('test', 'model')
    model = BinaryPredConv()
    with open(best_model_path, 'rb') as f:
        info = torch.load(f,map_location=torch.device('cpu'))
        for k in info.keys():
            if k == 'model':
                continue
            print(k, ':', info[k])
        weight_dict = {}
        for k in info['model']:
            weight_dict[k[7:]] = info['model'][k]

        model.load_state_dict(weight_dict)
    model.eval()
    #test_list = cfg.getlistint('test', 'test_list')

    train_start_list = cfg.getlistint('data', 'train_start')
    train_end_list = cfg.getlistint('data', 'train_end')
    eval_start_list = cfg.getlistint('data', 'eval_start')
    eval_end_list = cfg.getlistint('data', 'eval_end')
    test_start_list = cfg.getlistint('data', 'test_start')
    test_end_list = cfg.getlistint('data', 'test_end')

    test_list = []
    for i in range(len(test_start_list)):
        test_list.extend([_ for _ in range(test_start_list[i],test_end_list[i])])

    test_list = [_ for _ in range(4482,4507)]

    argument_list = cfg.getliststr('data', 'test_argument')
    transforms = [DATA_ARGUMENT_DICT[ag]() for ag in argument_list]

    if Draw:
        test_dataset = \
            BinaryFetalHeartDataset(test_list, transforms, reidx=True)
        test_dataloader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)
        print(len(test_dataloader))

        eval(model, test_dataloader, device='cpu', verbose=True, details=True,draw=True)
    else:
        test_dataset = \
            BinaryFetalHeartDataset(test_list, transforms)
        test_dataloader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)
        eval(model, test_dataloader, device='cpu', verbose=True)

if __name__ == '__main__':
    main()


