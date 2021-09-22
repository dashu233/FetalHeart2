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
import torch.nn as nn
from utils.draw import draw_pic,draw_cam,draw_cam_double
import matplotlib.pyplot as plt

from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        #print('image_shape:',self.image_shape)
        logits0 = self.model(image).view(-1,1)
        self.logits = torch.cat([-logits0,logits0],dim=1)
        self.probs = self.logits.sigmoid()
        return self.probs[:,1]  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer,normal=True):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        #print('grads_size:',grads.shape)
        #print('grads:',grads)
        weights = torch.mean(grads,dim=1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="linear", align_corners=False
        )

        #print(gcam)

        B, C, H = gcam.shape

        if normal:
            gcam = gcam.view(B, -1)
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam /= gcam.max(dim=1, keepdim=True)[0]
            gcam = gcam.view(B, C, H)

        return gcam




if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    best_model_path = cfg.get('test', 'model')
    model = BinaryPredConv()
    with open(best_model_path, 'rb') as f:
        info = torch.load(f,map_location=torch.device('cpu'))
        for k in info.keys():
            if k == 'model':
                continue
            #print(k, ':', info[k])
        weight_dict = {}
        for k in info['model']:
            weight_dict[k[7:]] = info['model'][k]

        model.load_state_dict(weight_dict)
    model.eval()
    #test_list = cfg.getlistint('test', 'test_list')

    test_start_list = cfg.getlistint('data', 'test_start')
    test_end_list = cfg.getlistint('data', 'test_end')

    test_list = []
    for i in range(len(test_start_list)):
        test_list.extend([_ for _ in range(test_start_list[i],test_end_list[i])])

    #test_list = [_ for _ in range(4482,4506)]

    argument_list = cfg.getliststr('data', 'test_argument')
    transforms = [DATA_ARGUMENT_DICT[ag]() for ag in argument_list]

    test_dataset = \
        BinaryFetalHeartDataset(test_list, transforms, reidx=True,zero_method='linear')
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)
    print(len(test_dataloader))
    model.to(device)
    gcam = GradCAM(model)
    target_layer = 'layer4'

    print('target layer:', target_layer)

    for i,(data,label,pid) in enumerate(test_dataloader):
        data = data.to(device)
        label = label.to(device)
        gcam.forward(data)
        output = model(data)
        output = output.sigmoid()
        output = output.view(-1)
        ids0 = torch.zeros([data.size(0),1],dtype=torch.int64,device=device)

        gcam.backward(ids=ids0)
        normal = True
        regions0 = gcam.generate(target_layer,normal)

        # draw data
        #draw_pic('test_pic',data,cfg.getfloat('Normalizer','mean'),cfg.getfloat('Normalizer','var'))
        #print(regions1)
        draw_cam('cam_retrain_normal/{}_{}'.format(target_layer,pid),data,regions0,
                 cfg.getfloat('Normalizer','mean'),cfg.getfloat('Normalizer','var'),
                        output=output,target=label.view(-1))

        #print(regions.size())


    #eval(model, test_dataloader, device='cpu', verbose=True, details=True, draw=True)


