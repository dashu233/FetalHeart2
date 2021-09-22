import torch
from torch.utils.data.dataset import Dataset
from data_argumentation import *
import pickle
import matplotlib.pyplot as plt
import math
import copy
EPS = 1e-8


def linear_zero(ar):
    rt = copy.deepcopy(ar)
    i = 0
    #print(len(ar))
    while i < len(ar):
        #print(i)
        if math.fabs(ar[i]) < EPS:
            start_id = i
            while i < len(ar) and math.fabs(ar[i]) < EPS:
                i += 1
            end_id = i
            if start_id > 0:
                start_value = ar[start_id-1]
            else:
                start_value = ar[end_id]
            if end_id < len(ar):
                end_value = ar[end_id]
            else:
                end_value = ar[start_id-1]

            rt[start_id:end_id] = np.array(
                [start_value + i/float(end_id - start_id) * (end_value-start_value) for i in range(end_id-start_id)])
        i += 1
    return rt

class BinaryFetalHeartDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_id_list, transform=None,draw=False,reidx=False,device='cpu',zero_method=None):
        self.reidx=reidx
        if type(transform)!=list:
            print(type(transform))
            print('convert!')
            transform = [transform]
        self.raw_data = []
        self.raw_label = []


        double_num = cfg.getint('data','double_num')
        data_dir = cfg.get('data','data_dir')
        self.file_list = file_id_list
        self.file_map = []
        self.abort_list = []
        for i in file_id_list:
            with open(data_dir+'/{}.pkl'.format(i), 'rb') as f:
                dt = pickle.load(f)
                if dt['y'] == 1:
                    for _ in range(double_num):
                        if zero_method == 'linear':
                            rdx = linear_zero(dt['x'])
                        else:
                            rdx = dt['x']
                        self.raw_data.append(rdx)
                        self.raw_label.append(dt['y'])
                        self.file_map.append(i)
                else:
                    if zero_method == 'linear':
                        rdx = linear_zero(dt['x'])
                    else:
                        rdx = dt['x']
                    self.raw_data.append(rdx)
                    self.raw_label.append(dt['y'])
                    self.file_map.append(i)
                if 'name' in dt:
                    print('file{}:{}'.format(i,dt['name']))

                    #plt.cla()
        if cfg.get('model','type') == 'fft':
            self.raw_data = np.array(self.raw_data)
            assert self.raw_data.shape[1] == 4800, 'fft only support datalen == 4800'
            transfered_data = np.fft.fft(self.raw_data)
            self.raw_data = torch.Tensor([transfered_data.real,transfered_data.imag])
            #print(self.raw_data.size())
            self.raw_data = self.raw_data.permute(1,0,2).reshape(-1,9600).to(device)
            #
        else:
            self.raw_data = torch.Tensor(self.raw_data)
        self.raw_label = torch.Tensor(self.raw_label)
        #print('label1:',len(torch.where(self.raw_label>0.5)[0]),
        #      'label0:',len(torch.where(self.raw_label<0.5)[0]))
        self.transform = transform


    def __len__(self):
        return len(self.raw_label)

    def __getitem__(self, idx):
        #print(idx)
        batchx = self.raw_data[idx]


        if isinstance(idx,int):
            batchx = batchx.view(1,-1)
        #print(batchx)
        if self.transform:
            for ttt in self.transform:
                batchx = ttt(batchx)

        batchy = self.raw_label[idx]
        if isinstance(idx,int):
            batchx.squeeze_(0)
        if len(batchx.size()) < 2:
            batchx.unsqueeze_(0)
        if self.reidx:
            return batchx,batchy,self.file_list[idx]
        else:
            return batchx,batchy

if __name__ == '__main__':

    import ast
    trans = ast.literal_eval(cfg.get('data','argument'))
    transforms = []
    for t in trans:
        transforms.append(DATA_ARGUMENT_DICT[t])
    dataset = BinaryFetalHeartDataset(file_id_list=[1,3,5,7],transform=transforms)