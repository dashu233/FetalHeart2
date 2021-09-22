import torch
import numpy as np

from config import cfg
import pickle
import numpy as np
import torch
from model import BinaryPredConv,ClipModel
from data_argumentation import DATA_ARGUMENT_DICT
from utils.draw import draw_pic,draw_pic_clip
import os
from torch.nn.parallel import DistributedDataParallel
from dataset import BinaryFetalHeartDataset
from train import eval
from torch.utils.data.dataloader import DataLoader
#from detectron2.utils import comm
#from detectron2.engine import launch
import sys
import argparse
from utils.log import setup_logger
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [1.2, 2.5, 4.5, 7.3]

def main():
    for _ in range(cfg.getint('model','embed_dim')):
        if not os.path.exists('picture_trans/class_{}'.format(_)):
            os.mkdir('picture_trans/class_{}'.format(_))
    thr = 0.5
    vote_thr = 50
    Draw = True
    best_model_path = cfg.get('test', 'model')
    model = ClipModel()
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
    #test_list = cfg.getlistint('test', 'test_list')

    patient_list = np.load('patient_list_remain.npy').transpose()
    argument_list = cfg.getliststr('data', 'test_argument')
    transforms = [DATA_ARGUMENT_DICT[ag]() for ag in argument_list]
    logger = setup_logger()
    wrong_patient_list = []

    patient_predict_list = []
    patient_acc = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for pti in range(patient_list.shape[0]):
        single_predict = []
        from utils.pather import PathManager
        pt = patient_list[pti]
        if Draw:
            PathManager.mkdirs('picture')
            PathManager.mkdirs('picture/{}'.format(pti))
        data_list = [_ for _ in range(pt[0],pt[1])]
        test_dataset = \
            BinaryFetalHeartDataset(data_list, transforms, zero_method='linear',reidx=True)
        test_dataloader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)
        model.eval()
        correct_pos = 0
        correct_neg = 0
        neg_num = 0
        pos_num = 0
        pred_pos = 0
        pred_neg = 0


        patient_label = 0
        for i, (data, target, idx) in enumerate(test_dataloader):
            patient_label = target.item()
            data, target = data.to(device), target.to(device)
            model_type = cfg.get('model', 'type')
            if model_type == 'lstm':
                data = data.permute(1, 0, 2)  # fit the input of lstm
            # print(data.size())
            if model_type == 'Clip':
                output,pred_class = model(data,pred_class=True)
            else:
                output = model(data)
            output.sigmoid_()
            output = output.view(-1)
            # print('eval:',output)
            target = target.view(-1)
            if not len(single_predict):
                single_predict.append(target.item())
            single_predict.append(output.item())
            if Draw:
                if not os.path.exists('picture_trans/{}'.format(int(pti))):
                    os.mkdir('picture_trans/{}'.format(int(pti)))
                pic_name = 'picture_trans/{}/'.format(int(pti)) + str(int(idx))
                draw_pic(pic_name,data,cfg.getfloat('Normalizer', 'mean'),
                         cfg.getfloat('Normalizer', 'var'),target=target,output=output)
                seq_len = cfg.getint('model','sequence_len')
                cls_num = cfg.getint('model','embed_dim')
                pred_id = torch.argmax(pred_class.view(-1,cls_num),dim=1)
                #print('plot')
                #print(data.shape)
                #print(pred_class.shape)
                #print('pred_id:',pred_id)
                for j in range(5):

                    print('plot:',j)
                    pic_name = 'picture_trans/class_{}/{}_{}_{}.png'.format(int(pred_id[j]),int(pti),int(idx),j)
                    draw_pic_clip(pic_name,data.view(-1)[seq_len*j:seq_len*(j+1)],cfg.getfloat('Normalizer', 'mean'),
                         cfg.getfloat('Normalizer', 'var'),target=target,output=pred_id[j])


            pos_ids = torch.where(target > 0.001)[0]
            neg_ids = torch.where(target == 0)[0]

            correct_pos += len(torch.where(output[pos_ids] >= thr)[0])
            correct_neg += len(torch.where(output[neg_ids] < thr)[0])
            neg_num += len(neg_ids)
            pos_num += len(pos_ids)
            pred_pos += len(torch.where(output>=thr)[0])
            pred_neg += len(torch.where(output<thr)[0])
        patient_predict_list.append(single_predict)

        accuracy_all = -1.0 if not pos_num + neg_num else 100. * (correct_pos + correct_neg) / (pos_num + neg_num)
        logger.info('Accuracy for patient {}: {}%({}/{})'.format(pti,accuracy_all,
                                                                 correct_pos+correct_neg,pos_num+neg_num))
        pred_res = pred_pos/(pred_neg + pred_pos)
        sick = pred_res>(vote_thr/100.0)
        if (sick and patient_label<0.001) or (not sick and patient_label>0.001):
            wrong_patient_list.append(pti)
        patient_acc.append(accuracy_all)
    print('wrong_patient_list:',wrong_patient_list)
    PathManager.mkdirs('vote_result')
    np.save('vote_result/patient_acc_{}_{}.npy'.format(int(thr*100),vote_thr),np.array(patient_acc))

    num_patient = len(patient_predict_list)
    max_data = max([len(d) for d in patient_predict_list])
    patient_result = -np.ones([num_patient,max_data])
    for inp in range(num_patient):
        datalen = len(patient_predict_list[inp])
        for dnp in range(datalen):
            patient_result[inp,dnp] = patient_predict_list[inp][dnp]
    np.save('vote_result/patient_result_{}_{}.npy'.format(int(thr*100,),vote_thr),patient_result)


if __name__ == '__main__':
    main()


