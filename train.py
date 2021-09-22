#from detectron2.utils import comm

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import cfg
from torch.nn.parallel import DistributedDataParallel
from utils.log import setup_logger
from utils.draw import draw_pic
import matplotlib.pyplot as plt

def train(model, optimizer, dataloader, device, epoch, verbose):
    log_interval = cfg.getint('other','log_interval')
    #print('enter train')
    model.train()
    total = 0
    if verbose:
        logger = setup_logger()
    correct_pos = 0
    correct_neg = 0
    neg_num = 0
    pos_num = 0
    weight_sum = 0.0
    #tart training')
    pos_weight = cfg.getfloat('loss','pos_weight')
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        #print('train_data:', data[:10,:10])
        model_type = cfg.get('model','type')
        if model_type=='lstm':
            data = data.permute(1,0,2) # fit the input of lstm
        optimizer.zero_grad()
        if model_type == 'Clip':
            outputs = model(data,True)
            for ot in outputs:
                ot.sigmoid_()
            output = outputs[-1]
        else:
            output = model(data)
            if model_type == 'conv':
                output.sigmoid_()
        output = output.view(-1)
        target = target.view(-1)
        #print('train:', output[:10])

        #batch_weight = target.new_ones(len(target))
        pos_ids = torch.where(target > 0.001)[0]
        neg_ids = torch.where(target == 0)[0]
        #batch_weight[pos_ids] = pos_weight
        loss = nn.BCELoss(reduction='sum')
        if model_type == 'Clip':
            pos_loss = 0
            neg_loss = 0
            for i,ot in enumerate(outputs):
                if not i == len(outputs) - 1:
                    pos_loss += 0.5*loss(ot[pos_ids],target[pos_ids])
                    neg_loss += 0.5*loss(ot[neg_ids],target[neg_ids])
                else:
                    pos_loss += loss(ot[pos_ids], target[pos_ids])
                    neg_loss += loss(ot[neg_ids], target[neg_ids])
        else:
            pos_loss = loss(output[pos_ids],target[pos_ids])
            neg_loss = loss(output[neg_ids],target[neg_ids])


        correct_pos += len(torch.where(output[pos_ids] >= 0.5)[0])
        correct_neg += len(torch.where(output[neg_ids] < 0.5)[0])
        neg_num += len(neg_ids)
        pos_num += len(pos_ids)
        weight_sum += pos_weight * len(pos_ids) + len(neg_ids)
        total += pos_loss.item()*pos_weight+neg_loss.item()
        train_loss = (pos_loss*pos_weight+neg_loss)/(len(pos_ids)*pos_weight + len(neg_ids))
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx == 0):
            logger.info('Train Epoch: {} [{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * data.size(0), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
            logger.info('output10: ' + str(output[:10]))
            logger.info('target10: ' + str(target[:10]))
            average_loss = total / weight_sum
            accuracy_pos = -1.0 if not pos_num else 100. * correct_pos / pos_num
            accuracy_neg = -1.0 if not neg_num else 100. * correct_neg / neg_num
            precision = -1.0 if not correct_pos + neg_num - correct_neg else \
                100. * correct_pos / (correct_pos + neg_num - correct_neg)
            recall = accuracy_pos
            accuracy_all = 100. * (correct_pos + correct_neg) / (pos_num + neg_num)
            logger.info('pos_acc:{},neg_acc:{},all_acc:{},precision:{},recall{}'
                        .format(accuracy_pos,accuracy_neg,accuracy_all,precision,recall))
    average_loss = total / weight_sum
    accuracy_pos = -1.0 if not pos_num else 100. * correct_pos / pos_num
    accuracy_neg = -1.0 if not neg_num else 100. * correct_neg / neg_num
    precision = -1.0 if not correct_pos + neg_num - correct_neg else \
        100. * correct_pos / (correct_pos + neg_num - correct_neg)
    recall = accuracy_pos
    accuracy_all = 100. * (correct_pos + correct_neg) / (pos_num + neg_num)
    if verbose:
        logger.info('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch,average_loss))
        logger.info('pos_acc:{},neg_acc:{},all_acc:{},precision:{},recall{}'
                    .format(accuracy_pos, accuracy_neg, accuracy_all, precision, recall))
    return total / weight_sum,precision,recall

def eval(model, dataloader, device, verbose,details=False,draw=False,thr = 0.5):
    #print('enter eval')
    #dataset = cfg.get('data','data_dir')
    model.eval()
    total = 0
    correct_pos = 0
    correct_neg = 0
    neg_num = 0
    pos_num = 0
    weight_sum = 0.0
    pos_weight = cfg.getfloat('loss','pos_weight')
    if verbose:
        logger = setup_logger()
    #print('dl:',len(dataloader))
    with torch.no_grad():
        if not draw:
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                model_type = cfg.get('model', 'type')
                if model_type == 'lstm':
                    data = data.permute(1, 0, 2)  # fit the input of lstm
                #print(data.size())
                output = model(data)
                if model_type == 'conv' or 'Clip':
                    output.sigmoid_()
                output = output.view(-1)
                # print('eval:',output)
                target = target.view(-1)
                #print('output_size:',output.size())
                if details:
                    print('output', output)
                    print('target', target)

                pos_ids = torch.where(target > 0.001)[0]
                neg_ids = torch.where(target == 0)[0]

                loss = nn.BCELoss(reduction='sum')
                if len(neg_ids):
                    neg_loss = loss(output[neg_ids], target[neg_ids]).item()
                else:
                    neg_loss = 0
                if len(pos_ids):
                    pos_loss = loss(output[pos_ids], target[pos_ids]).item()
                else:
                    pos_loss = 0
                total += (pos_loss * pos_weight + neg_loss)
                weight_sum += pos_weight * len(pos_ids) + len(neg_ids)

                correct_pos += len(torch.where(output[pos_ids] >= thr)[0])
                correct_neg += len(torch.where(output[neg_ids] < thr)[0])
                neg_num += len(neg_ids)
                pos_num += len(pos_ids)
        else:
            for i, (data, target, idx) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                model_type = cfg.get('model', 'type')
                if model_type == 'lstm':
                    data = data.permute(1, 0, 2)  # fit the input of lstm
                # print(data.size())
                output = model(data)
                if model_type == 'conv' or 'Clip':
                    output.sigmoid_()
                output = output.view(-1)
                # print('eval:',output)
                target = target.view(-1)

                pic_name = str(idx)
                draw_pic(pic_name,data,
                         cfg.getfloat('Normalizer','mean'),cfg.getfloat('Normalizer','var'),
                         target=target,output=output)

                pos_ids = torch.where(target > 0.001)[0]
                neg_ids = torch.where(target == 0)[0]
                # batch_weight[pos_ids] = pos_weight
                loss = nn.BCELoss(reduction='sum')
                if len(neg_ids):
                    neg_loss = loss(output[neg_ids], target[neg_ids]).item()
                else:
                    neg_loss = 0
                if len(pos_ids):
                    pos_loss = loss(output[pos_ids], target[pos_ids]).item()
                else:
                    pos_loss = 0
                total += (pos_loss * pos_weight + neg_loss)
                weight_sum += pos_weight * len(pos_ids) + len(neg_ids)

                correct_pos += len(torch.where(output[pos_ids] >= thr)[0])
                correct_neg += len(torch.where(output[neg_ids] < thr)[0])
                neg_num += len(neg_ids)
                pos_num += len(pos_ids)

    average_loss = -1.0 if not weight_sum>0.0001 else total / weight_sum
    accuracy_pos = -1.0 if not pos_num else 100. * correct_pos / pos_num
    accuracy_neg = -1.0 if not neg_num else 100. * correct_neg / neg_num
    precision = -1.0 if not correct_pos+neg_num-correct_neg else \
        100.* correct_pos/(correct_pos + neg_num-correct_neg)
    recall = accuracy_pos
    accuracy_all = -1.0 if not pos_num+neg_num else 100. * (correct_pos + correct_neg)/(pos_num + neg_num)
    if verbose:
        logger.info('Evaluation: Average loss: {:.4f},\n '
              'pos_accuracy:{}/{} ({:.2f}%),\n'
              'neg_accuracy:{}/{} ({:.2f}%),\n'
                'precision:{}/{}({:.2f}%),\n'
                    'recall:{}/{}({:.2f}%),\n'
              'All:{}/{}({:.2f}%),\n F1:{}'.format(
            average_loss,
            correct_pos,pos_num,accuracy_pos,
            correct_neg,neg_num,accuracy_neg,
            correct_pos,correct_pos+neg_num-correct_neg,precision,
            correct_pos,pos_num,recall,
            correct_neg+correct_pos,pos_num+neg_num,accuracy_all,0))
    return average_loss, accuracy_pos,accuracy_neg,accuracy_all,precision,recall

def train_eval_loop(model, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose,exp_dir):
    test_loss, apos, aneg, aall,prec,recall = eval(model, test_loader, device, verbose)
    rows = [[np.nan, test_loss, apos, aneg, aall,prec,recall]]
    checkpoint_interval = cfg.getint('other','checkpoint')
    best_loss = 10000
    if verbose:
        try:
            assert exp_dir is not None
        except AssertionError:
            print('please add exp_dir for train_eval_loop if verbose is True')
            quit()
        logger = setup_logger()
        for epoch in tqdm(range(epochs)):
            train_loss,train_precision,train_recall = train(model, optimizer, train_loader, device, epoch, verbose)
            test_loss, apos, aneg, aall,prec,recall = eval(model, test_loader, device, verbose)
            row = [train_loss, test_loss, apos, aneg, aall,prec,recall,train_precision,train_recall]
            scheduler.step()
            rows.append(row)
            if test_loss < best_loss:
                best_loss = test_loss
                save_dict = {'model':model.state_dict(),'epoch':epoch,
                             'best_loss':test_loss,
                             'pos_acc':apos,'neg_acc':aneg,'combine':aall,
                             'precision':prec,'recall':recall,
                             'pos_weight':cfg.getfloat('loss','pos_weight')}
                torch.save(save_dict,exp_dir + 'best_model.pkl')
            if epoch% checkpoint_interval == 0:
                save_dict = {'model': model.state_dict(), 'epoch': epoch,
                             'best_loss': test_loss,
                             'pos_acc': apos, 'neg_acc': aneg, 'combine': aall,
                             'precision': prec, 'recall': recall,
                             'pos_weight': cfg.getfloat('loss', 'pos_weight')}
                torch.save(save_dict,exp_dir + '{}.pkl'.format(epoch))
        save_dict = {'model': model.state_dict(), 'epoch': epoch,
                     'best_loss': test_loss,
                     'pos_acc': apos, 'neg_acc': aneg, 'combine': aall,
                     'precision': prec, 'recall': recall,
                     'pos_weight': cfg.getfloat('loss', 'pos_weight')}
        torch.save(save_dict, exp_dir + 'last_model.pkl')
    else:
        for epoch in range(epochs):
            train_loss,train_precision,train_recall = train(model, optimizer, train_loader, device, epoch, verbose)
            test_loss, apos, aneg, aall,prec,recall = eval(model, test_loader, device, verbose)
            row = [train_loss, test_loss, apos, aneg, aall,prec,recall,train_precision,train_recall]
            scheduler.step()
            rows.append(row)

    columns = ['train_loss', 'test_loss', 'pos_accuracy', 'neg_accuracy', 'accuracy','precision','recall','train_precision','train_recall',]
    return pd.DataFrame(rows, columns=columns)




