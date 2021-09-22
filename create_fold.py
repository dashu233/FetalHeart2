import numpy as np

patient_label = np.load('patient_label.npy').transpose()
patient_remain = np.load('patient_list.npy').transpose()

class_label = 0
class_num = 0
class_list = [[],[],[]]
for i in range(len(patient_label)):
    if not patient_label[i] == class_label:
        class_label = patient_label[i]
        class_num += 1
    #print(class_label)
    class_list[class_num].append(i)

#print(class_list)

class_lenth = [sum([patient_remain[j][1] for j in class_list[i]])-
               sum([patient_remain[j][0] for j in class_list[i]]) for i in range(3)]
print('class_lenth:',class_lenth)

# 8:1:1
fold_len = [_//10 for _ in class_lenth]

# fold_list: [3,10,2], 3:class1,class2,class3, 10:10 fold,2:start/end
fold_list = [[[0,0] for j in range(10)] for i in range(3)]
tick = 0
for class_num in range(3):
    fd = 0
    overload = 0
    while fd<9:
        fold_list[class_num][fd][0] = patient_remain[tick][0]
        fold_start = patient_remain[tick][0]
        now_lenth = 0
        #print('start_tick:',patient_remain[tick][0])
        while now_lenth + patient_remain[tick][1]-patient_remain[tick][0]< fold_len[class_num]:
            now_lenth += patient_remain[tick][1]-patient_remain[tick][0]
            tick += 1
        #print('end_tick:',patient_remain[tick-1][1])
        if overload > 0:
            fold_list[class_num][fd][1] = patient_remain[tick-1][1]
            overload -= (fold_len[class_num] - now_lenth)
        else:
            fold_list[class_num][fd][1] = patient_remain[tick][1]
            overload += (now_lenth + patient_remain[tick][1] - patient_remain[tick][0] - fold_len[class_num])
            tick += 1
        fd += 1
    fold_list[class_num][fd][0] = patient_remain[tick][0]
    fold_list[class_num][fd][1] = patient_remain[class_list[class_num][-1]][1]
    tick = class_list[class_num][-1]
    tick += 1
#print(np.array(fold_list[1]).transpose()[0])
class1_start_tick = np.array(fold_list[0]).transpose()[0]
class1_end_tick = np.array(fold_list[0]).transpose()[1]
class2_start_tick = np.array(fold_list[1]).transpose()[0]
class2_end_tick = np.array(fold_list[1]).transpose()[1]
class3_start_tick = np.array(fold_list[2]).transpose()[0]
class3_end_tick = np.array(fold_list[2]).transpose()[1]
import pickle
obj = {'class1_start':class1_start_tick,'class2_start':class2_start_tick,'class3_start':class3_start_tick,
         'class1_end':class1_end_tick,'class2_end':class2_end_tick,'class3_end':class3_end_tick}
with open("fold_list.pkl", "wb") as fp:
    pickle.dump(obj, fp)


