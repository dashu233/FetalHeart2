from configparser import ConfigParser
import ast
import pickle


class MyConfigParser(ConfigParser):
    def getliststr(self,section,option):
        return ast.literal_eval(self.get(section, option))
    def getlistint(self,section,option):
        return [int(x) for x in ast.literal_eval(self.get(section, option))]
    def getlistfloat(self,section,option):
        return [float(x) for x in ast.literal_eval(self.get(section, option))]


cfg = MyConfigParser()

cfg.read("binary_pred.conf")
if cfg.getboolean('data','default'):


    if cfg.get('data','data_dir') == 'dataset_remain' or 'dataset':
        eval_fold = cfg.getint('data','eval_fold')
        test_fold = cfg.getint('data','test_fold')
        train_fold = [i for i in range(5)]
        train_fold.remove(eval_fold)
        train_fold.remove(test_fold)
        print('train_fold:',train_fold)
        with open('fold_list.pkl', 'rb') as f:
            fold_list = pickle.load(f)
        test_start = []
        test_end = []
        for i in range(3):
            test_start.append(fold_list['class{}_start'.format(i+1)][test_fold])
            test_end.append(fold_list['class{}_end'.format(i + 1)][test_fold])
        eval_start = []
        eval_end = []
        for i in range(3):
            eval_start.append(fold_list['class{}_start'.format(i + 1)][eval_fold])
            eval_end.append(fold_list['class{}_end'.format(i + 1)][eval_fold])
        train_start = []
        train_end = []
        for i in range(3):
            for j in train_fold:
                train_start.append(fold_list['class{}_start'.format(i + 1)][j])
                train_end.append(fold_list['class{}_end'.format(i + 1)][j])
        #print('create_train_start')
        cfg['data']['train_start'] = str(train_start)
        cfg['data']['train_end'] = str(train_end)
        cfg['data']['eval_start'] = str(eval_start)
        cfg['data']['eval_end'] = str(eval_end)
        cfg['data']['test_start'] = str(test_start)
        cfg['data']['test_end'] = str(test_end)



if __name__ == '__main__':
    tst = cfg.getlistfloat('train','check')
    for t in tst:
        print(t)
        print(type(t))