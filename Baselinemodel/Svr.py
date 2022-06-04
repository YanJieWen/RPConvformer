# @Time    : 2022/5/23 11:15
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : Svr
# @Project Name :code
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from utilss_ import *
from parameters import Hyperparams as hp
import pandas as pd
from utilss_ import *
import os
os.chdir('./baselinemodel')
datas = read_data(hp.pkl_path_08)
train_data = datas[0]
test_data = datas[2]
sac = datas[-1]
test_label = sac.inverse_transform(np.squeeze(test_data)[:,-12:])
train_data = np.squeeze(train_data)
#build_model,girdsearch may use more time.....
svr = svm.SVR(kernel='rbf')
# c_can = np.logspace(-2, 2, 10)
# gamma_can = np.logspace(-2, 2, 10)
# svr = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
svr.fit(train_data[:,:hp.maxlen],train_data[:,hp.maxlen])
#cal muti steps predict
test_data = np.squeeze(test_data)
test_data_ = test_data.copy()
predicts = []
for i in range(test_data_.shape[0]):
    samples = []
    for j in range(12):
        y_hat = svr.predict(test_data_[i,None,:hp.maxlen])
        samples.append(y_hat)
        test_data_[i,hp.maxlen]=y_hat
        test_data_[i,0:hp.maxlen]=test_data_[i,0+1:hp.maxlen+1]
    predicts.append(samples)
predicts= sac.inverse_transform(np.squeeze(predicts))#->(N,12) after inverse
all = print_muti_error(predicts,test_label)
#each error to excel
df = pd.DataFrame(np.squeeze((all)).T)
df.to_excel('./each_step_metrics_pems08/SVR.xlsx')
#singel predict faltten pred
sing_pred = predicts[:,0]
sing_label = test_label[:,0]
draw_gt_pred(sing_label,sing_pred)
draw_gt_pred(sing_label[288:288*2],sing_pred[288:288*2])
