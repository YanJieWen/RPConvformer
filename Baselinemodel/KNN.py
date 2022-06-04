# @Time    : 2022/5/23 15:54
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : KNN
# @Project Name :code
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from utilss_ import *
from parameters import Hyperparams as hp
import os
os.chdir('./baselinemodel')
datas = read_data(hp.pkl_path_08)
#build data
train_data = np.array(datas[0])
test_data = np.array(datas[2])
sac = datas[-1]
test_label = sac.inverse_transform(np.squeeze(test_data)[:,-12:])
#train model
train_data = np.squeeze(train_data)
knn = KNeighborsRegressor(5)
knn.fit(train_data[:,:hp.maxlen], train_data[:,hp.maxlen])
#autoregression model
test_data = np.squeeze(test_data)
test_data_ = test_data.copy()
predicts = []
for i in range(test_data_.shape[0]):
    samples = []
    for j in range(12):
        y_hat = knn.predict(test_data_[i,None,0:hp.maxlen])
        samples.append(y_hat)
        test_data_[i,hp.maxlen]=y_hat
        test_data_[i,0:hp.maxlen]=test_data_[i,0+1:hp.maxlen+1]
    predicts.append(samples)
predicts = np.squeeze(predicts)
predicts= sac.inverse_transform(predicts)
#eval
all=print_muti_error(predicts,test_label)
df = pd.DataFrame(np.squeeze((all)).T)
df.to_excel('./each_step_metrics_pems08/KNN.xlsx')
#draw single pc
sing_pred = predicts[:,0]
sing_label = test_label[:,0]
draw_gt_pred(sing_label,sing_pred)#global visual
draw_gt_pred(sing_label[288:288*2],sing_pred[288:288*2])#local visual
