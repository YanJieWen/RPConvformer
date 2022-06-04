# @Time    : 2022/5/23 23:34
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : LSTM
# @Project Name :code

from utilss_ import *
from parameters import Hyperparams as hp
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed
import pandas as pd
import os
os.chdir('./baselinemodel')
#build data
datas = read_data(hp.pkl_path_08)
#train data
train_x = np.array(datas[0])[:,:hp.maxlen]#12 steps
train_y = np.array(datas[0])[:,hp.output_max_len:]
#test data
test_data = np.array(datas[2])
sca = datas[-1]
test_label = sca.inverse_transform(np.squeeze(test_data[:,hp.maxlen:]))
test_input = test_data[:,:hp.output_max_len]
#train model
model = Sequential()
#if LSTM
# model.add(LSTM(10, input_shape=(12,1)))
#if GRU
# model.add(GRU(10, input_shape=(12,1)))
# if bilstm
model.add(Bidirectional(LSTM(10,return_sequences=True),input_shape=(12,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=0)
#autoregression
predicts = model.predict(test_input)
predicts = sca.inverse_transform(np.squeeze(predicts))
#eval
all=print_muti_error(predicts,test_label)
df = pd.DataFrame(np.squeeze((all)).T)
df.to_excel('./each_step_metrics_pems08/bilstm.xlsx')#need to change when model has been changed
#draw single pc
sing_pred = predicts[:,0]
sing_label = test_label[:,0]
draw_gt_pred(sing_label,sing_pred)#global visual
draw_gt_pred(sing_label[288:288*2],sing_pred[288:288*2])#local visual