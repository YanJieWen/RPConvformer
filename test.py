# @Time    : 2022/5/19 22:27
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : test
# @Project Name :code
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle
from Hyperparameter import Hyperparams as hp
from utilss import *
from gen_batch import *
import pandas as pd
#get data
datas = read_pkl(hp.pkl_path_08)
test_data = datas[2]#test data
test_data = np.array(test_data)
scalar_ = datas[-1]
#load graph and weight from ckpt
test_sess = tf.Session()
saver = tf.train.import_meta_graph('./ckpt/weight-100.meta')
graph = tf.get_default_graph()
saver.restore(test_sess, tf.train.latest_checkpoint('ckpt'))
if hp.if_rm:
    mask_num = int(hp.input_max_len*hp.mask_rate)
    mask_idx =  np.arange(hp.input_max_len)
    test_data = np.array([single_mask(data,mask_idx,mask_num) for data in test_data])
#Define value
input_datas = test_data[2000:,:hp.maxlen]
output_datas = test_data[2000:,hp.maxlen:]
decoder_output = np.zeros_like(output_datas)
#define placehoder from graph
x_input = graph.get_tensor_by_name('Placeholder:0')
decoder_input = graph.get_tensor_by_name('Placeholder_1:0')
y_hat = graph.get_tensor_by_name('prediction/y_hat/Einsum:0')
#autoregression
for j in range(hp.outputlen):
    _pred = test_sess.run(y_hat, feed_dict={x_input:input_datas,decoder_input:decoder_output})
    decoder_output[:,j] = _pred[:, j]
# decoder_output = decoder_output*(max_-min_)+min_
decoder_output = scalar_.inverse_transform(np.squeeze(decoder_output))
output_datas = scalar_.inverse_transform(np.squeeze(output_datas))
df_ = pd.DataFrame(decoder_output)
df_.to_excel('./2000.xlsx')
all_ = list(map(cal_multi_step_errot,decoder_output,output_datas))
all_ = toarry(all_)
for i in range(all_.shape[  -1]):
    print(np.mean(all_[:,i]))#print interval 3 steps error-> 3,6,9,12