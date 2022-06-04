# @Time    : 2022/5/19 10:58
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : utilss
# @Project Name :code

import pickle
import numpy as np
from Hyperparameter import Hyperparams as hp
import random

#read_pkl
def read_pkl(datas_path):
    with open(datas_path,'rb') as f:
        return pickle.load(f)
#conver datas type to array
def toarry(datas):
    if isinstance(datas,list):
        return np.array(datas)
    else:
        return datas

#cal error for 3,6,9,12 steps
def cal_multi_step_errot(logit,label):
    error=np.abs(logit-label)
    erro_step = [error[i] for i in range(len(error)) if (i+1)%3==0]
    return toarry(erro_step)#->(N,4)

#mask a single sequanece 
def single_mask(data,mask_idx,mask_num):
    radom_idx = random.sample(list(mask_idx),mask_num)
    for i in radom_idx:
        data[i,:]=0
    return data