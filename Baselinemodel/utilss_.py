# @Time    : 2022/5/22 18:04
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : utilss_
# @Project Name :code

import pickle
import numpy as np
from parameters import Hyperparams as hp
import matplotlib.pyplot as plt

def read_data(data_path):
    with open(data_path,'rb') as f:
        return pickle.load(f)

def inverse_data(datas,scarla_):
    return scarla_.inverse_transform(np.squeeze(datas))

def draw_gt_pred(gt,pred):
    """

    Args:
        gt: ground truth
        pred: predictions

    Returns:a pc between gt and pred

    """
    plt.figure(figsize=(20,8),dpi=120)
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    # plt.xticks(np.arange(0,len(gt),288))#per day to visualization
    plt.tick_params(labelsize=18)
    plt.xlim(0,len(gt)+4)
    plt.ylim(0,max(np.max(gt),np.max(pred))+50)
    plt.plot(gt,'-',color='deepskyblue',linewidth=1)
    plt.plot(pred, '-.',color='orangered',linewidth=1.5)
    plt.show()

def print_muti_error(pred,test_label):
    """

    Args:
        pred: a array with shape->(N,12)
        test_label: the same with pred

    Returns:draw a pc and each metrics in each step

    """
    all = []
    muti_er = np.mean(np.abs(pred - test_label), axis=0)
    all.append(muti_er)
    muti_er = muti_er[[2, 5, 8, 11]]
    muti_er_mse = np.mean(np.square(np.abs(pred - test_label)), axis=0)
    all.append(muti_er_mse)
    muti_er_mse = muti_er_mse[[2, 5, 8, 11]]
    muti_er_mape = np.mean(np.abs(pred - test_label) / test_label, axis=0)
    all.append(muti_er_mape)
    muti_er_mape = muti_er_mape [[2, 5, 8, 11]]
    print('MAE is : The 3th {:.2f},6th {:.2f},9th {:.2f},12th {:.2f}'.format(muti_er[0], muti_er[1],
                                                                             muti_er[2], muti_er[3]))
    print('MSE is : The 3th {:.2f},6th {:.2f},9th {:.2f},12th {:.2f}'.format(muti_er_mse[0], muti_er_mse[1],
                                                                             muti_er_mse[2],
                                                                             muti_er_mse[3]))
    print('MAPE is : The 3th {:.4f},6th {:.4f},9th {:.4f},12th {:.4f}'.format(muti_er_mape[0], muti_er_mape[1],
                                                                              muti_er_mape[2],
                                                                              muti_er_mape[3]))
    print('mae is:', np.mean(np.abs(pred - test_label)))
    print('mse is:', np.mean(np.square(np.abs(pred - test_label))))
    print('mape is:', np.mean(np.abs(pred - test_label) / test_label))
    return all


