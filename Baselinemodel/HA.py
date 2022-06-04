# @Time    : 2022/5/22 19:04
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : HA
# @Project Name :code
import numpy as np

from utilss_ import *
from parameters import Hyperparams as hp
import pandas as pd
import os
os.chdir('./baselinemodel')

def history_avg_model(datas,label_time_step,pred_time_step):
    """

    Args:
        datas: a data with input seq+output seq
        label_time_step: input time step,for the Ha,short step is advised
        pred_time_step: output time step

    Returns:

    """
    y_hat = []
    datas_ = datas.copy()#avoid no real  datas
    for i in range(pred_time_step):
        pred = np.mean(datas_[:label_time_step])
        y_hat.append(pred)
        datas_[label_time_step] = pred
        datas_[0:label_time_step]=datas_[1:(label_time_step+1)]
    return y_hat

def main():
    datas = read_data(hp.pkl_path_08)
    test_data = datas[2]
    scarlar_ = datas[-1]
    test_data = inverse_data(test_data, scarlar_)
    #cal mul_ti step error
    test_label = test_data[:,-hp.outputlen:]
    y_hat = np.array([history_avg_model(sample,12,12) for sample in test_data])
    y_hat = y_hat[:,-hp.outputlen:]
    #each error to excel
    all = print_muti_error(y_hat,test_label)
    df = pd.DataFrame(np.squeeze((all)).T)
    df.to_excel('./each_step_metrics_pems08/HA.xlsx')
    #single draw
    draw_gt_pred(test_label[:,0],y_hat[:,0])
    draw_gt_pred(test_label[:, 0][288:288*2], y_hat[:, 0][288:288*2])


if __name__ == '__main__':
    main()