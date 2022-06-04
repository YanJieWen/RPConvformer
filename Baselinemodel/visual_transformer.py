# @Time    : 2022/5/22 20:06
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : visual_transformer
# @Project Name :code

import pandas as pd
from parameters import Hyperparams as hp
from utilss_ import *
import os
os.chdir('./baselinemodel')
"""
only to visualization pred and gt
"""
df = pd.read_excel('./0.20.xlsx')
pred = df.iloc[:,1:].values
datas = read_data(hp.pkl_path_08)
test_data = datas[2]
scarlar_ = datas[-1]
test_data = inverse_data(test_data, scarlar_)
test_label = test_data[:,-hp.outputlen:]
#mutil_error
all = print_muti_error(pred,test_label)
#each error to excel
df = pd.DataFrame(np.squeeze((all)).T)
df.to_excel('./each_step_metrics_pems08/0.20.xlsx')
#draw single
draw_gt_pred(test_label[:,0],pred[:,0])
draw_gt_pred(test_label[:,0][288:288*2],pred[:,0][288:288*2])
# draw_gt_pred(test_label[288:288*2][84:120],pred[288:288*2][84:120])