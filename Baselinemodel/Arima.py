# @Time    : 2022/5/23 9:15
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : Arima
# @Project Name :code

import numpy as np
import pickle
import matplotlib.pyplot as plt
#arima package for time series
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf#auto darw pacf&acf
from parameters import Hyperparams as hp
from utilss_ import *
import pandas as pd
import os
os.chdir('./baselinemodel')

def diff_anlysis(datas):
    """

    Args:
        datas: train_data with one dim

    Returns:dataframe

    """
    df = pd.DataFrame(datas)
    df['diff_1'] = df[0].diff(1)  # one stage diff
    df['diff_2'] = df['diff_1'].diff(1)  # two stage diff
    df.plot(subplots=True, figsize=(12, 8))  # based pandas column
    plt.show()
    return df

def acf_pacf(df,lag):
    """

    Args:
        df: a dataframe
        lag: a value lag

    Returns:draw a pc

    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df[0], lags=lag, ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()

    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df[0], lags=lag, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.show()

def corr_scattar(df):
    lags = 9
    ncol = 3
    nrows = int(np.ceil(lags // ncol))  # 计算大于等于该值的最小整数
    fig, axes = plt.subplots(ncols=ncol, nrows=nrows, figsize=(4 * ncol, 4 * nrows))
    for ax, lag in zip(axes.flat, np.arange(1, lags + 1, 1)):
        x = (pd.concat([df[0], df[0].shift(-lag)], axis=1, keys=['y'] + [lag]).dropna())
        x.plot(ax=ax, kind='scatter', y='y', x=lag)
        corr = x.corr().values[0][1]
        ax.set_title('corr is {:.2f}'.format(corr))
        sns.despine()
    fig.tight_layout()
    plt.show()
def build_model(scarlr_,test_data):
    """
    It will spend more time.......

    Args:
        test_data: test data
        scarlr_: to inverse data

    Returns:predictions

    """
    test_data_ = scarlr_.inverse_transform(np.squeeze(test_data))
    predict = []
    for i in range(test_data_.shape[0]):
        model = sm.tsa.arima.ARIMA(test_data_[i, :hp.maxlen], order=(1, 0, 1))  # p,d,q
        arima_res = model.fit()
        predict.append(arima_res.forecast(12))
    return np.squeeze(predict)

def main():
    datas = read_data(hp.pkl_path_08)
    train_datas = datas[0]
    test_datas = datas[2]
    scarlr_ = datas[-1]
    train_data = scarlr_.inverse_transform(np.squeeze(train_datas))[:, 0]  # anlysis datas
    df = diff_anlysis(train_data)
    acf_pacf(df,25)
    corr_scattar(df)
    prediction = build_model(scarlr_,test_datas)
    test_label = scarlr_.inverse_transform(np.squeeze(test_datas)[:,-12:])
    predict = np.squeeze(prediction)
    #muti error
    all = print_muti_error(predict,test_label)
    df = pd.DataFrame(np.squeeze((all)).T)
    df.to_excel('./each_step_metrics_pems08/Arima.xlsx')
    #single draw
    test_label_ = test_label[:, 0]
    predict_ = predict[:, 0]
    draw_gt_pred(test_label_,predict_)
    draw_gt_pred(test_label_[288:288*2], predict_[288:288*2])
if __name__ == '__main__':
    main()