# @Time    : 2022/5/19 10:49
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : gen_batch
# @Project Name :code

from utilss import *


def gen_batch(datas,batch_size):
    """

    Args:
        datas: only for train data
        batch_size: number of samples per training steps

    Returns:a yield,each iter with shape(batch_szie,time_seq)

    """
    len_datas = len(datas)
    idx = np.arange(len_datas)#to gen a idx series
    np.random.shuffle(idx)
    datas = toarry(datas)
    for i in range(0,len_datas,batch_size):
        start_idx = i
        end_idx = i+batch_size
        if end_idx>len_datas:
            end_idx = len_datas
        slc = idx[start_idx:end_idx]
        yield datas[slc]


def inverse_sample(datas,max_,min_):
    """

    Args:
        datas:suit for val data and test data
        u:data mean
        std:data std

    Returns:only label inverse,a arrry with shape((all_len,time_seq))

    """
    for i in range(len(datas)):
        datas[i][hp.input_max_len:hp.input_max_len+hp.output_max_len]=datas[i][hp.input_max_len:hp.input_max_len+hp.output_max_len]*(max_-min_)+min_
    return toarry(datas)