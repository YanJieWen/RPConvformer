# @Time    : 2022/5/19 10:00
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : data_store
# @Project Name :code

import numpy as np
import pickle
from Hyperparameter import Hyperparams as hp
from sklearn.preprocessing import MinMaxScaler



def get_singel_data(datas, input_max_len, output_max_len):
    """

    Args:
        datas: train data,val data and test data with the shape 1dims
        input_max_len: the max len of input value
        output_max_len: the max len of output value

    Returns:[[input+output],input+output]...]

    """
    datas_ = []
    for i in range(len(datas) - input_max_len - output_max_len):
        datas_.append(datas[i:i + input_max_len + output_max_len])
    return datas_
#get data as list type
def get_datas(flows,node_idx,featrue_idx,train_split,val_split):
    """

    Args:
        flows: pe_04 or pe_08
        node_idx:the index of detecors;for pe_04 is 307 in total and for pe_08 is 170 for total
        featrue_idx:three features:flow,occupancy and speed
        train_split:idx of list,stands how many days
        val_split:idx of list,stands how many days

    Returns:[[train_data],[val_data],[test_data],(u,std)]

    """
    datas_04 = flows['data'][:,node_idx,featrue_idx]
    train_split_ = train_split*288
    val_split = (train_split+val_split)*288
    #list store
    scaler = MinMaxScaler(feature_range=(-1, 1))
    datas_04_ = scaler.fit_transform(datas_04.reshape((-1, hp.output_units)))
    datas_04_train = datas_04_[:train_split_]
    datas_04_val = datas_04_[train_split_:val_split]
    datas_04_test = datas_04_[val_split:]
    datas = [datas_04_train,datas_04_val,datas_04_test]
    datas = [get_singel_data(data,hp.input_max_len,hp.output_max_len) for data in datas]
    #store (u,sig) to the list put into the last index of list
    datas.append(scaler)
    return datas
#store data as pkl
def to_pikle(datas,data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(datas, f)

def main():
    # load data as npz type
    pe_04 = np.load(hp.original_path_04)
    pe_08 = np.load(hp.original_path_08)
    datas_04 = get_datas(pe_04,0,0,44,5)
    datas_08 = get_datas(pe_08, 7, 0, 47, 5)
    to_pikle(datas_04,hp.pkl_path_04)
    to_pikle(datas_08, hp.pkl_path_08)
if __name__ == '__main__':
    main()

