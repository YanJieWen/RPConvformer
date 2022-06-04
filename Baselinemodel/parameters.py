# @Time    : 2022/5/22 18:04
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : parameters
# @Project Name :code

class Hyperparams:
    '''Hyperparameters'''
    # data
    original_path_04 = '../data/PeMS04.npz'
    original_path_08 = '../data/PeMS08.npz'
    pkl_path_04  = '../pems04.pkl'
    pkl_path_08 = '../pems08.pkl'
    input_max_len = 12
    output_max_len = 12

    # training params
    batch_size = 32# alias = N
    lr = 0.001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory
    ckpt_path = './ckpt/weight'
    # model
    maxlen = input_max_len  # Maximum number of words in a sentence. alias = T.
    outputlen = output_max_len
    hidden_units = 512  # alias = C
    hd_ff = 4*hidden_units #inner cell number
    output_units = 1 #output units number
    num_blocks = 1  # number of encoder/decoder blocks
    num_epochs = 200
    num_heads = 8
    dropout_rate = 0.1
    if_validation =False
    valid_thresh = 100#if > then keep,else alter