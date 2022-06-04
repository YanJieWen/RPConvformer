# @Time    : 2022/5/19 14:06
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : framework
# @Project Name :code


from pickle import TRUE
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from Hyperparameter import Hyperparams as hp
from moudels import *

class Ts_transformer():
    def __init__(self,xs,ys):
        """
        :param xs: a tensor embedding data->(batch_size,encoder_len,features)
        :param ys: a tensor ground truth->(batch_size,decoder_len,features)
        :param is_training: a bool value
        """
        self.xs = xs
        self.ys = ys
    def encode(self,if_training=True):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            #embedding
            enc = embedding_op(self.xs,num_units=hp.hidden_units,scaled=True,if_casual=True)#no embedding
            src_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(enc), axis=-1)),
                                       -1)  # padding mask->(batch_size,T_q,1)
            #positinal embedding
            enc+=positional_embedding(enc,scaled=True)#no position

            enc *= src_masks#broadcast,get paddding==0
            #Dropout
            enc = tf.layers.dropout(enc, hp.dropout_rate, training=if_training)
            #blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    #self-attention
                    enc = multihead_attention(queries=enc,keys=enc,values=enc,scope='self-attention',
                                              key_mask=src_masks,num_heads=hp.num_heads,
                                              drop_rate=hp.dropout_rate,if_causality=False,if_training=if_training
                                              )
                    #feed_forward
                    enc = feed_forward(enc,num_units=[hp.hd_ff,hp.hidden_units])
        enc_memory = enc
        return enc_memory,src_masks#[(batch_size,encoder_len,num_units),(batch_size,encoder_len,1)]

    def decode(self,memory,src_masks,if_training=True):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            decoder_inputs = tf.concat((tf.ones_like(self.ys[:, :1, :]) * 2, self.ys[:, :-1, :]), 1)# shift left 1 and padding with 2:start
            #embedding
            dec = embedding_op(decoder_inputs,num_units=hp.hidden_units,scaled=True,if_casual=True) #->(N,decoder_len,num_units)
            tgt_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(dec), axis=-1)), -1)
            #The sequence of label is fixed
            dec += positional_embedding(dec, scaled=True)
            dec*=tgt_masks#broadcast,get paddding==0
            # Dropout
            dec = tf.layers.dropout(dec, hp.dropout_rate, training=if_training)
            # blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    #self_attention&seq_padding&no paddiing mask because decoder inputs is fiexed len,there is no need for keymasking
                    #if need mask you should use tgt_mask
                    dec = multihead_attention(queries=dec, keys=dec, values=dec,scope='self-attention',
                                              key_mask=tgt_masks,if_mask=False,
                                              num_heads=hp.num_heads,
                                              drop_rate=hp.dropout_rate, if_causality=True,
                                              if_training=if_training)
                    #Interactive self-attiontion&only key mask,you should mask the part of padding from encder by using src_mask
                    dec = multihead_attention(queries=dec, keys=memory, values=memory,scope='interactive-attention',
                                              if_mask=True,key_mask=src_masks,num_heads=hp.num_heads,
                                              drop_rate=hp.dropout_rate, if_causality=False,
                                              if_training=if_training
                                              )
                    #feed_forward
                    dec = feed_forward(dec, num_units=[hp.hd_ff,hp.hidden_units])#the decode output->(batch_size,decode_seq,num_units)
        # liner output
        with tf.variable_scope('prediction', reuse=tf.AUTO_REUSE):
            out_variabel = tf.get_variable('out_weight',shape=[hp.hidden_units,hp.output_units],
                                           initializer=tf.random_normal_initializer(),dtype=tf.float32)
            y_hat = tf.einsum('ntd,dk->ntk', dec, out_variabel,name='y_hat')
        return y_hat
