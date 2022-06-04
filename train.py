# @Time    : 2022/5/19 14:28
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : train
# @Project Name :code
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from framework import Ts_transformer
from Hyperparameter import Hyperparams as hp
from moudels import *
import time
from gen_batch import *
import pandas as pd
from utilss import *
from sklearn.preprocessing import MinMaxScaler

#Get train datas&val datas
datas = read_pkl(hp.pkl_path_08)
train_data =datas[0]
val_data = datas[1]
scalar_ = datas[-1]
val_data = toarry(val_data)[100:500]#origin val is list need to be arry
predictions = np.zeros_like(val_data[:,hp.maxlen:(hp.maxlen+hp.outputlen)])
#Define tf.placeholder
xs = tf.placeholder(dtype=tf.float32,shape=[None,hp.maxlen,hp.output_units])
ys = tf.placeholder(dtype=tf.float32,shape=[None,hp.outputlen,hp.output_units])
#Define train model
m = Ts_transformer(xs,ys)
memory,src_masks = m.encode()
y_hat = m.decode(memory,src_masks)
#Define val model
if hp.if_validation:
    memory_, src_masks_ = m.encode(if_training=False)
    pred = m.decode(memory_, src_masks_,if_training=False)
#Define loss
loss = tf.losses.mean_squared_error(ys,y_hat)
#train scheme->glob_step,lr,optimization,train_op
global_step = tf.train.get_or_create_global_step()#Create a global variable to record the number of steps
lr = tf.train.exponential_decay(hp.lr, global_step ,
decay_steps=5 * (len(train_data)//hp.batch_size), decay_rate=0.7, staircase=True)
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss, global_step=global_step)
tf.summary.scalar('lr', lr)
tf.summary.scalar("loss", loss)
tf.summary.scalar("global_step", global_step)
summaries = tf.summary.merge_all()
#start Session
saver = tf.train.Saver(max_to_keep=3)
with tf.Session() as sess:
    writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(hp.num_epochs):
        for j, inputs_ in enumerate(gen_batch(train_data, hp.batch_size)):
            _, _gs, summary, loss_= sess.run([train_op, global_step, summaries, loss],
                                                   feed_dict={xs: inputs_[:, 0:hp.maxlen],
                                                              ys: inputs_[:, hp.maxlen:(hp.maxlen + hp.outputlen)]})
            writer.add_summary(summary, _gs)
            if j%20==0:
                # autoregression for each step
                if hp.if_validation:#set value is False to avoid
                    for id_ in range(hp.outputlen):
                        _predictions = sess.run(pred, feed_dict={xs: val_data[:,0:hp.maxlen], ys: predictions})
                        predictions[:,id_] = _predictions[:, id_]#keep dims and brodcast
                    predict = scalar_.inverse_transform(np.squeeze(predictions))
                    gt = scalar_.inverse_transform(np.squeeze(val_data[:,hp.maxlen:]))#inverse label to origin value
                    mae_ = cal_multi_step_errot(predict,gt)#(N,4)
                    ade = np.mean(mae_)
                    print('The validation MAE value is {:.2f}, for the {} steps in {} epochs.'.format(ade,j,i))
                    if ade<=hp.valid_thresh:
                        hp.valid_thresh=ade
                        saver.save(sess=sess, save_path=hp.ckpt_path, global_step=(i + 1))
                else:
                    if i%10==0 and j%(20*3)==0 or i==hp.num_epochs-1:
                        saver.save(sess=sess, save_path=hp.ckpt_path, global_step=(i + 1))
                print('The current loss value is {:.3f}, for the {} steps in {} epochs.'.format(loss_,j,i))
    sess.close()

