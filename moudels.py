# @Time    : 2022/5/19 14:03
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : moudels
# @Project Name :code


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def embedding_op(inputs,num_units,scaled=True,if_casual=True):
    """

    :param inputs: a tensor->(batch_size,seq_len,features)
    :param num_units: the number of the units cell
    :param scaled: /dk**(1/2),a bool value
    :return:(batch_size,seq_len,features)
    """
    with tf.variable_scope('embedding',reuse=tf.AUTO_REUSE):
        if if_casual==True:
            # outputs = tf.layers.dense(inputs,num_units,activation=tf.nn.sigmoid)#one layer to get tensor->(batch_size,encode_len,num_units)
            outputs_1 = tf.layers.conv1d(inputs, filters=num_units//4, kernel_size=1)
            outputs_1 = tf.layers.conv1d(outputs_1, filters=num_units//2, kernel_size=1)
            outputs_1 = tf.layers.conv1d(outputs_1, filters=num_units, kernel_size=1)
            outputs_3 = tf.layers.conv1d(outputs_1, filters=num_units, kernel_size=9,padding='causal')
            outputs_5 = tf.layers.conv1d(outputs_1, filters=num_units, kernel_size=9,padding='causal')
            outputs_7 = tf.layers.conv1d(outputs_1, filters=num_units, kernel_size=9,padding='causal')
            outputs = outputs_3+outputs_5+outputs_7
        else:
            outputs = tf.layers.conv1d(inputs, filters=num_units//4, kernel_size=1)
            outputs = tf.layers.conv1d(outputs, filters=num_units //2, kernel_size=1)
            outputs = tf.layers.conv1d(outputs, filters=num_units, kernel_size=1)
        if scaled:
            outputs = outputs * (num_units ** 0.5)
        return outputs

def positional_embedding(inputs,zero_pad=True,scaled=True):
    """

    :param inputs: after embedding->(batch_size,seq_len,num_units)
    :param scaled:a bool value,whther to /np.sqrt(num_units)
    :return:(batch_size,seq_len,features)
    """
    # N, T ,num_units = inputs.get_shape().as_list()
    N = tf.shape(inputs)[0]
    batch_size, T ,num_units = inputs.get_shape().as_list()
    # num_units = hp.hidden_units
    with tf.variable_scope('positinal_embedding',reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N,1])#generate a sequence tuple like (0,1,2,3...)->(N,T)
        # Sinusoidal Positional_Encoding
        # PE = np.array([[pos/np.power(10000,2.*i/num_units) for i in range(num_units)] for pos in range(T)])
        PE = np.array([
            [pos / np.power(10000, (i-i%2)/num_units) for i in range(num_units)]
            for pos in range(T)])  
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        lookup_table = tf.convert_to_tensor(PE,dtype=tf.float32)#(seq_len,num_units),no train
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        if scaled:
            outputs = outputs * num_units ** 0.5
        outputs = tf.cast(outputs,'float32')#positinal embedding may be to float64
    return outputs

def multihead_attention(queries,keys,key_mask,values,num_heads,scope,
                        drop_rate,if_causality=False,if_training=True,if_mask=True):
    """

    :param queries:a tensor->([N, T_q, num_units].)
    :param keys:a tensor->([N, T_k, num_units].)
    :param values:a tensor->([N, T_k, num_units].)
    :param key_masks:A 2d tensor with shape of [N, key_seqlen]
    :param num_heads:An int. Number of heads.When value=1,is self_atttion
    :param drop_rate:A floating point number.
    :param if_mask:Boolean.If true, mask op will be done
    :param if_causality:Boolean. If true, units that reference the future are masked
    :return:a tensor with(N,T_q,d_model)
    """
    d_model = queries.get_shape().as_list()[-1]#int num_units
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections,RPE
        Q = tf.layers.dense(queries, d_model, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True) # (N, T_k, d_model),learn to rpe
        V = tf.layers.dense(values, d_model, use_bias=True) # (N, T_k, d_model),learn to rpe

        # Split and concat,multi_head,RPE
        Q_= tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_= tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        #get attention score->dot_product
        with tf.variable_scope('scaled_dot_product_attention',reuse=tf.AUTO_REUSE):
            d_k = Q_.get_shape().as_list()[-1]
            #dot product
            outputs = tf.matmul(Q_,tf.transpose(K_, [0, 2, 1]))# (h*N, T_q, T_k)
            #scale
            outputs /= d_k ** 0.5
            #mask，the kernel tech in the transformer,including padding mask(encodeing processing)&sequence mask(decoding processing inputs)
            padding_num = -2 ** 32 + 1#an inf
            if if_mask:#padding masking
                key_masks = tf.to_float(key_mask) # (N, T_k,1)
                key_masks = tf.transpose(key_masks,[0,2,1])# (N, 1,T_k)
                key_masks = tf.tile(key_masks, [num_heads, tf.shape(queries)[1],1]) # (h*N, T_q, T_k)
                # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
            elif if_causality:#sequence masking
                diag_vals = tf.ones_like(outputs[0, :, :])#generate a matrix->(seq_len,seq_len),filled 1
                tril = tf.linalg.band_part(diag_vals, -1, 0)#Upper triangular matrix
                future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])#for each heads*batch satisfied with matrix
                paddings = tf.ones_like(future_masks) * padding_num
                outputs = tf.where(tf.equal(future_masks, 0), paddings, outputs)

            #softmax
            outputs = tf.nn.softmax(outputs)#->(h*batch_size,seq_len,seq_len)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))#save the first feature image

            #dropout
            outputs = tf.layers.dropout(outputs, rate=drop_rate, training=if_training)
            #weight_sum->(head*batch,seq_len,num_units/heads)
            outputs = tf.matmul(outputs, V_)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, num_units)
        # Residual connection
        outputs += queries#->(N, T_q, num_units)
        # Normalize
        outputs = ln(outputs)
    return outputs


def feed_forward(inputs,num_units):
    """

    :param inputs: a 3d tensor
    :param num_units: the number of the hidden cell
    :return: a 3d tensor
    """
    with tf.variable_scope('feedforward',reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.leaky_relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs
        outputs = ln(outputs)
        return outputs

def ln(inputs, epsilon=1e-8):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope('ln', reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer(), )
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs