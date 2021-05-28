# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:16:13 2018

@author: wangyt
"""

#导入所需包
import tensorflow as tf




class Net(object):
    def __init__(self):
        self.name='cpsc2018'
        self.seq_len = 3000          # length of ECG_data
        self.n_channels = 8
        self.n_classes = 9        
        
        
        self.inputs = tf.placeholder(tf.float32, [None, self.seq_len, self.n_channels], name = 'inputs')
        self.labels  = tf.placeholder(tf.float32, [None, self.n_classes], name = 'labels')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep')
        self.tf_is_train = tf.placeholder(tf.bool, None)     # flag for using BN on training or testing
        self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        
        
        self.one_block_result = self.one_block(self.inputs,self.tf_is_train,self.keep_prob)
        self.logits = self.net(self.one_block_result,self.tf_is_train,self.keep_prob)


        # Cost function and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.cost)

        # Prediction
        self.prediction = tf.argmax(self.logits,axis =1)
        
        #Accuracy
        self.correct_pred = tf.equal(self.prediction, tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

        #probility
        self.probability=tf.nn.softmax(self.logits)
    
    
    def one_block(self,inputs,tf_is_train,keep_prob):
        #输入标准化处理
        inputs_BN = tf.layers.batch_normalization(inputs,training = tf_is_train)
        #卷积，
        conv1 = tf.layers.conv1d(inputs=inputs_BN, filters=64, kernel_size=16, strides=1, padding='same', activation = None)
        #标准化处理
        BN1 = tf.layers.batch_normalization(conv1, training=tf_is_train)
        #激活函数，作为第二层卷积的输入
        conv1_relu = tf.nn.relu(BN1)
        #第一层池化,将与第三层卷积合并
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1_relu, pool_size=2, strides=2, padding='same')
        #第2层卷积，上接conv1_relu
        conv2 = tf.layers.conv1d(inputs=conv1_relu, filters=64, kernel_size=16, strides=1, padding='same', activation = None)
        #标准化处理
        conv2_BN2 = tf.layers.batch_normalization(conv2, training=tf_is_train)
        #激活函数，
        conv2_BN2_relu = tf.nn.relu(conv2_BN2)
        #dropout
        conv2_BN2_relu_dropout = tf.nn.dropout(conv2_BN2_relu, keep_prob=keep_prob)
        #第3层卷积，上接conv2_BN2_relu_dropout
        conv3 = tf.layers.conv1d(inputs=conv2_BN2_relu_dropout, filters=64, kernel_size=16, strides=2, padding='same', activation = None)
        #第一次合并：合并max_pool_1与conv3
        result = tf.add(max_pool_1,conv3)

        return result 
    
    def conv_max_residual_block(self,inputs,num_filters_factor,subsample_factor,tf_is_train,keep_prob,i):
    
        #对输入进行卷积主体（two conv）操作
        BN_1 = tf.layers.batch_normalization(inputs, training=tf_is_train)
        relu_1 = tf.nn.relu(BN_1)
        dropout_1 = tf.nn.dropout(relu_1, keep_prob=keep_prob)
        conv_1 = tf.layers.conv1d(inputs=dropout_1, filters=64*num_filters_factor, kernel_size=16, strides=1, padding='same', activation = None)
        BN_2 = tf.layers.batch_normalization(conv_1, training=tf_is_train)
        relu_2 = tf.nn.relu(BN_2)
        dropout_2 = tf.nn.dropout(relu_2, keep_prob=keep_prob)
        conv_2 = tf.layers.conv1d(inputs=dropout_2, filters=64*num_filters_factor, kernel_size=16, strides=subsample_factor, padding='same', activation = None)

        #对输入做池化操作（one max）
        if (i-1)%4 == 0:
            inputs = tf.layers.conv1d(inputs=inputs, filters=64*num_filters_factor, kernel_size=16, strides=1, padding='same', activation = None)
        max_pooling_input = tf.layers.max_pooling1d(inputs=inputs, pool_size=2, strides=subsample_factor, padding='same')
        #将主体结果和池化结果合并
        result = tf.add(max_pooling_input,conv_2)

        return result
    
    def net(self,inputs,tf_is_train,keep_prob):
        for i in range(2,17):
            #选择下采样率
            if i%2 == 0:
                subsample_factor = 1
            else:
                subsample_factor = 2
            #选择滤波器增加率
            num_filters_factor = int((i-1)/4)+1 
            inputs = self.conv_max_residual_block(inputs,num_filters_factor,subsample_factor,tf_is_train,keep_prob,i)
        
        
        #标准化
        inputs = tf.layers.batch_normalization(inputs, training=tf_is_train)
        #激活函数，
        inputs = tf.nn.relu(inputs)
        
        flat = tf.reshape(inputs, (-1, 12*64*num_filters_factor))
        flat_drop = tf.nn.dropout(flat, keep_prob=keep_prob)
        logits = tf.layers.dense(flat_drop, self.n_classes)

        return logits  
    
    
    
        
