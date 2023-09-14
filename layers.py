#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:51:15 2023

@author: angadsingh
"""
import tensorflow as tf
from tensorflow import keras

class AttentionLayerImpressions(keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def build(self, input_shape):
        self.W_h = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="W_h", trainable=True)
        self.W_x1 = self.add_weight(shape=(self.config.n_input, self.config.n_hidden), initializer="random_normal", name="W_x1", trainable=True)
        self.v_a2 = self.add_weight(shape=(self.config.n_hidden,), initializer="random_normal", name='v_a2', trainable = True)
      
    @tf.function
    def call(self, inputs):
    
        x_last = inputs['x_last']
        states_h = inputs['states_h']
        
        e2 = []
        Ux = tf.matmul(x_last, self.W_x1)
      
        for i in range(self.config.seq_max_len):
            e2_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_h[i], self.W_h) + Ux), self.v_a2), axis = 1)
            e2.append(e2_)
      
        e2 = tf.stack(e2)
        a2 = tf.nn.softmax(e2, axis=0)
        a2 = tf.split(a2, self.config.seq_max_len, 0)
      
        c2 = tf.zeros([self.config.batch_size, self.config.n_hidden])
      
        for i in range(self.config.seq_max_len):
            c2 = c2 +  tf.multiply(states_h[i], tf.transpose(a2[i]))
      
        return c2

class AttentionLayerClicks(keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def build(self, input_shape):
        self.U_s = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="U_s", trainable=True)
        self.W_x2 = self.add_weight(shape=(self.config.n_input, self.config.n_hidden), initializer="random_normal", name="W_x2", trainable=True)
        self.v_a3 = self.add_weight(shape=(self.config.n_hidden,), initializer="random_normal", name='v_a3', trainable = True)
      
    @tf.function
    def call(self, inputs):
    
        x_last = inputs['x_last']
        states_s = inputs['states_s']
        
        # states_s = tf.reshape(states_s, shape = (-1, self.config.n_hidden))
        # states_s = tf.split(states_s, self.config.seq_max_len, axis = 0)
        
        e3 = []
        Ux2 = tf.matmul(x_last, self.W_x2)
    
        for i in range(self.config.seq_max_len):
            e3_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_s[i], self.U_s) + Ux2), self.v_a3), axis = 1)
            e3.append(e3_)
      
        e3 = tf.stack(e3)
        a3 = tf.nn.softmax(e3, axis=0)
        a3 = tf.split(a3, self.config.seq_max_len, 0)
      
        c3 = tf.zeros([self.config.batch_size, self.config.n_hidden])
      
        for i in range(self.config.seq_max_len):
            c3 = c3 + tf.multiply(states_s[i], tf.transpose(a3[i]))
      
        return c3

class ConversionLayer(keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def build(self, input_shape):
        self.W_x3 = self.add_weight(shape=(self.config.n_input, self.config.n_hidden), initializer="random_normal", name="W_x3", trainable=True)
        self.W_C = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="W_C", trainable=True)
        self.v_a = self.add_weight(shape=(self.config.n_hidden,), initializer="random_normal", name='v_a', trainable = True)
        self.b_c = self.add_weight(shape=(self.config.batch_size,), initializer="random_normal", name='b_c', trainable = True)
        self.W_c = self.add_weight(shape=(self.config.n_hidden, ), initializer="random_normal", name="W_C", trainable=True)
    
    @tf.function
    def call(self, inputs, training):
        x_last = inputs['x_last']
        c2 = inputs['c2']
        c3 = inputs['c3']
      
        e4 = []
        Ux3 = tf.matmul(x_last, self.W_x3)
        e4.append(tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(c2, self.W_C) + Ux3), self.v_a), axis=1))
        e4.append(tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(c3, self.W_C) + Ux3), self.v_a), axis=1))
        e4 = tf.stack(e4)
        a4 = tf.split(tf.nn.softmax(e4, axis=0), 2, 0)
        C = tf.multiply(c2, tf.transpose(a4[0])) + tf.multiply(c3, tf.transpose(a4[1]))
        cvr = tf.reduce_sum(tf.multiply(C, self.W_c), axis=1) + self.b_c
        if training == True:
            cvr = tf.nn.dropout(cvr, rate=(1 - self.config.keep_prob))

        return cvr
