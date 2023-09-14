#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:46:26 2023

@author: angadsingh
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GRU

class Decoder(keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def build(self, input_shape):
        self.W_s = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="W_s", trainable=True)
      
        self.U_z = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="U_z", trainable=True)
        self.C_z = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="C_z", trainable=True)
        self.U_r = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="U_r", trainable=True)
        self.C_r = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="C_r", trainable=True)
      
        self.W_z = self.add_weight(shape=(self.config.n_classes, self.config.n_hidden), initializer="random_normal", name="W_z", trainable=True)
        self.W_r = self.add_weight(shape=(self.config.n_classes, self.config.n_hidden), initializer="random_normal", name="W_r", trainable=True)
      
        self.W = self.add_weight(shape=(self.config.n_classes, self.config.n_hidden), initializer="random_normal", name="W", trainable=True)
        self.U = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="U", trainable=True)
        self.C = self.add_weight(shape=(self.config.n_hidden, self.config.n_hidden), initializer="random_normal", name="C", trainable=True)
      
        self.W_o = self.add_weight(shape=(self.config.n_hidden, self.config.n_classes), initializer="random_normal", name="W_o", trainable=True)
        self.b_o = self.add_weight(shape=(self.config.n_classes,), initializer="random_normal", name="b_o", trainable=True)
        
        if self.config.layers > 1:
          self.gru_layers = [GRU(self.config.n_hidden, return_sequences=True, return_state=True, 
                                 dropout=(1 - self.config.keep_prob), recurrent_dropout=(1 - self.config.keep_prob), unroll = True) for _ in range(self.config.layers - 1)]
    @tf.function
    def call(self, inputs, training):
    
        last_h = inputs['last_h']
        click_label = inputs['click_label']
        
        # states_h = tf.reshape(states_h, [-1, self.config.n_hidden])
        # states_h = tf.split(states_h, self.config.seq_max_len, 0)

        y = tf.transpose(click_label, [1,0,2])
        y = tf.reshape(y, [-1, self.config.n_classes])
        y = tf.split(value = y, num_or_size_splits=self.config.seq_max_len, axis = 0)
        
        # s0 =  tanh(Ws * h_last)
        state_s = tf.tanh(tf.matmul(last_h, self.W_s))
        
        states_s = [state_s]
        
        outputs = []
        output = tf.zeros(shape=(self.config.batch_size, self.config.n_classes))
        
        for i in range(self.config.seq_max_len):
      
          c = last_h
      
          if training == True:
            last_output = y[i]
          else:
            last_output = tf.nn.softmax(output)
        
          z = tf.sigmoid(tf.matmul(last_output, self.W_z) + tf.matmul(states_s[i], self.U_z) + tf.matmul(c, self.C_z))
          r = tf.sigmoid(tf.matmul(last_output, self.W_r) + tf.matmul(states_s[i], self.U_r) + tf.matmul(c, self.C_r))
      
          s_hat = tf.tanh(tf.matmul(last_output, self.W) + tf.matmul(tf.multiply(r, states_s[i]), self.U) + tf.matmul(c, self.C))
         
          state_s = tf.multiply(tf.subtract(1.0, z), states_s[i]) + tf.multiply(z, s_hat)
      
          states_s.append(state_s)
          
          if training == True and self.config.layers == 1:
              state_s = tf.nn.dropout(state_s, rate = (1 - self.config.keep_prob))
      
          if self.config.layers == 1:
              # print('hello')
              output = tf.matmul(state_s, self.W_o) + self.b_o
              outputs.append(output)
      
        #Exclude s0 from the states
        states_s = states_s[1:]
      
        if self.config.layers > 1:
            
            #Reshape states_s again
            states_s = tf.stack(states_s, axis = 1)
            
            for gru_layer in self.gru_layers:
                  states_s, _ = gru_layer(states_s, training = training)
        
            states_s = tf.reshape(states_s, shape=(-1, self.config.n_hidden)) 
            states_s = tf.split(states_s, self.config.seq_max_len, axis = 0)
           
            for i in range(self.config.seq_max_len):
                output = tf.matmul(states_s[i], self.W_o) + self.b_o
                output = tf.reshape(output, (self.config.batch_size, self.config.n_classes))
                outputs.append(output)
      
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
      
        return states_s, outputs