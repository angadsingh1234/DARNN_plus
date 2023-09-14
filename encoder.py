#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:44:01 2023

@author: angadsingh
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GRU


class Encoder(keras.layers.Layer):
    def __init__(self, config, name="encoder"):
        super().__init__(name = name)
        self.config = config
    
    def build(self, input_shape):
        
        self.gru_layers = [GRU(self.config.n_hidden, return_sequences = True, return_state=True,
                                 dropout =(1 - self.config.keep_prob), recurrent_dropout=(1 - self.config.keep_prob), 
                                 unroll = True, time_major = False) for _ in range(self.config.layers)]
     
    @tf.function
    def call(self, inputs, training):
    
        x = inputs
      
        for gru_layer in self.gru_layers:
              x = gru_layer(x, training = training)
      
        states_h, last_h = x
        
        states_h = tf.reshape(states_h, shape=(-1, self.config.n_hidden)) 
        states_h = tf.split(states_h, self.config.seq_max_len, axis = 0)
      
        return states_h, last_h
