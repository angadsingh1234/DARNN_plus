#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:52:20 2023

@author: angadsingh
"""
import tensorflow as tf
import os
import numpy as np
import time

from encoder import Encoder
from decoder import Decoder
from layers import  AttentionLayerImpressions, AttentionLayerClicks, ConversionLayer

from datetime import timedelta
from sklearn.metrics import roc_auc_score, log_loss
from tensorflow import keras
from tensorflow.keras.layers import Embedding

from model_config import Config
from wrapped_loadCriteo import loaddualattention

class DARNN(keras.Model):
    
    def __init__(self, path, trainpath, testpath, config):
        super().__init__()
        self._path = path
        self._save_path, self._logs_path = None, None
        self.testpath = testpath
        self.trainpath = trainpath
        self.config = config
        
       
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=self.config.learning_rate,
          decay_steps=50000,
          decay_rate=0.96,
          staircase=True
        )
        
        self.clk_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001, epsilon=1e-7)
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule, epsilon=1e-7)

    def get_config(self):
        
        base_config = super().get_config()
        return {**base_config, "config": self.config.get_config()}  # Serialize the config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        
        model_path = config["path"]
        trainpath = config["trainpath"]
        testpath = config["testpath"]
        model_config = Config.from_config(config["config"])  # Deserialize the config
      
        return cls(model_path, trainpath, testpath, config=model_config)


    @property
    def save_path(self):
        if self._save_path is None:
          save_path = '%s/checkpoint' % self._path
          if not os.path.exists(save_path):
            os.makedirs(save_path)
          save_path = os.path.join(save_path, 'model.ckpt')
          self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
          logs_path = '%s/logs' % self._path
          if not os.path.exists(logs_path):
            os.makedirs(logs_path)
          self._logs_path = logs_path
        return self._logs_path

    def build(self, input_shape):
        self.encoder = Encoder(config = self.config)
        self.decoder = Decoder(config = self.config)
        self.a_imp = AttentionLayerImpressions(config = self.config)
        self.a_clicks = AttentionLayerClicks(config = self.config)
        self.embedding_layer = Embedding(self.config.max_features, self.config.embedding_output)
        self.conv_layer = ConversionLayer(config = self.config)

    @tf.function
    def call(self, inputs, training):
    
        x = inputs['total_data']
        seqlen = inputs['seqlen']
        y = inputs['click_label']
        labels = inputs['labels']
        
        x1, x2 = tf.split(x, [2,10], 2)
        x2 = tf.cast(x2, dtype=tf.int32)
        x2 = self.embedding_layer(x2)
        x2 = tf.reshape(x2, [-1, self.config.seq_max_len, 10 * self.config.embedding_output])
      
        x = tf.concat((x1, x2), axis=2)
      
        index = tf.range(0, self.config.batch_size) * self.config.seq_max_len + (seqlen - 1)
        x_last = tf.gather(params = tf.reshape(x, [-1, self.config.n_input]), indices = index)
      
        states_h, last_h = self.encoder(x, training)
        states_s, outputs = self.decoder({'last_h': tf.cast(last_h, dtype=tf.float32), 'click_label': tf.cast(y, dtype=tf.float32)}, training)
        c2 = self.a_imp({'states_h': tf.cast(states_h, dtype=tf.float32), 'x_last': tf.cast(x_last, dtype=tf.float32)})
        c3 = self.a_clicks({'states_s': tf.cast(states_s, dtype=tf.float32), 'x_last': tf.cast(x_last, dtype=tf.float32)})
        cvr = self.conv_layer({'x_last': tf.cast(x_last, dtype=tf.float32), 'c2': tf.cast(c2, dtype=tf.float32), 'c3': tf.cast(c3, dtype=tf.float32)}, training)
        cvr_pred = tf.nn.sigmoid(cvr)
        click_pred = tf.nn.softmax(outputs)
        
        conversion_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.cast(labels, dtype=tf.float32), logits = cvr))
        mask = tf.sequence_mask(seqlen, self.config.seq_max_len)
        loss_click = tf.nn.softmax_cross_entropy_with_logits(labels = tf.cast(y, dtype=tf.float32), logits = outputs)
        loss_click = tf.boolean_mask(loss_click, mask)
        loss_click = tf.reduce_mean(loss_click)
        
        loss = conversion_loss + loss_click
      
        return loss, conversion_loss, loss_click, cvr_pred, click_pred

    @tf.function
    def train_step_total(self, inputs):
    
        with tf.GradientTape() as tape_total:
            loss, conversion_loss, loss_click, cvr_pred, click_pred = self(inputs, training = True)
            
            for v in self.trainable_weights:
                loss += self.config.miu * tf.nn.l2_loss(v)
                conversion_loss += self.config.miu * tf.nn.l2_loss(v)
                loss_click += self.config.miu * tf.nn.l2_loss(v)
        
        # Calculate gradients and apply optimization steps for the overall loss
        gradients = tape_total.gradient(loss, self.trainable_weights)
        
        clipped_gradients_total, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        self.optimizer.apply_gradients(zip(clipped_gradients_total, self.trainable_weights))
      
        return loss, conversion_loss, loss_click, cvr_pred, click_pred
    
    @tf.function
    def train_step_click(self, inputs):
    
        with tf.GradientTape() as tape_clk:
            loss, conversion_loss, loss_click, cvr_pred, click_pred = self(inputs, training = True)
            
            for v in self.trainable_weights:
                loss += self.config.miu * tf.nn.l2_loss(v)
                conversion_loss += self.config.miu * tf.nn.l2_loss(v)
                loss_click += self.config.miu * tf.nn.l2_loss(v)
         
        # Calculate gradients and apply optimization steps for loss_click
        gradients_clk = tape_clk.gradient(loss_click, self.trainable_weights)
        
        clipped_gradients_clk, _ = tf.clip_by_global_norm(gradients_clk, 5.0)
        
        self.clk_optimizer.apply_gradients(zip(clipped_gradients_clk, self.trainable_weights))
      
        return loss, conversion_loss, loss_click, cvr_pred, click_pred


    def train_one_epoch(self, batch_size, learning_rate, train_fn):
        total_loss= []
        total_clk_loss=[]
        total_cov_loss= []
        clk_pred = []
        clk_label = []
        cvr_pred = []
        cvr_label = []
        infile = open(self.trainpath, 'rb')
        # i = 0
        while True:
          batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
          train_data, train_compaign_data, click_label, train_label, train_seqlen = batch
          # i += 1
          # print(i)
          if len(train_label) != batch_size:
            break
          inputs = {
            'total_data': tf.cast(train_data, dtype=tf.float32),
            'seqlen': tf.cast(train_seqlen, dtype=tf.int32),
            'click_label': tf.cast(tf.constant(click_label), dtype=tf.int32),
            'labels': tf.cast(train_label, dtype=tf.int32)
          }
          loss, conversion_loss, click_loss, cvr, clk = train_fn(inputs)
          total_loss.append(loss)
          total_clk_loss.append(click_loss)
          total_cov_loss.append(conversion_loss)
          clk = np.reshape(clk, (-1, 2)).tolist()
          click_label = np.reshape(click_label, (-1, 2)).tolist()
          clk_pred += clk
          clk_label += click_label
          cvr_pred += cvr.numpy().tolist()
          cvr_label += train_label
      
        clk_pred = np.array(clk_pred)
        auc_clk = roc_auc_score(np.argmax(clk_label, 1), clk_pred[:, 1])
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        print("click_AUC = " + "{:.4f}".format(auc_clk))
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        mean_clk_loss = np.mean(total_clk_loss)
        mean_cov_loss = np.mean(total_cov_loss)
        print("Clk Loss = " + "{:.4f}".format(mean_clk_loss))
        print("Cov_Loss = " + "{:.4f}".format(mean_cov_loss))
        return mean_loss, mean_cov_loss, mean_clk_loss, auc_cov, auc_clk

    def train_all_epochs(self, start_epoch=1):
        n_epoches = self.config.n_epochs
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size
      
        total_start_time = time.time()
        for epoch in range(start_epoch, n_epoches + 1):
          print('\n', '-' * 30, 'Train epoch: %d' % epoch, '-' * 30, '\n')
          start_time = time.time()
      
          print("Training...")
          result = self.train_one_epoch(batch_size, learning_rate, self.train_step_total)
          self.log(epoch, result, prefix='train')
          time_per_epoch = time.time() - start_time
          seconds_left = int((n_epoches - epoch) * time_per_epoch)
          print('Time per epoch: %s, Est. complete in: %s' % (
            str(timedelta(seconds=time_per_epoch)),
            str(timedelta(seconds=seconds_left))
          ))
      
        self.save_model()
      
        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def train_until_cov(self):
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size
      
        total_start_time = time.time()
        epoch = 1
        losses = []
        clk_losses = []
        
        train_function = self.train_step_click
        flag = 0
        
        while True:
          print('-' * 30, 'Train epoch: %d' % epoch, '-' * 30)
          start_time = time.time()
      
          print("Training...")
          if flag == 0 and (epoch > 10 or (epoch > 3 and clk_losses[-1] < clk_losses[-2] < clk_losses[-3])):
              flag = epoch
              train_function = self.train_step_total
      
          result = self.train_one_epoch(batch_size, learning_rate, train_function)
          self.log(epoch, result, prefix='train')
      
          loss = self.test(epoch)
          time_per_epoch = time.time() - start_time
          losses.append(loss[0])
          clk_losses.append(loss[1])
          if flag != 0 and (epoch > flag + 3 and losses[-1] < losses[-2] < losses[-3]):
              self.save_model()
              break
          print('Time per epoch: %s' % (
              str(timedelta(seconds=time_per_epoch))
          ))
          epoch += 1
          self.save_model()
      
        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def test(self, epoch):
        batch_size = self.config.batch_size
        total_loss = []
        clk_pred = []
        clk_label = []
        cvr_pred = []
        total_clk_loss = []
        total_cov_loss = []
        cvr_label = []
        infile = open(self.testpath, 'rb')
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            inputs = {
              'total_data': tf.cast(test_data, dtype=tf.float32),
              'seqlen': tf.cast(test_seqlen, dtype=tf.int32),
              'click_label': tf.cast(tf.constant(click_label), dtype=tf.int32),
              'labels': tf.cast(test_label, dtype=tf.int32)
            }
            result = self(inputs, training = False)
            loss, cov_loss, clk_loss, cvr, clk = result
            total_loss.append(loss)
            total_clk_loss.append(clk_loss)
            total_cov_loss.append(cov_loss)
            clk = np.reshape(clk, (-1, 2)).tolist()
            click_label = np.reshape(click_label, (-1, 2)).tolist()
            clk_pred += clk
            clk_label += click_label
            cvr_pred += cvr.numpy().tolist()
            cvr_label += test_label
          
        clk_pred = np.array(clk_pred)
        auc_clk = roc_auc_score(np.argmax(clk_label, 1), clk_pred[:, 1])
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        loglikelyhood = -log_loss(cvr_label, cvr_pred)
        print("click_AUC = " + "{:.4f}".format(auc_clk))
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        print("loglikelyhood = " + "{:.4f}".format(loglikelyhood))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        mean_clk_loss = np.mean(total_clk_loss)
        mean_cov_loss = np.mean(total_cov_loss)
        print("Clk Loss = " + "{:.4f}".format(mean_clk_loss))
        print("Cov_Loss = " + "{:.4f}".format(mean_cov_loss))
        self.log(epoch, [mean_loss, mean_cov_loss, mean_clk_loss, auc_cov, auc_clk], 'test')
        return auc_cov, auc_clk

    def save_model(self):
        self.save(self.save_path)

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
          s += ('\t' + str(i))
        fout = open("%s/%s_%s_%s_%s" % (self.logs_path, str(self.config.learning_rate), str(self.config.batch_size), str(self.config.n_hidden), str(self.config.miu)),'a')
        fout.write(s + '\n')

    def load_model(self, custom_objects = None):
        try:
          return tf.keras.models.load_model(self.save_path, custom_objects=custom_objects)
        except Exception:
          raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)

c = Config(max_features = 5897, learning_rate = 0.000001, batch_size = 256, feature_number = 12,
           seq_max_len = 20, n_input = 2, embedding_output = 256, n_hidden = 512, n_classes = 2, n_epochs = 50, isseq=True, keep_prob=0.71, miu = 1e-4, layers = 3)

path = 'models/DARNN_3'
traindata = 'data/train_usr.yzx.txt'
testdata = 'data/test_usr.yzx.txt'
model = DARNN(path, traindata, testdata, config = c)
# model.train_one_epoch(256, 0.001, model.train_step_total)
model.train_until_cov()
model.test(0)