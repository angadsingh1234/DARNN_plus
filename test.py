#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:36:44 2023

@author: angadsingh
"""

from model_config import Config
from wrapped_loadCriteo import loaddualattention

c = Config(max_features = 5897, learning_rate = 0.001, batch_size = 256, feature_number = 12,
           seq_max_len = 20, n_input = 2, embedding_output = 256, n_hidden = 512, n_classes = 2, n_epochs = 50, isseq=True, keep_prob=0.5, miu = 1e-6, layers = 1)


infile = open('data/test_usr.yzx.txt', 'rb')
while True:
    batch = loaddualattention(c.batch_size, c.seq_max_len, c.feature_number, infile)
    test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
    print(type(click_label))
    if len(test_label) != c.batch_size:
        break