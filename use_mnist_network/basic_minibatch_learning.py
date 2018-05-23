#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 06:43:25 2018

@author: Takanori
"""

from mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# ランダムに訓練データからとってきていることが分かる。
print(batch_mask)











