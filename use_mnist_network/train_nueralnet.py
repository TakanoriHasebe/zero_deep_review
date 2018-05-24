#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 07:51:47 2018

@author: Takanori
"""

import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet
from zodbpickle import pickle

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 作成したネットワークを用いる
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配を算出している
    grad = network.numerical_gradient(x_batch, t_batch)
    # 誤差逆伝播法を用いた高速版もあるよ
    
    # 算出した勾配からパラメータを調節している
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    # 誤差算出
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

#pickle形式で保存
pickle.dump(train_loss_list, open('two_train_loss_list.pkl','wb'), protocol=3)
  













