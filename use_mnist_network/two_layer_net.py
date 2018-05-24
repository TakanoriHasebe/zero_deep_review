#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:31:35 2018

@author: Takanori
"""

import sys
sys.path.append('../functions/')
import numpy as np
from activationfunctions import sigmoid, softmax, cross_entropy_error, numerical_gradient


class TwoLayerNet:
    
    # 最初に重みとバイアスの初期化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
                  np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                  np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    # 予測する
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        # print('a1:'+str(a1.shape))
        z1 = sigmoid(a1)
        # print('z1:'+str(z1.shape))
        a2 = np.dot(z1, W2) + b2
        # print('a2:'+str(a2.shape))
        y = softmax(a2)
        # print('y'+str(y.shape))
        
        return y
    
    # 誤差算出
    def loss(self, x, t):
        # predictを用いて
        y = self.predict(x)
        # 交差エントロピーで算出
        return cross_entropy_error(y, t)
    
    # 予測
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 勾配を算出している
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])        
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads        

# 入力784, 隠れ層100, 出力層10のニューラルネットを作成 
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)

x = np.random.rand(100, 784)
y = net.predict(x)

temp = np.random.rand(2,1)















