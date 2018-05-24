#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:47:34 2018

@author: Takanori
"""

import numpy as np

"""
乗算レイヤ
"""
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

# クラスを作成
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward処理
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
# print(price)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# print(dapple, dapple_num, dtax)

"""
加算レイヤの実装
"""

class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple_num = 2
apple = 100
mikan = 150
mikan_num = 3
tax = 1.1

# 個々にレイヤを実装していることに注意
mul_apple_layer = MulLayer()
mul_mikan_layer = MulLayer()
add_apple_mikan_layer = AddLayer()
mul_tax_layer = MulLayer()

# forawrd
apple_price = mul_apple_layer.forward(apple_num, apple)
mikan_price = mul_mikan_layer.forward(mikan, mikan_num)
all_price = add_apple_mikan_layer.forward(apple_price, mikan_price)
price = mul_tax_layer.forward(all_price, tax)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_mikan_layer.backward(dall_price)
dapple_num, dapple = mul_apple_layer.backward(dapple_price)
dmikan, dmikan_num = mul_mikan_layer.backward(dorange_price)

print(price)
print(dall_price, dtax)
print(dapple_price, dorange_price)
print(dapple_num, dapple)
print(dmikan, dmikan_num)

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class Sigmoid():
    def __init__(self):
        self.out = None
    
    # 順伝播
    def forward(self, x):
        out = 1 / (1 + np.exp(x))
        self.out = out
        return out

    # 逆伝播
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    # 順伝搬    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
        
    # 逆伝搬
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx

class SoftmaxWithLoss():
    
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        
        self.loss = cross_entropy_error(t, y)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx














