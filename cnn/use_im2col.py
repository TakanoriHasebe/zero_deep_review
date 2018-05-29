#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:44:25 2018

@author: Takanori
"""

import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

x1 = np.random.rand(1,3,7,7) # バッチサイズが1
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)

x1 = np.random.rand(10,3,7,7) # バッチサイズが10
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)

# Convolution層を作成
class Convolution:
    
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forwawrd(self, x):
        FN, C, FH, FW = self.W.shape # フィルターの個数, チャネル, フィルターの高さ, フィルターの横幅
        N, C, H, W = x.shape # バッチサイズ, チャネル, 高さ, 横幅
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride) # 入力サイズ, パディング, フィルターの高さ, ストライド
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride) # 横幅, パディング, フィルターの横幅, ストライド
        
        col = im2col(x, FH, FW, self.stride, self.pad) # x : データ, フィルターの高さ, フィルターの横幅, ストライド, パディング
        col_W = self.W.reshape(FN, -1).T # im2col関数を用いるために広げる
        out = np.dot(col, col_W) + self.b
        
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # N, C, H, W
        
        return out

print()
arr = np.random.randn(1,2)
print(arr)
print(arr.transpose(1,0))
print()

arr = np.random.randn(1,2,3)
print(arr)
print(arr.transpose(0,1,2))
print("transpose")
print(arr.transpose(1,0,2).shape)




