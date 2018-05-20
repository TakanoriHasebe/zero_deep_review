#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:14:57 2018

@author: Takanori
"""

import numpy as np

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

# ソフトマックス関数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


