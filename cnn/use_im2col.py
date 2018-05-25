#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:26:18 2018

@author: Takanori
"""
import numpy as np
from im2col_ import im2col

x1 = np.random.rand(1, 3, 7, 7) # 1個のデータ
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)




"""
x2 = np.random.rand(10, 3, 7, 7) # 10個のデータ
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)
"""







