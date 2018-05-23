#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 08:20:58 2018

@author: Takanori
"""

import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)    
    
    # それぞれに対して偏微分を行なっている
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))












