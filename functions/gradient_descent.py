#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 08:41:33 2018

@author: Takanori
"""

import numpy as np

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

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        
        x -= lr * grad
        
    return x

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))



