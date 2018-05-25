#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:15:05 2018

@author: Takanori
"""

import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    print(N, C, H, W)
    out_h = (H + 2*pad - filter_h)//stride + 1
    print('out_h: '+str(out_h))
    out_w = (W + 2*pad - filter_w)//stride + 1
    print('out_w : '+str(out_w))
    
    # データが入っている
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    print('img.shape : '+str(img.shape))
    # 最終的に
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    print("col.shape : "+str(col.shape))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    print('col.shape : '+str(col.shape))
    return col











