# -*- coding: utf-8 -*-
"""
Created on 2025/07/24 09:53:53

@File -> resampler.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 重采样器
"""

import numpy as np

class Resampler(object):
    """马尔可夫链采样器"""

    def __init__(self):
        pass

    def set_samples(self, samples: dict) -> None:
        """
        初始化样本

        Parameters:
        ----------
        samples : dict
            一个包含样本数组的字典，每个键对应一个变量名，值为该变量的样本数组。
        """
        # 将所有x合并为一个数组
        self.keys = list(samples.keys())
        self.arr = np.column_stack([np.ravel(samples[key]) for key in self.keys])
        self.arr = np.atleast_2d(self.arr)
        self.N, self.D = self.arr.shape

    def resample(self, N: int = None):
        """
        重采样

        Params:
        -------
        N: 重采样的样本数量，默认为None，表示与原样本数量相同
        """
        if N is None:
            N = self.N
        
        # 直接有放回重采样
        idxs = np.random.choice(self.N, size=N, replace=True)

        # NOTE：去重对于KNN类方法非常关键，否则大量重复样本会导致概率估计不准确
        
        # 去重
        idxs_unique, _ = np.unique(idxs, return_counts=True)
        
        # 采样
        arr_resampled = self.arr[idxs_unique]

        # 解析
        samples_resampled = {key: arr_resampled[:, i] for i, key in enumerate(self.keys)}

        return samples_resampled