# -*- coding: utf-8 -*-
"""
Created on 2025/07/09 10:20:19

@File -> signal_wavelet_decomp.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 信号小波分解
"""

from typing import Dict
import numpy as np
import pywt


class WTFilter(object):
    """小波滤波器"""
    
    def __init__(self, wavelet_name: str, wt_levels: int):
        """
        初始化
        
        Params:
        -------
        wavelet_name: 小波基名称
        wt_levels: 小波分解层数
        """
        self.wavelet_name = wavelet_name
        self.wt_levels = wt_levels
    
    def fit(self, x):
        self.coeffs = pywt.wavedec(x, self.wavelet_name, level=self.wt_levels)
    
    def filter_high_freqs(self, frac: float = None):
        """滤除高频趋势"""
        frac = frac if frac is not None else 0.1
        n = len(self.coeffs)
        for i in range(int(frac * n), n):
            self.coeffs[i] *= 0

    def filter_low_freqs(self, frac: float = None):
        """滤除低频趋势"""
        frac = frac if frac is not None else 0.1
        n = len(self.coeffs)
        for i in range(0, int(frac * n)):
            self.coeffs[i] *= 0
    
    def inverse_transform(self) -> np.ndarray:
        """逆变换获得滤波结果"""
        return pywt.waverec(self.coeffs, self.wavelet_name)


class SignalWaveletDecomp:
    """信号小波滤波"""
    
    def __init__(self, series: np.ndarray):
        self.series = series.flatten()

    def decompose(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        信号滤波

        Params:
        -------
        kwargs: 传入WTFilter的参数
            wavelet_name: 小波基名称
            wt_levels: 小波分解层数
            low_freq_frac_thres: 滤除低频趋势的比例阈值
            high_freq_frac_thres: 滤除高频趋势的比例阈值
        """

        wavelet_name = kwargs.get("wavelet_name", "haar")
        wt_levels = kwargs.get("wt_levels", 20)
        low_freq_frac_thres = kwargs.get("low_freq_frac_thres", None)
        high_freq_frac_thres = kwargs.get("high_freq_frac_thres", None)

        wtf = WTFilter(wavelet_name=wavelet_name, wt_levels=wt_levels)
        wtf.fit(self.series)
        wtf.filter_low_freqs(frac=low_freq_frac_thres)
        wtf.filter_high_freqs(frac=high_freq_frac_thres)
        series_ = wtf.inverse_transform()

        signals_decomp = {
            "filtered": series_
        }

        return signals_decomp
    

if __name__ == "__main__":
    pass