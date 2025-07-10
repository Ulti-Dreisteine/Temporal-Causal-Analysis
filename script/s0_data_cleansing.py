# -*- coding: utf-8 -*-
"""
Created on 2025/07/10 14:57:48

@File -> s0_data_cleansing.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 载入数据
"""

from typing import Dict
import pandas as pd
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.signal_wavelet_decomp import SignalWaveletDecomp

cols_alias = {
    "Global Temperature Anomalies": "全球温度异常",
    "Global Precipitation": "全球降水量",
    "Nino 1+2 SST": "Nino 1+2 区域海表温度",
    "Nino 1+2 SST Anomalies": "Nino 1+2 区域海表温度异常",
    "Nino 3 SST": "Nino 3 区域海表温度",
    "Nino 3 SST Anomalies": "Nino 3 区域海表温度异常",
    "Nino 3.4 SST": "Nino 3.4 区域海表温度",
    "Nino 3.4 SST Anomalies": "Nino 3.4 区域海表温度异常",
    "Nino 4 SST": "Nino 4 区域海表温度",
    "Nino 4 SST Anomalies": "Nino 4 区域海表温度异常",
    "TNI OISST": "跨尼诺指数（OISST）",
    # "TNI HadISST": "跨尼诺指数（HadISST）",
    "PNA": "太平洋北美指数",
    # "OLR": "向外长波辐射",
    "SOI": "南方涛动指数",
    "MEI.ext": "多变量厄尔尼诺指数（扩展）",
    "MEI.v2": "多变量厄尔尼诺指数（版本2）",
    "ONI": "海洋尼诺指数",
}


def perform_filtering(df: pd.DataFrame, col_wt_params: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    对数据进行小波滤波

    Params:
    -------
    df: pd.DataFrame, 原始数据
    col_wt_params: Dict[str, Dict[str, str]], 每列的滤波参数

    Returns:
    --------
    pd.DataFrame, 滤波后的数据
    """
    df_filtered = pd.DataFrame()
    for col in df.columns:
        x = df[col].values
        decomp = SignalWaveletDecomp(x)
        params = col_wt_params[col]
        signals_decomp = decomp.decompose(
            wavelet_name=params["wavelet_name"],
            wt_levels=params["wt_levels"],
            low_freq_frac_thres=params["low_freq_frac_thres"],
            high_freq_frac_thres=params["high_freq_frac_thres"]
        )
        df_filtered[col] = signals_decomp["filtered"]

    return df_filtered


def show_filtered_signals(df: pd.DataFrame, df_filtered: pd.DataFrame):
    """
    显示滤波前后的信号

    Params:
    -------
    df: pd.DataFrame, 原始数据
    df_filtered: pd.DataFrame, 滤波后的数据
    """
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(df.columns):
        plt.subplot(len(df.columns) // 3 + 1, 3, i + 1)
        plt.plot(df.index, df[col], label="原始信号")
        plt.plot(df_filtered.index, df_filtered[col], label="滤波信号")
        plt.title(col)
        plt.xlim(0, len(df))
        plt.legend(loc="upper right")
        plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(f"{BASE_DIR}/data/ENSO.csv", index_col=0)

    # ---- 提取字段 ---------------------------------------------------------------------------------

    cols = [
        "Global Temperature Anomalies",
        "Global Precipitation",
        "Nino 1+2 SST",
        "Nino 1+2 SST Anomalies",
        "Nino 3 SST",
        "Nino 3 SST Anomalies",
        "Nino 3.4 SST",
        "Nino 3.4 SST Anomalies",
        "Nino 4 SST",
        "Nino 4 SST Anomalies",
        "TNI OISST",
        # "TNI HadISST",
        "PNA",
        # "OLR",
        "SOI",
        "MEI.ext",
        "MEI.v2",
        "ONI",
    ]

    df = df[cols]
    df.rename(columns=cols_alias, inplace=True)

    # 画图
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(df.columns):
        plt.subplot(len(cols) // 3 + 1, 3, i + 1)
        plt.plot(df.index, df[col])
        plt.title(col)
        plt.xlim(0, len(df))
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # ---- 去趋势和去噪声 ----------------------------------------------------------------------------

    wt_levels = 7

    col_wt_params_denoise = {
        # "全球温度异常": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.2, "high_freq_frac_thres": 1.0},
        "全球温度异常": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "全球降水量": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "Nino 1+2 区域海表温度": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "Nino 1+2 区域海表温度异常": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "Nino 3 区域海表温度": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "Nino 3 区域海表温度异常": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "Nino 4 区域海表温度": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "Nino 4 区域海表温度异常": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "Nino 3.4 区域海表温度": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "Nino 3.4 区域海表温度异常": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "跨尼诺指数（OISST）": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "跨尼诺指数（HadISST）": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "太平洋北美指数": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "向外长波辐射": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "南方涛动指数": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "多变量厄尔尼诺指数（扩展）": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "多变量厄尔尼诺指数（版本2）": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
        "海洋尼诺指数": {"wavelet_name": "db4", "wt_levels": wt_levels, "low_freq_frac_thres": 0.0, "high_freq_frac_thres": 1.0},
    }
    df_filtered = perform_filtering(df, col_wt_params_denoise)
    show_filtered_signals(df, df_filtered)

    # ---- 保存数据 ---------------------------------------------------------------------------------

    df = df.copy()
    df["全球温度异常"] = df_filtered["全球温度异常"]

    df.to_csv("runtime/s0_data.csv", index=False)