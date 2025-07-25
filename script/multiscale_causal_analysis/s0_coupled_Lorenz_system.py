# -*- coding: utf-8 -*-
"""
Created on 2025/07/25 14:44:24

@File -> s0_coupled_Lorenz_system.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 耦合Lorenz系统
"""

from itertools import permutations
from typing import List
import seaborn as sns
from tqdm import tqdm
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.causal_entropy_analyzer.analyzer import CausalEntropyAnalyzer


# **************************************************************************************************
# 通用工具
# **************************************************************************************************

def _gen_embed_series(x: np.ndarray, idxs: np.ndarray, m: int, tau: int) -> np.ndarray:
    """生成m维时延嵌入序列，返回 shape=(len(idxs), m) 的嵌入矩阵"""
    N = len(idxs)
    X_embed = np.empty((N, m), dtype=x.dtype)
    for i in range(m):
        X_embed[:, i] = x[idxs - (m - 1 - i) * tau]
    return X_embed


def symbolize(x: np.ndarray, tau: int, m: int, tau_max: int) -> np.ndarray:
    """符号化：将嵌入后的序列映射为排列符号索引"""
    patterns = list(permutations(range(1, m + 1)))
    dict_pattern_index = {p: i for i, p in enumerate(patterns)}

    idxs = np.arange((m - 1) * tau_max, len(x))
    X_embed = _gen_embed_series(x, idxs, m, tau)
    # 按行排序并转为tuple作为pattern
    X = np.array([dict_pattern_index[tuple(np.argsort(row) + 1)] for row in X_embed])
    return X


# **************************************************************************************************
# 项目工具
# **************************************************************************************************

def gen_samples(size: int, lags: List[int], alphas: List[float], show: bool = False):
    """
    生成样本数据

    Args:
        size: 样本数量
        lags: x对y作用的时延，单位为采样点数
        alphas: 平滑系数，0<alpha<1，越大则输出的响应越慢
    Returns:
        x: 输入信号
        y: 输出信号
    """
    N_preheat = 500  # 预热长度，避免初始条件对结果的影响

    x = np.random.normal(size=size + N_preheat)

    y_1 = np.zeros_like(x)
    for i in range(len(x)):
        if i < lags[0]:
            y_1[i] = x[i]
        else:
            y_1[i] = alphas[0] * y_1[i - 1] + (1 - alphas[0]) * x[i - lags[0]]

    y_2 = np.zeros_like(x)
    for i in range(len(x)):
        if i < lags[1]:
            y_2[i] = x[i]
        else:
            y_2[i] = alphas[1] * y_2[i - 1] + (1 - alphas[1]) * x[i - lags[1]]

    # y = y_1 * y_2  # 每个时刻上的乘积耦合
    y = 0.1 * y_1 + y_2  # 每个时刻上的加和耦合

    # 截去预热部分
    x = x[N_preheat:]
    y = y[N_preheat:]
    y_1 = y_1[N_preheat:]
    y_2 = y_2[N_preheat:]

    if show:
        plt.figure(figsize=(5, 5))
        plt.subplot(2, 1, 1)
        plt.plot(x[-200:], "k", linewidth=1.5, label="x")

        plt.subplot(2, 1, 2)
        plt.plot(y[-200:], "k", linewidth=1.5, label="y")
        plt.plot(y_1[-200:], "b--", linewidth=1.0, label="y_1")
        plt.plot(y_2[-200:], "r--", linewidth=1.0, label="y_2")
        
        plt.legend()
        plt.tight_layout()
        plt.show()
            
    return x, y


if __name__ == "__main__":
    x_series, y_series = gen_samples(size=10000, lags=[5, 10], alphas=[0.1, 0.9], show=True)

    # ---- 多尺度分析 --------------------------------------------------------------------------------

    m = 3  # 嵌入维度
    scales = np.arange(1, 30 + 3, 3)
    lags_to_test = np.arange(-30, 31)

    # 分析参数
    analysis_params = {
        "method": "MIT",
        "size_bt": 100,
        "rounds_bt": 20,
        "k": 3
    }

    # 预先计算符号序列，避免重复计算
    print("预计算符号序列...")
    tau_max = max(scales)
    symbols_cache = {}
    for scale in tqdm(scales):
        symbols_cache[scale] = {
            "x": symbolize(x_series, tau=scale, m=m, tau_max=tau_max),
            "y": symbolize(y_series, tau=scale, m=m, tau_max=tau_max)
        }

    # 执行多尺度分析
    print("执行多尺度分析...")
    multiscale_results = {}
    for scale in tqdm(scales):
        x_symbol = symbols_cache[scale]["x"]
        y_symbol = symbols_cache[scale]["y"]

        # 初始化分析器
        analyzer = CausalEntropyAnalyzer(x_symbol, y_symbol)
        analyzer.set_params(**analysis_params)
        
        # 执行分析
        results = analyzer.analyze_lag_range(lags_to_test, show_progress=False)

        # 计算各时延上的均值
        multiscale_results[scale] = {
            "avg_bt": {lag: np.mean(res) for lag, res in results["bt_records"].items()},
            "avg_bg": {lag: np.mean(res) for lag, res in results["bg_records"].items()},
            "avg_size_bt": results["avg_size_bt"]
        }

    # ---- 结果可视化 -------------------------------------------------------------------------------

    # 将结果转换为矩阵
    avg_bt_matrix = np.array([[multiscale_results[scale]["avg_bt"][lag] for lag in lags_to_test] for scale in scales])
    avg_bg_matrix = np.array([[multiscale_results[scale]["avg_bg"][lag] for lag in lags_to_test] for scale in scales])

    # 创建子图，高度比例为2:1
    fig, axs = plt.subplots(2, 1, figsize=(3, 5), gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制转移熵热图
    sns.heatmap(
        avg_bt_matrix,
        cmap="Greys",
        ax=axs[0],
        vmin=0,
        vmax=0.3,
        cbar=True,
        cbar_kws={
            "orientation": "horizontal",
            "pad": 0.1,
            "location": "top",
            "label": "entropy",
        }
    )
    # 设置colorbar标题字体大小
    cbar = axs[0].collections[0].colorbar
    cbar.set_label("熵值", fontsize=8)
    
    # 上下翻转热图
    axs[0].invert_yaxis()
    axs[0].set_title("自举平均值", fontsize=8)
    axs[0].set_xticks(np.arange(0, len(lags_to_test), 5) + 0.5)
    axs[0].set_xticklabels(lags_to_test[::5], rotation=90, fontsize=8)
    axs[0].set_yticks(np.arange(0, len(scales), 5) + 0.5)
    axs[0].set_yticklabels(scales[::5], fontsize=8)
    axs[0].set_ylabel("时间尺度（秒）", fontsize=10)

    # 各尺度强度加和
    entropy_sum = np.sum(avg_bt_matrix, axis=0)
    axs[1].plot(lags_to_test, entropy_sum, linewidth=1, color="k")
    axs[1].axvline(0, color="grey", linestyle="-", linewidth=0.5)
    axs[1].axhline(0, color="grey", linestyle="-", linewidth=0.5)
    axs[1].set_xlim(lags_to_test[0], lags_to_test[-1])
    axs[1].set_title("所有尺度熵值之和", fontsize=8)
    axs[1].set_xlabel("滞后（秒）", fontsize=8)
    axs[1].set_ylabel("熵值之和", fontsize=8)

    plt.tight_layout()