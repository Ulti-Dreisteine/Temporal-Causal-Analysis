# -*- coding: utf-8 -*-
"""
Created on 2025/07/24 16:05:28

@File -> s1_multiscale_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 多尺度分析
"""

from itertools import permutations
import seaborn as sns
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.causal_entropy_analyzer.analyzer import CausalEntropyAnalyzer


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


if __name__ == "__main__":
    from script.toy_model.util import gen_samples
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import pandas as pd

    # ---- 生成样本 ---------------------------------------------------------------------------------

    fig_savepath = "fig/xy_series.png"
    x_series, y_series = gen_samples(taus=[10, 10], N=3000, show=True, fig_savepath=fig_savepath)

    # ---- 多尺度分析 --------------------------------------------------------------------------------

    m = 5  # 嵌入维度
    scales = np.arange(1, 100+3, 3)
    lags_to_test = np.arange(-30, 31)

    # 分析参数
    analysis_params = {
        "method": "MIT",
        "size_bt": 200,
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

    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制转移熵热图
    sns.heatmap(avg_bt_matrix, cmap="Reds", ax=axs[0], vmin=0, vmax=0.3)
    axs[0].set_title("Average BT Results")
    axs[0].set_xticks(np.arange(0, len(lags_to_test), 5) + 0.5)
    axs[0].set_xticklabels(lags_to_test[::5], rotation=90)
    axs[0].set_yticks(np.arange(len(scales)) + 0.5)
    axs[0].set_yticklabels(scales)
    axs[0].set_xlabel("Lags")
    axs[0].set_ylabel("Scales")
    
    # 绘制背景熵热图
    sns.heatmap(avg_bg_matrix, cmap="Reds", ax=axs[1], vmin=0, vmax=0.3)
    axs[1].set_title("Average BG Results")
    axs[1].set_xticks(np.arange(0, len(lags_to_test), 5) + 0.5)
    axs[1].set_xticklabels(lags_to_test[::5], rotation=90)
    axs[1].set_yticks(np.arange(len(scales)) + 0.5)
    axs[1].set_yticklabels(scales)
    axs[1].set_xlabel("Lags")
    
    plt.tight_layout()
    plt.savefig("fig/multiscale_analysis.png", dpi=300)
    plt.show()
