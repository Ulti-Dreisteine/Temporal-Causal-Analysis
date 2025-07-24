# -*- coding: utf-8 -*-
"""
Created on 2025/07/24 15:21:30

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 玩具模型案例
"""

from itertools import permutations
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.causal_entropy_analyzer.analyzer import CausalEntropyAnalyzer, AnalysisConfig


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

    from util import show_results
    from script.toy_model.util import gen_samples

    # ---- 生成样本 ---------------------------------------------------------------------------------
    fig_savepath = "fig/xy_series.png"
    x_series, y_series = gen_samples(taus=[10, 10], N=5000, show=True, fig_savepath=fig_savepath)

    # ---- 生成符号序列 ------------------------------------------------------------------------------
    m = 5  # 嵌入维度
    scale = 3
    x_symbol = symbolize(x_series, tau=scale, m=m, tau_max=scale)
    y_symbol = symbolize(y_series, tau=scale, m=m, tau_max=scale)

    # 可视化原始序列和符号序列
    N = 500
    fig, axs = plt.subplots(2, 2, figsize=(8, 3))
    axs[0, 0].plot(x_symbol[:N], "b-", linewidth=1.0, label="X Symbol")
    axs[1, 0].plot(x_series[:N], "b-", linewidth=1.0, label="X Series")
    axs[0, 1].plot(y_symbol[:N], "r-", linewidth=1.0, label="Y Symbol")
    axs[1, 1].plot(y_series[:N], "r-", linewidth=1.0, label="Y Series")
    
    for ax in axs.flatten():
        ax.legend()
    plt.tight_layout()
    plt.show()

    # ---- 初始化因果熵分析器 -------------------------------------------------------------------------
    analysis_params = {
        "method": "MIT",
        "size_bt": 200,
        "rounds_bt": 50,
        "k": 3
    }
    
    analysis = CausalEntropyAnalyzer(x_symbol, y_symbol)
    analysis.set_params(**analysis_params)
    print(f"分析参数: {analysis_params}")

    # ---- 执行分析 ----------------------------------------------------------------------------------
    lags_to_test = np.arange(-30, 31)
    results = analysis.analyze_lag_range(lags_to_test, show_progress=True)

    # ---- 显示结果 ---------------------------------------------------------------------------------
    show_results(results)

    # 显示样本大小变化和CMI统计信息
    fig, axs = plt.subplots(2, 1, figsize=(5, 3))
    
    # 样本大小变化图
    sample_sizes = list(results["avg_size_bt"].values())
    min_threshold = AnalysisConfig.MIN_SAMPLE_RATIO * analysis_params["size_bt"]
    
    axs[0].plot(lags_to_test, sample_sizes, "b-", linewidth=1.0)
    axs[0].axhline(y=min_threshold, color="r", linestyle="--", alpha=0.7, 
                  label=f"阈值 ({min_threshold:.0f})")
    axs[0].set_xlabel("Lag")
    axs[0].set_ylabel("平均样本大小")
    axs[0].set_title("有效样本大小分析")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)
    
    # CMI统计信息
    bt_means = [np.nanmean(results["bt_records"][lag]) for lag in lags_to_test]
    bt_stds = [np.nanstd(results["bt_records"][lag]) for lag in lags_to_test]
    
    axs[1].errorbar(lags_to_test, bt_means, yerr=bt_stds, fmt="o-", markersize=3, linewidth=1.0, capsize=2)
    axs[1].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axs[1].axvline(x=0, color="r", linestyle="--", alpha=0.5)
    axs[1].set_xlabel("Lag")
    axs[1].set_ylabel("CMI (均值 ± 标准差)")
    axs[1].set_title("条件互信息分析结果")
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
