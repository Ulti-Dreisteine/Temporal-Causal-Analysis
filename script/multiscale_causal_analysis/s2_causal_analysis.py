# -*- coding: utf-8 -*-
"""
Created on 2025/08/15 13:45:37

@File -> s2_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 因果分析
"""

import seaborn as sns
from tqdm import tqdm
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.causal_entropy_analyzer.analyzer import CausalEntropyAnalyzer

from s0_gen_samples import symbolize, gen_samples

if __name__ == "__main__":
    a = [0.9, 0.99]
    w = [0.06, 1.0]  # 不同的耦合强度
    x_series, y_series = gen_samples(size=10000, lags=[5, 10], a=a, w=w, show=True)

    # ---- 多尺度分析 --------------------------------------------------------------------------------

    m = 3  # 嵌入维度
    scales = np.arange(1, 60 + 3, 3)
    lags_to_test = np.arange(-20, 21)

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
    print("\n执行多尺度分析...")
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
    fig, axs = plt.subplots(2, 1, figsize=(2.5, 5), gridspec_kw={"height_ratios": [3, 1]})
    plt.suptitle(f"多尺度因果分析 \n$a$={a}, $w$={w}", fontsize=10)
    
    # 绘制转移熵热图
    sns.heatmap(
        avg_bt_matrix,
        cmap="Greys",
        ax=axs[0],
        vmin=0,
        vmax=0.1,
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
    axs[0].set_ylabel("时间尺度（样本数）", fontsize=10)

    # 各尺度强度加和
    entropy_sum = np.sum(avg_bt_matrix, axis=0)
    axs[1].plot(lags_to_test, entropy_sum, linewidth=1, color="k")
    axs[1].axvline(0, color="grey", linestyle="-", linewidth=0.5)
    axs[1].axhline(0, color="grey", linestyle="-", linewidth=0.5)
    axs[1].set_xlim(lags_to_test[0], lags_to_test[-1])
    axs[1].set_title("所有尺度熵值之和", fontsize=8)
    axs[1].set_xlabel("滞后（样本数）", fontsize=8)
    axs[1].set_ylabel("熵值之和", fontsize=8)

    plt.tight_layout()

    plt.savefig(f"asset/多尺度因果分析_a_{a}_w_{w}.png", dpi=600)