# -*- coding: utf-8 -*-
"""
Created on 2025/07/24 15:21:30

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 玩具模型案例
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.causal_entropy_analyzer.analyzer import CausalEntropyAnalyzer, AnalysisConfig

if __name__ == "__main__":

    from util import show_results
    from script.toy_model.util import gen_samples

    # ---- 生成样本 ---------------------------------------------------------------------------------

    fig_savepath = "fig/xy_series.png"
    x_series, y_series = gen_samples(taus=[5, 10], N=1000, show=True, fig_savepath=fig_savepath)

    # ---- 初始化因果熵分析器 -------------------------------------------------------------------------

    analysis = CausalEntropyAnalyzer(x_series, y_series)

    # 设置分析参数
    analysis_params = {
        "method": "MIT",
        "size_bt": 100,
        "rounds_bt": 50,
        "k": 3
    }
    
    analysis.set_params(**analysis_params)
    print(f"分析参数: {analysis_params}")

    # ---- 执行分析 ----------------------------------------------------------------------------------

    lags_to_test = np.arange(-30, 31)
    
    # 执行分析
    results = analysis.analyze_lag_range(lags_to_test, show_progress=True)

    # ---- 显示结果 ---------------------------------------------------------------------------------

    show_results(results)

    # 显示样本大小变化图
    plt.figure(figsize=(5, 3))
    sample_sizes = list(results["avg_size_bt"].values())
    plt.subplot(2, 1, 1)
    plt.plot(lags_to_test, sample_sizes, "b-", linewidth=1.0)
    plt.axhline(y=AnalysisConfig.MIN_SAMPLE_RATIO * analysis_params["size_bt"], 
                color="r", linestyle="--", alpha=0.7, 
                label=f"阈值 ({AnalysisConfig.MIN_SAMPLE_RATIO * analysis_params['size_bt']:.0f})")
    plt.xlabel("Lag")
    plt.ylabel("平均样本大小")
    plt.title("有效样本大小分析")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    # 显示CMI统计信息
    plt.subplot(2, 1, 2)
    bt_means = [np.nanmean(results["bt_records"][lag]) for lag in lags_to_test]
    bt_stds = [np.nanstd(results["bt_records"][lag]) for lag in lags_to_test]
    
    plt.errorbar(lags_to_test, bt_means, yerr=bt_stds, fmt="o-", markersize=3, linewidth=1.0, capsize=2)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Lag")
    plt.ylabel("CMI (均值 ± 标准差)")
    plt.title("条件互信息分析结果")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()