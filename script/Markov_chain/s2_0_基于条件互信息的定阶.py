# -*- coding: utf-8 -*-
"""
Created on 2025/07/20 16:03:20

@File -> s2_2阶投掷样本定阶.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 二阶投掷样本定阶
"""

from collections import defaultdict
from typing import Callable
import seaborn as sns
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.cit_entropy import cal_cmi, kraskov_mi as cal_mi


# **************************************************************************************************
# 通用组件
# **************************************************************************************************

def exec_perm_test(x: np.ndarray, y: np.ndarray, z: np.ndarray, N_resample: int, stat_func: Callable):
    """
    执行排列置换检验
    """
    # 计算原始统计量
    stat_orig = stat_func(x, y, z)

    # 置换检验
    stat_H0 = np.array([stat_func(np.random.permutation(x), y, z) for _ in range(N_resample)])

    # 计算p值
    p_value = np.sum(stat_H0 >= stat_orig) / N_resample

    return stat_orig, stat_H0, p_value


# **************************************************************************************************
# 项目组件
# **************************************************************************************************

def load_and_prepare_samples(name: str, max_order: int) -> np.ndarray:
    """载入并准备样本数据"""
    X_samples = np.load(f"runtime/{name}_X_samples.npy")[0, :1000]
    X_lagged = np.zeros((len(X_samples), max_order + 1), dtype=int)
    
    for order in range(max_order + 1):
        X_lagged[:, order] = np.roll(X_samples, order)
    
    return X_lagged[max_order:, :]


def calculate_statistics(X, Y, Z=None, is_zero_order=False):
    """计算统计量"""
    return cal_mi(X, Y) if is_zero_order else cal_cmi(X, Y, Z)


def perform_bootstrap_test(X, Y, Z, size_bt: int, rounds_bt: int, is_zero_order: bool):
    """执行自举检验"""
    bt_records = np.zeros(rounds_bt)
    bg_records = np.zeros(rounds_bt)
    
    for i in range(rounds_bt):
        idxs = np.unique(np.random.choice(len(X), size=size_bt, replace=True))
        X_bt, Y_bt = X[idxs], Y[idxs]
        Z_bt = Z[idxs] if not is_zero_order else Z
        
        bt_records[i] = calculate_statistics(X_bt, Y_bt, Z_bt, is_zero_order)
        bg_records[i] = calculate_statistics(np.random.permutation(X_bt), Y_bt, Z_bt, is_zero_order)
    
    return bt_records, bg_records


def plot_order_test(ax, bt, bg, order: int):
    """绘制单个阶数的检验结果"""
    bt_clean = bt[np.isfinite(bt)]
    bg_clean = bg[np.isfinite(bg)]
    range_ = (min(np.min(bt_clean), np.min(bg_clean)), max(np.max(bt_clean), np.max(bg_clean)))
    
    bt_mean = np.mean(bt_clean)
    bg_mean = np.mean(bg_clean)
    
    ax.hist(bt_clean, bins=30, alpha=0.4, label="自举分布", color="blue", range=range_, density=True)
    ax.hist(bg_clean, bins=30, alpha=0.4, label="背景分布", color="red", range=range_, density=True)
    
    sns.kdeplot(bt_clean, ax=ax, color="blue", label="自举KDE", linewidth=2)
    sns.kdeplot(bg_clean, ax=ax, color="red", label="背景KDE", linewidth=2)
    
    ax.axvline(bt_mean, color="blue", linestyle="--", alpha=0.8, label="自举均值")
    ax.axvline(bg_mean, color="red", linestyle="--", alpha=0.8, label="背景均值")
    
    ax.set_title(f"阶数 {order + 1}\n自举均值={bt_mean:.3f}, 背景均值={bg_mean:.3f}", fontsize=10)
    ax.set_xlabel("互信息或条件互信息")
    ax.set_ylabel("概率密度")
    ax.legend()


if __name__ == "__main__":
    names = ["独立投掷", "1阶投掷", "2阶投掷"]
    max_order = 6
    size_bt = 100
    rounds_bt = 500

    for name in names:
        X_lagged = load_and_prepare_samples(name, max_order)
        results = defaultdict(dict)
        
        for order in range(max_order):
            X = X_lagged[:, 0]
            Y = X_lagged[:, order + 1]
            Z = X_lagged[:, 1:order + 1] if order > 0 else X_lagged[:, 1]
            is_zero_order = (order == 0)
            
            bt_records, bg_records = perform_bootstrap_test(X, Y, Z, size_bt, rounds_bt, is_zero_order)
            results[order] = {"bt": bt_records, "bg": bg_records}
        
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        for order in range(max_order):
            plot_order_test(axes[order // 3, order % 3], 
                            results[order]["bt"],
                            results[order]["bg"],
                            order)
        
        plt.suptitle(f"{name}样本的马尔可夫链阶数检验", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"fig/{name}样本的马尔可夫链阶数检验.png", dpi=600, bbox_inches="tight")
        plt.show()



