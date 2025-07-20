# -*- coding: utf-8 -*-
"""
Created on 2025/07/20 14:48:07

@File -> s1_独立投掷样本分析.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 独立投掷样本分析
"""

from collections import defaultdict
from typing import Callable
import numpy as np
import sys
import os
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.dit_entropy import cal_mi

# 置换排列检验（perm）

def exec_perm_test(x: np.ndarray, y: np.ndarray, N_resample: int, stat_func: Callable):
    """
    执行排列置换检验
    """
    # 计算原始统计量
    stat_obs = stat_func(x, y)

    # 零假设分布
    z = np.append(x, y)
    z_perm = np.array([np.random.permutation(z) for _ in range(N_resample)])
    x_perm = z_perm[:, :len(x)]
    y_perm = z_perm[:, len(x):]
    stat_H0 = np.array([stat_func(x_perm[i], y_perm[i]) for i in range(N_resample)])

    # 计算p值
    p_value = np.mean(stat_H0 >= stat_obs)

    return stat_obs, stat_H0, p_value


# 蒙特卡洛置换检验（MC_perm）

def exec_MC_perm_test(x: np.ndarray, y: np.ndarray, N_resample: int, stat_func: Callable):
    """
    执行蒙特卡洛置换检验
    """
    # 计算原始统计量
    stat_obs = stat_func(x, y)

    # 零假设分布
    x_perm = np.array([np.random.permutation(x) for _ in range(N_resample)])
    stat_H0 = np.array([stat_func(x_perm[i], y) for i in range(N_resample)])

    # 计算p值
    p_value = np.mean(stat_H0 >= stat_obs)

    return stat_obs, stat_H0, p_value


# Bootstrap检验（Bootstrap）

def exec_Bootstrap_test(x: np.ndarray, y: np.ndarray, N_resample: int, stat_func: Callable):
    """
    执行自助法检验
    """
    # 计算原始统计量
    stat_obs = stat_func(x, y)

    # 零假设分布
    x_boot = np.array([np.random.choice(x, size=len(x), replace=True) for _ in range(N_resample)])
    stat_H0 = np.array([stat_func(x_boot[i], y) for i in range(N_resample)])

    # 计算p值
    p_value = np.mean(stat_H0 >= stat_obs)

    return stat_obs, stat_H0, p_value


# Markov链检验（Markov_chain）

def _gen_Markov_surrogate(series: np.ndarray, order: int):
    """
    生成马尔可夫代理序列。根据给定的时间序列，生成一个新的序列，使其尽可能遵循与原序列相同的马尔可夫性质。
    """
    # 统计转移频率
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(series) - order):
        state = tuple(series[i : i + order])
        next_val = series[i + order]
        counts[state][next_val] += 1

    # 生成代理序列
    surrogate = list(series[:order])
    for _ in range(len(series) - order):
        state = tuple(surrogate[-order:])
        try:
            next_vals, counts_ = zip(*counts[state].items())
            probs = np.array(counts_) / sum(counts_)
            surrogate.append(np.random.choice(next_vals, p=probs))
        except:
            # 如果当前状态没有下一个值的统计信息，则随机选择
            surrogate.append(np.random.choice(series))

    surrogate = np.array(surrogate)
    return surrogate


def exec_Markov_test(x: np.ndarray, y: np.ndarray, N_resample: int, stat_func: Callable, Markov_order: int):
    """
    执行排列置换检验
    """
    # 计算原始统计量
    stat_orig = stat_func(x, y)

    # 置换检验
    x_srg_arr = np.array([_gen_Markov_surrogate(x, Markov_order) for _ in range(N_resample)])
    stat_H0 = np.array([stat_func(x_srg_arr[i], y) for i in range(N_resample)])
        
    # 计算p值
    p_value = np.sum(stat_H0 >= stat_orig) / N_resample

    return stat_orig, stat_H0, p_value


def do_indep_text(x, y, method, N_resample, stat_func, **kwargs):
    """
    执行独立性检验
    """
    if method == "perm":
        return exec_perm_test(x, y, N_resample, stat_func)
    elif method == "MC_perm":
        return exec_MC_perm_test(x, y, N_resample, stat_func)
    elif method == "Bootstrap":
        return exec_Bootstrap_test(x, y, N_resample, stat_func)
    elif method == "Markov_chain":
        return exec_Markov_test(x, y, N_resample, stat_func, **kwargs)
    else:
        raise NotImplementedError(f"未知方法: {method}")


if __name__ == "__main__":

    # ---- 载入样本 ---------------------------------------------------------------------------------

    X_samples = np.load("runtime/1阶投掷_X_samples.npy")
    Y_samples = np.load("runtime/1阶投掷_Y_samples.npy")

    N_trials = X_samples.shape[0]

    # ---- 绘制实际关联值和置换关联值的分布 --------------------------------------------------------------

    # 实际的互信息值
    mi_true = [cal_mi(X_samples[i], Y_samples[i]) for i in range(N_trials)]

    # 取某次小样本进行独立性检验
    i = 0
    Xi = X_samples[i]
    Yi = Y_samples[i]

    methods = ["perm", "MC_perm", "Bootstrap", "Markov_chain"]

    stat_H0_dict = {}
    p_value_dict = {}

    N_resample = 1000
    for method in methods:
        if method == "Markov_chain":
            kwargs = {"Markov_order": 1}
        else:
            kwargs = {}

        stat_orig, stat_H0, p_value = do_indep_text(Xi, Yi, method, N_resample, cal_mi, **kwargs)

        stat_H0_dict[method] = stat_H0
        p_value_dict[method] = p_value

    # ---- 绘制实际关联值和零假设采样关联值的分布 --------------------------------------------------------

    range_ = [0, 0.6]

    plt.figure(figsize=(5, 3))
    plt.suptitle("1阶投掷：第1组样本的零假设分布与实际分布对比", fontsize=12)

    colors = ["red", "green", "orange", "purple"]
    for method in methods:
        stat_H0 = stat_H0_dict[method]
        p_value = p_value_dict[method]
        plt.hist(
            stat_H0,
            bins=50,
            range=range_,
            alpha=0.7,
            label=f"{method}, p-value={p_value:.4f}",
            color=colors[methods.index(method)],
            density=True,
            histtype="step",
            linewidth=1,
            linestyle="--"
        )

    plt.hist(
        mi_true, 
        bins=50, 
        range=range_, 
        alpha=0.7, 
        label="MI True", 
        color="blue", 
        density=True, 
        histtype="step", 
        linewidth=1.5, 
        linestyle="-"
    )

    plt.xlabel("互信息值 (MI)")
    plt.ylabel("概率密度")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"fig/1阶投掷_零假设分布对比_试验组_{i}.png", dpi=600)