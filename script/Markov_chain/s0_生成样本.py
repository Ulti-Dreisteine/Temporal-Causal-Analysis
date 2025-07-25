# -*- coding: utf-8 -*-
"""
Created on 2025/07/20 13:40:54

@File -> s0_gen_samples.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 生成各阶马尔可夫链的样本序列
"""

from typing import Union, Literal
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt


class MarkovChainGenerator(object):
    """
    马尔可夫链生成器
    """

    def __init__(self, Pi: Union[np.ndarray, dict]) -> None:
        self.Pi = Pi.copy()

    def gen_a_trial(self, N_steps: int):
        X_series = np.zeros(N_steps, dtype=int)

        if isinstance(self.Pi, np.ndarray):
            Dx = self.Pi.shape[0]  # 状态空间的大小
            for i in range(N_steps):
                X_series[i] = np.random.choice(np.arange(Dx), p=self.Pi[X_series[i - 1]] if i > 0 else self.Pi[np.random.randint(Dx)])
        elif isinstance(self.Pi, dict):
            # 初始化状态
            x_t_k1 = 2
            x_t_k = 1
            for i in range(N_steps):
                # 根据转移概率生成下一个状态
                x_new = np.random.choice([0, 1, 2], p=self.Pi[(x_t_k1, x_t_k)])
                X_series[i] = x_new
                
                # 更新状态
                x_t_k, x_t_k1 = x_new, x_t_k
        else:
            raise TypeError("转移概率矩阵 Pi 必须是 numpy.ndarray 或 dict 类型")
        
        return X_series
    
    def exec_multi_trials(self, N_trials: int, N_steps: int):
        """
        执行多次试验

        Params:
        -------
        N_trials: 试验次数
        N_steps: 每次试验的递推步数
        """
        X_samples = np.zeros((N_trials, N_steps), dtype=int)

        for i in range(N_trials):
            X_samples[i] = self.gen_a_trial(N_steps)

        return X_samples
    
    def show(self, X_samples: np.ndarray):
        """
        显示生成的样本序列
        """
        plt.plot(X_samples[0], color="k", linewidth=1.0)


def gen_Pi(order: Literal[0, 1, 2]):
    match order:
        case 0:
            Pi = np.array([
                    [1 / 6] * 6,
                    [1 / 6] * 6,
                    [1 / 6] * 6,
                    [1 / 6] * 6,
                    [1 / 6] * 6,
                    [1 / 6] * 6,
                ])
        case 1:
            Pi = np.array([
                    [0.5, 0.25, 0, 0, 0, 0.25],
                    [0.25, 0.5, 0.25, 0, 0, 0],
                    [0, 0.25, 0.5, 0.25, 0, 0],
                    [0, 0, 0.25, 0.5, 0.25, 0],
                    [0, 0, 0, 0.25, 0.5, 0.25],
                    [0.25, 0, 0, 0, 0.25, 0.5]
                ])
        case 2:
            Pi = {
                (0, 0): [0.8, 0.1, 0.1],
                (1, 0): [0.3, 0.65, 0.05],
                (2, 0): [0.05, 0.35, 0.6],
                (0, 1): [0.65, 0.3, 0.05],
                (1, 1): [0.1, 0.8, 0.1],
                (2, 1): [0.05, 0.4, 0.55],
                (0, 2): [0.6, 0.05, 0.35],
                (1, 2): [0.1, 0.5, 0.4],
                (2, 2): [0.15, 0.15, 0.7]
            }
    return Pi



if __name__ == "__main__":

    # ---- 0阶马尔可夫链 -----------------------------------------------------------------------------

    Pi = gen_Pi(0)
    N_trials = 1000
    N_steps = 1000

    self = MarkovChainGenerator(Pi)
    X_samples = self.exec_multi_trials(N_trials=N_trials, N_steps=N_steps)
    Y_samples = self.exec_multi_trials(N_trials=N_trials, N_steps=N_steps)

    plt.figure(figsize=(5, 4))
    plt.suptitle("独立投掷样本序列", fontsize=12)

    plt.subplot(2, 1, 1)
    self.show(X_samples)
    plt.ylabel("状态")
    plt.legend(["$X$"], loc="upper right")

    plt.subplot(2, 1, 2)
    self.show(Y_samples)
    plt.xlabel("步数")
    plt.ylabel("状态")
    plt.legend(["$Y$"], loc="upper right")
    plt.tight_layout()

    plt.savefig("fig/s0_独立投掷样本序列.png", dpi=600, bbox_inches="tight")

    # 保存样本
    np.save("runtime/独立投掷_X_samples.npy", X_samples)
    np.save("runtime/独立投掷_Y_samples.npy", Y_samples)

    # ---- 1阶马尔可夫链 -----------------------------------------------------------------------------

    Pi = gen_Pi(1)
    N_trials = 1000
    N_steps = 1000

    self = MarkovChainGenerator(Pi)
    X_samples = self.exec_multi_trials(N_trials=N_trials, N_steps=N_steps)
    Y_samples = self.exec_multi_trials(N_trials=N_trials, N_steps=N_steps)

    plt.figure(figsize=(5, 4))
    plt.suptitle("1阶投掷样本序列", fontsize=12)

    plt.subplot(2, 1, 1)
    self.show(X_samples)
    plt.ylabel("状态")
    plt.legend(["$X$"], loc="upper right")

    plt.subplot(2, 1, 2)
    self.show(Y_samples)
    plt.xlabel("步数")
    plt.ylabel("状态")
    plt.legend(["$Y$"], loc="upper right")
    plt.tight_layout()

    plt.savefig("fig/s0_1阶投掷样本序列.png", dpi=600, bbox_inches="tight")

    # 保存样本
    np.save("runtime/1阶投掷_X_samples.npy", X_samples)
    np.save("runtime/1阶投掷_Y_samples.npy", Y_samples)

    # ---- 2阶马尔可夫链 -----------------------------------------------------------------------------

    Pi = gen_Pi(2)
    N_trials = 1
    N_steps = 30000

    self = MarkovChainGenerator(Pi)
    X_samples = self.exec_multi_trials(N_trials=N_trials, N_steps=N_steps)

    plt.figure(figsize=(5, 2.3))
    plt.suptitle("2阶投掷样本序列", fontsize=12)
    self.show(X_samples)
    plt.xlabel("步数")
    plt.ylabel("状态")
    plt.legend(["$X$"], loc="upper right")
    plt.tight_layout()

    plt.savefig("fig/s0_2阶投掷样本序列.png", dpi=600, bbox_inches="tight")

    # 保存样本
    np.save("runtime/2阶投掷_X_samples.npy", X_samples)


    