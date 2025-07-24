# -*- coding: utf-8 -*-
"""
Created on 2025/07/24 09:53:09

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 玩具模型案例
"""

from collections import defaultdict
from typing import Literal, Dict, Any, Tuple
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.cit_entropy import cal_cmi
from core.resampler import Resampler

SUPPORTED_METHODS = Literal["TE", "MIT"]


class CausalEntropyAnalysis:
    """基于瞬时信息传输（Momentary Information Transfer, MIT）识别变量之间的时延"""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x).flatten()
        self.y = np.asarray(y).flatten()

        # tau_x和tau_y为固定常数，提升MIT检测准确性
        self.tau_x = 1
        self.tau_y = 1

        if len(self.x) != len(self.y):
            raise ValueError(f"len(x) = {len(self.x)} != len(y) = {len(self.y)}")
        self.N = len(self.x)
        
        self.method = None
        self.size_bt = None
        self.rounds_bt = None
        self.k = None
        self.alpha = None

    def set_params(self, rounds_bt: int, size_bt: int, method: SUPPORTED_METHODS, **kwargs: Any) -> None:
        """
        设置计算参数

        Params:
        -------
        rounds_bt: int,  # 迭代轮数
        size_bt: int,    # 每轮样本大小
        method: str      # 计算方法
        """
        self.rounds_bt = rounds_bt
        self.size_bt = size_bt
        self.method = method

        # Kraskov互信息参数
        self.k = kwargs.get("k", 3)
        self.alpha = kwargs.get("alpha", 0.0)

    def __build_lagged_cutoff_samples(self, lag: int) -> Dict[str, np.ndarray]:
        """
        构建时延样本并截断

        Params:
        -------
        lag: 时延，以样本单位计，正值表示X到Y的时延，负值表示Y到X的时延
        """
        # 预先计算所有需要的滚动版本，避免多次复制
        x_rolled = {
            'lag': np.roll(self.x, lag),
            'lag_tau_x': np.roll(self.x, lag + self.tau_x)
        }
        y_rolled = {
            't': self.y,
            'tau_y': np.roll(self.y, self.tau_y)
        }

        # 计算截断区间索引，确保所有数组长度一致
        indices = [0, lag, lag + self.tau_x, self.tau_y]
        min_idx = min(indices)
        max_idx = max(indices)

        start = max_idx
        end = self.N + min_idx

        lagged_samples = {
            "x_t_lag": x_rolled['lag'][start:end],
            "x_t_lag_tau_x": x_rolled['lag_tau_x'][start:end],
            "y_t": y_rolled['t'][start:end],
            "y_t_tau_y": y_rolled['tau_y'][start:end]
        }
        return lagged_samples

    def cal_entropy(self, samples_bt: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        计算单个延迟处的熵

        Params:
        -------
        samples_bt: dict, 包含采样数据
        """
        x_t_lag_bt = samples_bt["x_t_lag"]
        x_t_lag_tau_x_bt = samples_bt["x_t_lag_tau_x"]
        y_t_bt = samples_bt["y_t"]
        y_t_tau_y_bt = samples_bt["y_t_tau_y"]

        # 构造替代样本，打乱x_t_lag和x_t_lag_tau_x的对应关系
        idxs_bg = np.random.permutation(len(x_t_lag_bt))
        x_t_lag_bg = x_t_lag_bt[idxs_bg]
        x_t_lag_tau_x_bg = x_t_lag_tau_x_bt[idxs_bg]

        if self.method == "TE":
            z_bt = y_t_tau_y_bt[:, None]
            z_bg = y_t_tau_y_bt[:, None]
        elif self.method == "MIT":
            z_bt = np.column_stack((x_t_lag_tau_x_bt, y_t_tau_y_bt))
            z_bg = np.column_stack((x_t_lag_tau_x_bg, y_t_tau_y_bt))
        else:
            raise ValueError(f"不支持的method: {self.method}")

        cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, z_bt, k=self.k, alpha=self.alpha)
        cmi_bg = cal_cmi(x_t_lag_bg, y_t_bt, z_bg, k=self.k, alpha=self.alpha)

        return cmi_bt, cmi_bg

    def cal_lagged_entropy(self, lag: int):
        """
        计算时延熵

        Params:
        -------
        lag: 时延，以样本单位计，正值表示X到Y的时延，负值表示Y到X的时延
        """
        lagged_samples = self.__build_lagged_cutoff_samples(lag)

        sampler = Resampler()
        sampler.set_samples(lagged_samples)

        # 批量重采样
        samples_bt_list = [sampler.resample(N=self.size_bt) for _ in range(self.rounds_bt)]

        avg_size_bt = np.mean([len(samples_bt["y_t"]) for samples_bt in samples_bt_list])

        # 批量计算熵指标
        cmi_results = [self.cal_entropy(samples_bt) for samples_bt in samples_bt_list]
        bt_records = np.array([cmi_bt for cmi_bt, _ in cmi_results])
        bg_records = np.array([cmi_bg for _, cmi_bg in cmi_results])

        return bt_records, bg_records, avg_size_bt


if __name__ == "__main__":
    from util import show_results
    from script.toy_model.util import gen_samples

    # ---- 生成样本 ---------------------------------------------------------------------------------

    fig_savepath = "fig/xy_series.png"
    x_series, y_series = gen_samples(taus=[0, 10], N=2000, show=True, fig_savepath=fig_savepath)

    # ---- 时延熵分析 -----------------------------------------------------------------------------

    analysis = CausalEntropyAnalysis(x_series, y_series)

    method = "MIT"
    size_bt = 300
    rounds_bt = 50

    analysis.set_params(rounds_bt=rounds_bt,
                        size_bt=size_bt,
                        method=method,
                        k=3)

    # ---- 测试 -----------------------------------------------------------------------------------

    lags_to_test = np.arange(-30, 31)
    
    cmi_lag_records = defaultdict(dict)

    for lag in lags_to_test:
        print(f"\rProcessing lag: {lag}", end="", flush=True)

        bt_records, bg_records, avg_size_bt = analysis.cal_lagged_entropy(lag=lag)

        cmi_lag_records["bt_records"][lag] = bt_records
        cmi_lag_records["bg_records"][lag] = bg_records
        cmi_lag_records["avg_size_bt"][lag] = avg_size_bt

    # 基线校正
    base_value = np.mean([bg.mean() for bg in cmi_lag_records["bg_records"].values()])
    for lag in cmi_lag_records["bt_records"]:
        cmi_lag_records["bt_records"][lag] -= base_value
        cmi_lag_records["bg_records"][lag] -= base_value

    show_results(cmi_lag_records)

    avg_size_bt = np.mean(list(cmi_lag_records["avg_size_bt"].values()))
    if avg_size_bt < 0.9 * size_bt:
        print("\n警告: 平均有效样本量低于设定阈值，可能影响结果的可靠性！考虑增加样本量或减小size_bt值")

    plt.figure(figsize=(5, 3))
    plt.plot(cmi_lag_records["avg_size_bt"].values(), label="Average Sample Size (bt)")
    plt.xlabel("Lag")
    plt.ylabel("Average Sample Size")
    plt.legend()
    plt.show()