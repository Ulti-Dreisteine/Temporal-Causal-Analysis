# -*- coding: utf-8 -*-
"""
Created on 2025/07/24 15:44:43

@File -> causal_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 因果熵分析模块
"""

from collections import defaultdict
from typing import Literal, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import warnings
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from core.cit_entropy import cal_cmi
from core.resampler import Resampler

# 类型定义
SUPPORTED_METHODS = Literal["TE", "MIT"]

# 常量定义
@dataclass(frozen=True)
class AnalysisConfig:
    """分析配置参数"""
    DEFAULT_TAU_X: int = 1
    DEFAULT_TAU_Y: int = 1
    DEFAULT_K: int = 3
    DEFAULT_ALPHA: float = 0.0
    MIN_SAMPLE_RATIO: float = 0.9
    BASELINE_CORRECTION: bool = True


class CausalEntropyAnalyzer:
    """
    基于瞬时信息传输（Momentary Information Transfer, MIT）识别变量之间的时延
    
    该类实现了基于条件互信息的因果关系分析，支持传输熵(TE)和瞬时信息传输(MIT)两种方法。
    
    Attributes:
        x (np.ndarray): 一维输入时间序列X
        y (np.ndarray): 一维输入时间序列Y
        tau_x (int): X序列的时延参数
        tau_y (int): Y序列的时延参数
        N (int): 时间序列长度
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, 
                 tau_x: int = AnalysisConfig.DEFAULT_TAU_X, 
                 tau_y: int = AnalysisConfig.DEFAULT_TAU_Y) -> None:
        """
        初始化因果熵分析器
        
        Args:
            x: 时间序列X，shape为(N,)
            y: 时间序列Y，shape为(N,)
            tau_x: X序列的时延参数，默认为1
            tau_y: Y序列的时延参数，默认为1
            
        Raises:
            ValueError: 当输入序列长度不匹配时
            TypeError: 当输入不是数值类型时
        """
        # 输入验证和预处理
        self.x = self._validate_and_preprocess_input(x.flatten(), "x")
        self.y = self._validate_and_preprocess_input(y.flatten(), "y")
        
        if len(self.x) != len(self.y):
            raise ValueError(f"序列长度不匹配: len(x)={len(self.x)} != len(y)={len(self.y)}")
        
        self.N = len(self.x)
        self.tau_x = self._validate_tau(tau_x, "tau_x")
        self.tau_y = self._validate_tau(tau_y, "tau_y")
        
        # 分析参数
        self._reset_params()
        
    def _validate_and_preprocess_input(self, data: np.ndarray, name: str) -> np.ndarray:
        """验证和预处理输入数据"""
        try:
            data = np.asarray(data, dtype=np.float64).flatten()
        except (ValueError, TypeError) as e:
            raise TypeError(f"无法将{name}转换为数值数组: {e}")
            
        if len(data) == 0:
            raise ValueError(f"{name}不能为空")
            
        if not np.isfinite(data).all():
            warnings.warn(f"{name}包含非有限值(NaN或Inf)，将被移除")
            data = data[np.isfinite(data)]
            
        return data
    
    def _validate_tau(self, tau: int, name: str) -> int:
        """验证时延参数"""
        if not isinstance(tau, int) or tau < 0:
            raise ValueError(f"{name}必须是非负整数，得到: {tau}")
        if tau >= self.N:
            raise ValueError(f"{name}={tau}不能大于等于序列长度{self.N}")
        return tau
    
    def _reset_params(self):
        """重置分析参数"""
        self.method: Optional[SUPPORTED_METHODS] = None
        self.size_bt: Optional[int] = None
        self.rounds_bt: Optional[int] = None
        self.k: int = AnalysisConfig.DEFAULT_K
        self.alpha: float = AnalysisConfig.DEFAULT_ALPHA

    def set_params(self, rounds_bt: int, size_bt: int, method: SUPPORTED_METHODS, 
                   k: int = AnalysisConfig.DEFAULT_K, 
                   alpha: float = AnalysisConfig.DEFAULT_ALPHA) -> None:
        """
        设置计算参数
        
        Args:
            rounds_bt: 迭代轮数，必须为正整数
            size_bt: 每轮样本大小，必须为正整数且小于序列长度
            method: 计算方法，"TE"或"MIT"
            k: Kraskov互信息的k近邻参数
            alpha: 正则化参数
            
        Raises:
            ValueError: 当参数取值不合理时
        """
        # 参数验证
        if not isinstance(rounds_bt, int) or rounds_bt <= 0:
            raise ValueError(f"rounds_bt必须是正整数，得到: {rounds_bt}")
        if not isinstance(size_bt, int) or size_bt <= 0:
            raise ValueError(f"size_bt必须是正整数，得到: {size_bt}")
        if size_bt >= self.N:
            raise ValueError(f"size_bt={size_bt}不能大于等于序列长度{self.N}")
        if method not in ["TE", "MIT"]:
            raise ValueError(f"不支持的方法: {method}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k必须是正整数，得到: {k}")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise ValueError(f"alpha必须是非负数，得到: {alpha}")
            
        self.rounds_bt = rounds_bt
        self.size_bt = size_bt
        self.method = method
        self.k = k
        self.alpha = alpha

    def _build_lagged_cutoff_samples(self, lag: int) -> Dict[str, np.ndarray]:
        """
        构建时延样本并截断
        
        Args:
            lag: 时延，正值表示X到Y的时延，负值表示Y到X的时延
            
        Returns:
            包含时延样本的字典
        """
        if abs(lag) >= self.N:
            raise ValueError(f"时延绝对值{abs(lag)}不能大于等于序列长度{self.N}")
        
        # 预计算滚动版本，使用更高效的方式
        x_lag = np.roll(self.x, lag)
        x_lag_tau_x = np.roll(self.x, lag + self.tau_x)
        y_tau_y = np.roll(self.y, self.tau_y)

        # 计算有效区间，避免边界效应
        indices = [0, lag, lag + self.tau_x, self.tau_y]
        min_idx = min(indices)
        max_idx = max(indices)
        
        start = max_idx
        end = self.N + min_idx
        
        if end <= start:
            raise ValueError(f"在lag={lag}时，有效样本长度为0")

        return {
            "x_t_lag": x_lag[start:end],
            "x_t_lag_tau_x": x_lag_tau_x[start:end],
            "y_t": self.y[start:end],
            "y_t_tau_y": y_tau_y[start:end]
        }
    
    def _calculate_entropy_pair(self, samples_bt: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        计算单个延迟处的熵对（原始和背景）
        
        Args:
            samples_bt: 包含采样数据的字典
            
        Returns:
            (原始CMI, 背景CMI)的元组
        """
        x_t_lag_bt = samples_bt["x_t_lag"]
        x_t_lag_tau_x_bt = samples_bt["x_t_lag_tau_x"]
        y_t_bt = samples_bt["y_t"]
        y_t_tau_y_bt = samples_bt["y_t_tau_y"]

        # 构造替代样本：打乱x序列的对应关系
        n_samples = len(x_t_lag_bt)
        perm_indices = np.random.permutation(n_samples)
        x_t_lag_bg = x_t_lag_bt[perm_indices]
        x_t_lag_tau_x_bg = x_t_lag_tau_x_bt[perm_indices]

        # 根据方法构造条件变量Z
        if self.method == "TE":
            z_bt = y_t_tau_y_bt.reshape(-1, 1)
            z_bg = y_t_tau_y_bt.reshape(-1, 1)
        elif self.method == "MIT":
            z_bt = np.column_stack((x_t_lag_tau_x_bt, y_t_tau_y_bt))
            z_bg = np.column_stack((x_t_lag_tau_x_bg, y_t_tau_y_bt))
        else:
            raise ValueError(f"不支持的方法: {self.method}")

        # 计算条件互信息
        try:
            cmi_bt = cal_cmi(x_t_lag_bt, y_t_bt, z_bt, k=self.k, alpha=self.alpha)
            cmi_bg = cal_cmi(x_t_lag_bg, y_t_bt, z_bg, k=self.k, alpha=self.alpha)
        except Exception as e:
            warnings.warn(f"计算CMI时出错: {e}")
            return np.nan, np.nan

        return cmi_bt, cmi_bg
    
    def calculate_lagged_entropy(self, lag: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算指定时延的熵分析结果
        
        Args:
            lag: 时延值
            
        Returns:
            (原始CMI序列, 背景CMI序列, 平均有效样本大小)的元组
            
        Raises:
            ValueError: 当参数未设置或时延值无效时
        """
        if self.method is None:
            raise ValueError("请先调用set_params()设置参数")
            
        # 构建时延样本
        try:
            lagged_samples = self._build_lagged_cutoff_samples(lag)
        except ValueError as e:
            warnings.warn(f"构建时延样本失败 (lag={lag}): {e}")
            return np.full(self.rounds_bt, np.nan), np.full(self.rounds_bt, np.nan), 0.0

        # 设置重采样器
        sampler = Resampler()
        sampler.set_samples(lagged_samples)

        # 批量重采样和计算
        bt_records = []
        bg_records = []
        sample_sizes = []

        for _ in range(self.rounds_bt):
            try:
                samples_bt = sampler.resample(N=self.size_bt)
                sample_sizes.append(len(samples_bt["y_t"]))
                
                cmi_bt, cmi_bg = self._calculate_entropy_pair(samples_bt)
                bt_records.append(cmi_bt)
                bg_records.append(cmi_bg)
                
            except Exception as e:
                warnings.warn(f"重采样或计算熵时出错: {e}")
                bt_records.append(np.nan)
                bg_records.append(np.nan)
                sample_sizes.append(0)

        avg_sample_size = np.mean(sample_sizes).astype(float) if sample_sizes else 0.0
        
        return np.array(bt_records), np.array(bg_records), avg_sample_size
    
    def analyze_lag_range(self, lags_to_test: np.ndarray, 
                         show_progress: bool = True) -> Dict[str, Dict[int, Any]]:
        """
        分析一系列时延值的因果关系
        
        Args:
            lags_to_test: 要测试的时延值数组
            show_progress: 是否显示进度条
            
        Returns:
            包含分析结果的字典
        """
        if self.method is None:
            raise ValueError("请先调用set_params()设置参数")
            
        results = defaultdict(dict)
        
        # 使用进度条
        iterator = tqdm(lags_to_test, desc="分析时延") if show_progress else lags_to_test
        
        for lag in iterator:
            bt_records, bg_records, avg_size_bt = self.calculate_lagged_entropy(lag)
            
            results["bt_records"][lag] = bt_records
            results["bg_records"][lag] = bg_records
            results["avg_size_bt"][lag] = avg_size_bt

        # 基线校正
        if AnalysisConfig.BASELINE_CORRECTION:
            self._apply_baseline_correction(results)
            
        # 检查样本大小
        self._check_sample_size_warning(results)
        
        return dict(results)
    
    def _apply_baseline_correction(self, results: Dict[str, Dict[int, Any]]) -> None:
        """应用基线校正"""
        # 计算背景CMI的全局平均值作为基线
        bg_means = []
        for bg_records in results["bg_records"].values():
            if not np.isnan(bg_records).all():
                bg_means.append(np.nanmean(bg_records))
        
        if bg_means:
            base_value = np.mean(bg_means)
            
            # 应用基线校正
            for lag in results["bt_records"]:
                results["bt_records"][lag] = results["bt_records"][lag] - base_value
                results["bg_records"][lag] = results["bg_records"][lag] - base_value
    
    def _check_sample_size_warning(self, results: Dict[str, Dict[int, Any]]) -> None:
        """检查并警告样本大小不足"""
        avg_sizes = list(results["avg_size_bt"].values())
        overall_avg = np.mean([s for s in avg_sizes if s > 0])
        
        if self.size_bt is None:
            warnings.warn("未设置size_bt参数，无法检查样本大小是否充足")
        elif overall_avg < AnalysisConfig.MIN_SAMPLE_RATIO * self.size_bt:
            warnings.warn(
            f"平均有效样本量 ({overall_avg:.1f}) 低于设定阈值 "
            f"({AnalysisConfig.MIN_SAMPLE_RATIO * self.size_bt:.1f})，"
            "可能影响结果可靠性！建议增加样本量或减小size_bt值"
            )
        else:
            print(f"平均有效样本量: {overall_avg:.1f}，满足要求")