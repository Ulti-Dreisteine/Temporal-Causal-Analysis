# -*- coding: utf-8 -*-
"""
Created on 2025/07/11 14:53:04

@File -> prob_est.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于KNN的概率估计
"""

from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Union
from scipy.special import gamma
import numpy as np


# ---- 数据标准化 -----------------------------------------------------------------------------------

def normalize(X: np.ndarray):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.copy())
    return X


def _convert_1d_series2int(x: np.ndarray):
    """
    将一维数据按照标签进行编码为连续整数
    """
    
    x = x.flatten()
    x_unique = np.unique(x)

    if len(x_unique) > 100:
        raise RuntimeWarning(
            f"too many labels: {len(x_unique)} for the discrete data")

    x = np.apply_along_axis(lambda x: np.where(
        x_unique == x)[0][0], 1, x.reshape(-1, 1))
    
    return x


def _convert_arr2int(arr: np.ndarray):
    """
    将一维数据按照标签进行编码为连续整数
    """
    
    _, D = arr.shape
    
    for d in range(D):
        arr[:, d] = _convert_1d_series2int(arr[:, d])
    
    return arr.astype(int)


def stdize_values(x: np.ndarray, dtype: str, eps: float = 1e-10) -> np.ndarray:
    """
    数据预处理: 标签值整数化、连续值归一化, 将连续和离散变量样本处理为对应的标准格式用于后续分析
    """
    
    x = x.copy()
    x = x.reshape(x.shape[0], -1)  # (N, ) 转为 (N, 1), (N, D) 转为 (N, D)
    
    if dtype == "c":
        # 连续值加入噪音并归一化
        x += eps * np.random.random_sample(x.shape)
        return normalize(x)
    elif dtype == "d":
        # 将标签值转为连续的整数值
        x = _convert_arr2int(x)
        return x
    else:
        raise ValueError("dtype必须为'c'或'd', 分别表示连续型和离散型变量.")


# ---- K近邻查询 ------------------------------------------------------------------------------------
    
def build_tree(x: np.ndarray, metric: str = "chebyshev") -> Union[BallTree, KDTree]:
    """
    建立近邻查询树. 低维用具有欧式距离特性的KDTree; 高维用具有更一般距离特性的BallTree
    """
    
    x = x.reshape(len(x), -1)
    
    return BallTree(x, metric=metric) if x.shape[1] >= 20 else KDTree(x, metric=metric)


def query_neighbors_dist(tree: Union[BallTree, KDTree], x: Union[np.ndarray, list], k: int) -> np.ndarray:
    """
    求得x样本在tree上的第k个近邻样本
    
    Note:
    -----
    如果tree的样本中包含了x, 则返回结果中也会含有x
    """
    
    x = np.array(x).reshape(1, len(x))
    
    # 返回在x处，tree的样本中距离其最近的k个样本信息
    nbrs_info = tree.query(x, k=k)
    
    return nbrs_info[0][:, -1]


# ---- 空间球体积 -----------------------------------------------------------------------------------

def get_unit_ball_volume(d: int, metric: str = "euclidean") -> Optional[float]:
    """
    d维空间中按照euclidean或chebyshev距离计算所得的单位球体积
    """
    
    if metric == "euclidean":
        return (np.pi ** (d / 2)) / gamma(1 + d / 2)  
    elif metric == "chebyshev":
        return 2 ** d
    else:
        raise ValueError(f"unsupported metric {metric}")


def cal_knn_prob_dens(x: Union[np.ndarray, list], X: Optional[np.ndarray] = None, 
                      tree: Union[BallTree, KDTree, None] = None, k: int = 3, 
                      metric: str = "chebyshev") -> float:
    """
    这段代码实现了KNN方法用于连续型变量的概率密度计算, 包括了以下几个步骤:
    - 构建距离树：使用Scikit-learn的BallTree或KDTree类构建距离树，如果总体样本集不为空，则使用KDTree进行构建。
    - 查询距离：使用距离树查询近邻，获取k nearest neighbors距离。
    - 计算概率密度：使用单位球体体积和k nearest neighbors距离的D次幂进行积分，得到概率密度
    
    Params:
    -------
    x: 待计算位置
    X: 总体样本集
    tree: 使用总体样本建立的距离树
    k: 近邻数
    metric: 距离度量指标
    
    Note:
    -----
    - x和X可以为一维或多维
    - X和tree中必须有一个赋值, 如果有tree则优先使用tree
    """
    
    x = np.array(x).flatten()               # 转为一维序列
    x = stdize_values(x, "c").flatten()     # 标准化
    
    # 构建距离树
    if tree is not None:
        N, D = tree.get_arrays()[0].shape
    elif X is not None:
        X = stdize_values(X.reshape(len(X), -1), "c")
        N, D = X.shape
        tree = build_tree(X)
    else:
        raise ValueError("Either X or tree must be specified.")
    
    k_dist = query_neighbors_dist(tree, x, k)[0]  # type: float
    v = get_unit_ball_volume(D, metric)
    assert v is not None, "单位球体积计算失败, 请检查维度和距离度量指标."
    p = (k / N) / (v * k_dist**D)
    
    return p


if __name__ == "__main__":
    pass