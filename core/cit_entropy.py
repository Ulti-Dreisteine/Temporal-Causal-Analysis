# -*- coding: utf-8 -*-
"""
Created on 2025/03/17 08:57:00

@File -> MI_kroskov.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Kraskov互信息
"""

from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import MinMaxScaler
from scipy.special import digamma
import numpy.linalg as la
from typing import Union
from numpy import log
import numpy as np


def _build_tree(points) -> Union[BallTree, KDTree]:
    """根据点的维度选择合适的树结构"""
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")


def _query_neighbors(tree, x, k):
    """查询k近邻的距离"""
    return tree.query(x, k=k + 1)[0][:, k]


def _count_neighbors(tree, x, r):
    """查询半径r内的邻居数量"""
    return tree.query_radius(x, r, count_only=True)


def _avgdigamma(points, dvec):
    """
    在边际空间中找到某个半径内的邻居数量
    返回<psi(nx)>的期望值
    """
    tree = _build_tree(points)
    dvec = dvec - 1e-15
    num_points = _count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def lnc_correction(tree, points, k, alpha):
    """
    局部非正态校正 (local Non-normality Correction)
    """
    
    e = 0
    n_sample = points.shape[0]
    
    for point in points:
        # 在联合空间中找到k近邻，p=inf表示最大范数
        knn = tree.query(point[None, :], k=k+1, return_distance=False)[0]
        knn_points = points[knn]
        
        # 减去k近邻点的均值
        knn_points = knn_points - knn_points[0]
        
        # 计算k近邻点的协方差矩阵，获得特征向量
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        
        # 使用特征向量计算PCA边界框
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        
        # 计算原始框的体积
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # 执行局部非均匀性检查并更新校正项
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    
    return e


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


def stdize_values(x: np.ndarray, dtype: str, noise_scale: float = 1e-6) -> np.ndarray:
    """
    数据预处理: 标签值整数化、连续值归一化, 将连续和离散变量样本处理为对应的标准格式用于后续分析
    """
    
    x = x.copy()
    x = x.reshape(x.shape[0], -1)  # (N, ) 转为 (N, 1), (N, D) 转为 (N, D)
    
    if dtype == "c":
        # <<----------------------------------------------------------------------------------------
        # 按照标准差加入噪音
        # x_std = np.std(x, axis=0)
        # x += noise_scale * x_std * np.random.random_sample(x.shape)
        # <<----------------------------------------------------------------------------------------
        x = normalize(x)
        x += noise_scale * np.random.random_sample(x.shape)
        # >>----------------------------------------------------------------------------------------    
        return x
    elif dtype == "d":
        # 将标签值转为连续的整数值
        x = _convert_arr2int(x)
        return x
    else:
        raise ValueError(f"Invalid dtype {dtype}")


def kraskov_mi(x, y, z = None, k = 3, base = np.e, alpha = 0, noise_scale: float = 1e-6) -> float:
    """
    计算x和y的互信息（如果z不为None，则为条件互信息）
    如果x是一维标量且我们有四个样本，x和y应该是向量的列表，例如x = [[1.3], [3.7], [5.1], [2.4]]
    """
    x = stdize_values(np.array(x), "c", noise_scale)
    y = stdize_values(np.array(y), "c", noise_scale)

    if z is not None:
        z = stdize_values(np.array(z), "c", noise_scale)
    
    # 数组应该具有相同的长度
    assert len(x) == len(y)
    
    # 设置 k 小于样本数 - 1
    assert k <= len(x) - 1
    
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)

    points = [x, y]
    
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    
    points = np.hstack(points)
    
    # 在联合空间中找到最近的邻居，p=inf 表示最大范数
    tree = _build_tree(points)
    dvec = _query_neighbors(tree, points, k)
    
    if z is None:
        a, b, c, d = _avgdigamma(x, dvec), _avgdigamma(
            y, dvec), digamma(k), digamma(len(x))
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = _avgdigamma(xz, dvec), _avgdigamma(
            yz, dvec), _avgdigamma(z, dvec), digamma(k)
        
    return (-a - b + c + d) / log(base)


def SU(x: np.ndarray, y: np.ndarray, noise_scale: float = 1e-10) -> float:
    """对称不确定性系数"""
    a = kraskov_mi(x, y, noise_scale=noise_scale)
    b = kraskov_mi(x, x, noise_scale=noise_scale) + kraskov_mi(y, y, noise_scale=noise_scale)
    return 2 * a / b


def cal_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, **kwargs) -> float:
    """
    计算条件互信息：
        I(X;Y|Z) = I(X;Y) + I(X,Y;Z) - I(X;Z) - I(Y;Z)    
    """
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    z = np.array(z).flatten()
    
    # <<--------------------------------------------------------------------------------------------
    # mi_xy_z = kraskov_mi(np.c_[x, y], z, **kwargs)
    # mi_x_y = kraskov_mi(x, y, **kwargs)
    # mi_x_z = kraskov_mi(x, z, **kwargs)
    # mi_y_z = kraskov_mi(y, z, **kwargs)

    # cmi = mi_xy_z + mi_x_y - mi_x_z - mi_y_z
    # <<--------------------------------------------------------------------------------------------
    cmi = kraskov_mi(x, y, z, alpha=0.1, **kwargs)
    # >>--------------------------------------------------------------------------------------------
    
    return cmi