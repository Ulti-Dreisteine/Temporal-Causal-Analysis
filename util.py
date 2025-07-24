from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def show_results(cmi_lag_records, **kwargs):
    """展示时延挖掘结果"""
    lags2test = list(cmi_lag_records["bt_records"].keys())

    try:
        x_col = kwargs["x_col"]
        y_col = kwargs["y_col"]
    except:
        x_col = "x"
        y_col = "y"

    # 如果cmi最大值超过1，则y轴限制为最大值，否则为1
    y_max = max([max(p) for p in cmi_lag_records["bt_records"].values()])
    y_lim = y_max + 0.1 if y_max > 0.6 else 0.6
    
    plt.figure(figsize=(5, 3))
    plt.title(f"CMI of {x_col} $\\rightarrow$ {y_col} with lag")
    
    # 绘制各lag上的CMI和背景CMI散点图
    # for lag in lags2test:
    #     plt.scatter([lag] * rounds_bt, cmi_lag_records["bt_records"][lag], color="b", alpha=0.2, s=1)
    #     plt.scatter([lag] * rounds_bt, cmi_lag_records["bg_records"][lag], color="k", alpha=0.1, s=1)

    # 绘制各自90%置信区间，使用填充区域表示
    cmi_bt_ub = [np.percentile(cmi_lag_records["bt_records"][lag], 80) for lag in lags2test]
    cmi_bg_ub = [np.percentile(cmi_lag_records["bg_records"][lag], 80) for lag in lags2test]
    cmi_bt_lb = [np.percentile(cmi_lag_records["bt_records"][lag], 20) for lag in lags2test]
    cmi_bg_lb = [np.percentile(cmi_lag_records["bg_records"][lag], 20) for lag in lags2test]
    plt.fill_between(lags2test, cmi_bt_ub, cmi_bt_lb, color="b", alpha=0.2)
    plt.fill_between(lags2test, cmi_bg_ub, cmi_bg_lb, color="r", alpha=0.2)

    # 绘制均值变化
    cmi_bt_means = [np.mean(cmi_lag_records["bt_records"][lag]) for lag in lags2test]
    cmi_bg_means = [np.mean(cmi_lag_records["bg_records"][lag]) for lag in lags2test]
    plt.plot(lags2test, cmi_bt_means, linewidth=1.0, color="b", label="bt_mean")
    plt.plot(lags2test, cmi_bg_means, linewidth=1.0, color="r", label="bg_mean")
    
    # 添加基准线
    plt.axvline(0, color="r", label="lag = 0", alpha=0.5, linewidth=0.5)
    plt.axhline(0, color="r", alpha=0.5, linewidth=0.5)
    plt.ylim(-0.2, y_lim)
    plt.xlabel("lag (s)")
    plt.ylabel("CMI")
    plt.legend()
    plt.show()


def gen_Markov_surrogate(series: np.ndarray, order: int):
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