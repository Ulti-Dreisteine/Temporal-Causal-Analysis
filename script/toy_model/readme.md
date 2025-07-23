#### 一、样本生成

模型公式如下：

$$
\begin{align*}
    \left\{
        \begin{align*}
            X_t & = a X_{t-1} + 0.01\epsilon_t \\
            Y_t & = b Y_{t-1} + c X_{t-{\tau_a}} + d X_{t-\tau_b} + 0.01\eta_t
        \end{align*}
    \right.
    \tag{1}
\end{align*}
$$

参数和噪声设置如下：

$$
\begin{aligned}
    &\tau_a = 0, \tau_b = 10 \\
    &a = 0.7, b=0.2, c=0.5, d=0.5 \\
    &\epsilon_t \sim N(0, 0.1), \eta_t \sim N(0, 0.1)
\end{aligned}
$$

图像如下：

<p align="center">
    <img src="fig/xy_series.png" alt="toy_model" width="400"/>
</p>


> 注意：
> 从式（1）可见，$X$ 对 $Y$ 具有时延分别为 $\tau_a=0$ 的即时作用 和 $\tau_b=10$ 的延迟作用。本文希望仅通过数据识别准确出这两种因果作用。  

#### 二、预计算

分别计算 $X$ 和 $Y$ 各自的时间常数，公式为：

$$
\begin{aligned}  
    \tau_X = \argmin_{\tau} \rho(x_t, x_{t-\tau}) < e \tag{2}\\
    \tau_Y = \argmin_{\tau} \rho(y_t, y_{t-\tau}) < e\\
\end{aligned}
$$

其中 $\rho$ 为相关系数，$e$ 为阈值。计算结果如下：

<p align="center">
    <img src="fig/tau_x.png" alt="tau_xy" width="350"/>
</p>

<p align="center">
    <img src="fig/tau_y.png" alt="tau_xy" width="350"/>
</p>

从时间常数结果来看，两个变量具有一定的时序性。