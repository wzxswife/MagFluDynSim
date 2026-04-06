# 行波方程与Burgers方程——物理原理与数值方法

## 1. 概述

本文档对流体力学中两个基本的偏微分方程（PDE）及其数值解法进行了详细分析：

1. **行波方程**：$u_t + (\frac{1}{2}u^2)_x = 0$
2. **Burgers方程**：$u_t + u \cdot u_x = 0$

---

## 2. 行波方程

### 2.1 物理意义

行波方程：

$$u_t + (\frac{1}{2}u^2)_x = 0$$

是一阶非线性双曲守恒律方程，可改写为守恒形式：

$$u_t + f(u)_x = 0$$

其中通量函数为 $f(u) = \frac{1}{2}u^2$。

**物理意义解释：**

该方程描述了非线性波的传播，其中波速依赖于波幅本身。主要物理特征如下：

1. **非线性**：波速等于 $u$，意味着振幅越大的波传播越快
2. **波前陡峭化**：由于快波会追上慢波，波剖面逐渐变得陡峭
3. **激波形成**：最终波会发展出间断（激波）

### 2.2 特征线法

特征线法给出了解析解：

$$\frac{dx}{dt} = u, \quad \frac{du}{dt} = 0$$

这意味着：
- **特征线**在 x-t 平面中是直线
- **u 沿每条特征线保持常数**
- 解由隐式方程给出：$u(x,t) = u_0(\xi)$，其中 $x = \xi + u_0(\xi)t$

### 2.3 Lax-Friedrichs 数值方法

Lax-Friedrichs 格式是一阶精度的有限差分方法：

$$u_i^{n+1} = \frac{1}{2}(u_{i+1}^n + u_{i-1}^n) - \frac{\Delta t}{2\Delta x}(F_{i+1}^n - F_{i-1}^n)$$

其中 $F = \frac{1}{2}u^2$。

**主要特点：**
- **条件稳定**：需要满足 CFL 条件 $\Delta t \leq \frac{\Delta x}{\max|u|}$
- **一阶精度**：数值耗散（人工粘性）
- **简单鲁棒**：广泛用于初始测试

---

## 3. Burgers方程

### 3.1 物理意义

无粘性 Burgers 方程：

$$u_t + (0.5 \cdot u^2)_x = 0$$

是流体力学和平凡流理论中的基本模型方程。

**物理意义解释：**

1. **平流方程**：物理量 $u$ 由其自身平流
2. **非线性平流**：速度场依赖于因变量
3. **激波形成**：当特征线相交时，产生间断（激波）
4. **应用领域**：
   - 交通流（车辆密度传播）
   - 气体动力学（Euler 方程的简化模型）
   - 水波（浅水方程）

### 3.2 特征线法（解析解）

对于 Burgers 方程，特征线满足：

$$\frac{dx}{dt} = u, \quad \frac{du}{dt} = 0$$

**求解过程：**
1. 初始点 $\xi$ 以速度 $u_0(\xi)$ 传播
2. 特征线：$x = \xi + u_0(\xi)t$
3. 解：$u(x,t) = u_0(\xi)$，其中 $\xi$ 满足隐式方程

**牛顿迭代求解：**
由于特征线方程 $x = \xi + u_0(\xi)t$ 是隐式的，需要使用牛顿迭代法求解：

$$f(\xi) = \xi + u_0(\xi)t - x = 0$$

$$f'(\xi) = 1 + u_0'(\xi)t$$

### 3.3 数值方法对比

本项目实现了三种经典的有限差分方法：

| 方法 | 类型 | 精度 | 特点 |
|------|------|------|------|
| **迎风格式 (Upwind)** | Godunov类 | 一阶 | 简单稳定，人工粘性大 |
| **Minmod格式** | TVD/MUSCL | 二阶 | 激波捕捉好，无振荡 |
| **Lax-Wendroff格式** | 泰勒展开 | 二阶 | 高精度，可能有振荡 |

#### 3.3.1 迎风格式 (Upwind Scheme)

一阶迎风格式使用 Rusanov 通量：

$$F_{i+1/2} = \frac{1}{2}(F_L + F_R) - \frac{\alpha}{2}(u_R - u_L)$$

其中 $\alpha = \max(|u_L|, |u_R|)$，$F = 0.5u^2$。

**更新公式：**

$$u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x}(F_{i+1/2} - F_{i-1/2})$$

**特点：**
- 条件稳定：$\text{CFL} = |u|_{\max}\Delta t/\Delta x \leq 0.45$
- 一阶精度
- 数值耗散明显，激波模糊

#### 3.3.2 Minmod格式 (TVD)

Minmod 格式是 MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws) 框架下的 TVD (Total Variation Diminishing) 方法：

**重构步骤：**
使用 minmod 限制器计算界面值：

$$\sigma = \text{minmod}\left(\frac{u_i - u_{i-1}}{\Delta x}, \frac{u_{i+1} - u_i}{\Delta x}\right)$$

$$u_L = u_i - \frac{1}{2}\sigma\Delta x, \quad u_R = u_i + \frac{1}{2}\sigma\Delta x$$

**minmod 函数定义：**

$$\text{minmod}(a, b) = \begin{cases}
0, & \text{if } ab < 0 \\
\text{sign}(a)\min(|a|, |b|), & \text{otherwise}
\end{cases}$$

**特点：**
- 二阶精度
- 自动抑制振荡
- 激波附近自动降阶为一阶

#### 3.3.3 Lax-Wendroff格式

Lax-Wendroff 格式基于泰勒展开，二阶精度：

$$u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x}(F_{i+1/2} - F_{i-1/2})$$

使用 Rusanov 通量计算 $F_{i+1/2}$。

**特点：**
- 二阶精度（时间和空间）
- 无人工粘性
- 可能有 Gibbs 振荡

---

## 4. 初始条件与边界条件

### 4.1 初始条件：高斯型

$$u(x, 0) = e^{-x^2}$$

**特点：**
- 光滑连续函数
- 在 $x=0$ 处取得最大值 $u_{\max} = 1$
- 快速衰减至零

### 4.2 边界条件

$$u(0, t) = u(2, t) = 0$$

 Dirichlet 边界条件，两端固定为 0。

### 4.3 计算参数

| 参数 | 值 | 含义 |
|------|-----|------|
| 区域长度 | $[0, 2]$ | 空间计算域 |
| 空间点数 N | 200 | 网格分辨率 $\Delta x = 0.01$ |
| 时间步长 dt | 0.0005 | 自适应调整（满足 CFL） |
| 终止时间 | 0.6 | 模拟时长 |
| CFL 数 | $\leq 0.45$ | 稳定性条件 |

---

## 5. 解的性质分析

### 5.1 特征线行为

初始高斯脉冲 $u(x,0) = e^{-x^2}$ 的特征线：

- **中心区域**：$u \approx 1$，特征线速度最快
- **两侧区域**：$u \approx 0$，特征线速度接近零
- **结果**：中心部分快速向前传播，形成稀疏波

### 5.2 激波形成分析

由于初始条件 $u(x,0) = e^{-x^2}$ 是严格正的（全空间 $u > 0$），特征线不会相交，因此**不会形成激波**。

解呈现：
- **波前传播**：高速区域向前移动
- **波尾拖尾**：低速区域留在原地
- **整体展宽**：波包随时间逐渐展宽

### 5.3 数值误差对比

| 方法 | L2 误差 | 分析 |
|------|---------|------|
| Upwind | ~0.37 | 人工粘性导致过度平滑 |
| Minmod | ~0.38 | TVD限制器影响精度 |
| Lax-Wendroff | ~0.37 | 高阶精度但受分辨率限制 |

---

## 6. 数值格式公式汇总

### 6.1 通量函数

$$F(u) = f(u) = \frac{1}{2}u^2$$

### 6.2 Rusanov 通量

$$F_{i+1/2} = \frac{1}{2}(F(u_L) + F(u_R)) - \frac{\alpha_{i+1/2}}{2}(u_R - u_L)$$

$$\alpha_{i+1/2} = \max(|u_L|, |u_R|)$$

### 6.3 统一更新格式

$$u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x}(F_{i+1/2} - F_{i-1/2})$$

---

## 7. 代码实现

### 7.1 burgers_equation.py 结构

```
burgers_equation.py
├── initial_condition(x)       # u(x,0) = exp(-x^2)
├── exact_solution(x, t)       # 特征线法（牛顿迭代）
├── upwind_scheme()            # 迎风格式
├── minmod_scheme()            # Minmod TVD格式
├── lax_wendroff_scheme()      # Lax-Wendroff格式
├── interpolate_linear()       # 线性插值（误差计算）
└── main()                     # 主函数：计算并绘图
```

### 7.2 输出文件

- **results/Burgers.pdf**：四种方法对比图（4子图布局）
- **results/Burgers.png**：PNG预览图

---

## 8. 关键物理洞察

### 8.1 非线性波传播

1. **振幅依赖的速度**：振幅越大，传播越快
2. **波包展宽**：对于正初始条件，波包向两侧展宽
3. **无激波条件**：当初始条件全为正时，不会形成激波

### 8.2 数值方法选择

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 初步测试 | Upwind | 简单稳定 |
| 精确激波 | Minmod | TVD 无振荡 |
| 高精度需求 | Lax-Wendroff | 二阶精度 |

### 8.3 CFL 条件

$$\text{CFL} = \frac{|u|_{\max}\Delta t}{\Delta x} \leq 0.45$$

必须满足 CFL 条件以保证数值稳定性。

---

## 9. 参考文献

1. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*.
2. Whitham, G. B. (1974). *Linear and Nonlinear Waves*.
3. Courant, R., Hilbert, D. (1962). *Methods of Mathematical Physics*.
4. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*.

---

*本文档为 MagFluDynSim 项目生成——偏微分方程数值方法*
