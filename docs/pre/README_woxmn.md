# woxmn.py — 2D 有限体积约束输运 MHD 模拟（Orszag-Tang 涡旋问题）

## 概述

`woxmn.py` 是一个**二维理想磁流体力学（MHD）**数值模拟程序，使用**有限体积法（Finite Volume Method）**结合**约束输运（Constrained Transport, CT）**格式，求解经典的 **Orszag-Tang 涡旋问题**。

该代码由 Philip Mocz（@PMocz, 2023） 原创，以教学/演示为目的，用纯 Python + NumPy 实现了一套完整的 MHD 求解器，代码结构清晰，适合学习 MHD 数值方法。

---

## 物理背景

### Orszag-Tang 涡旋

Orszag-Tang 涡旋是 MHD 湍流研究中的经典测试问题。初始条件设置如下：

- **密度**：均匀 $\displaystyle \rho = \frac{\gamma^2}{4\pi}$
- **速度场**：大尺度涡旋
  - $v_x = -\sin(2\pi y)$
  - $v_y = \sin(2\pi x)$
- **磁场**：由磁势 $A_z$ 生成
  - $\displaystyle A_z = \frac{\cos(4\pi x)}{4\pi\sqrt{4\pi}} + \frac{\cos(2\pi y)}{2\pi\sqrt{4\pi}}$
  - $\boldsymbol{B} = \nabla \times (A_z \hat{\boldsymbol{z}})$
- **气压**：均匀 $P_{\text{gas}} = \gamma / (4\pi)$，再加磁压 $P_{\text{mag}} = \frac{1}{2}|\boldsymbol{B}|^2$ 得到总压

该初始条件会产生复杂的 MHD 激波、磁重联和湍流结构，是验证 MHD 代码鲁棒性的标准算例。

### 控制方程

求解**理想 MHD 方程组**（忽略粘性和电阻耗散）：

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \boldsymbol{v}) = 0
\tag{质量守恒}
$$

$$
\frac{\partial (\rho\boldsymbol{v})}{\partial t} + \nabla \cdot \left( \rho\boldsymbol{v}\boldsymbol{v} - \boldsymbol{B}\boldsymbol{B} + P^* \mathbb{I} \right) = 0
\tag{动量守恒}
$$

$$
\frac{\partial E}{\partial t} + \nabla \cdot \left( (E + P^*)\boldsymbol{v} - \boldsymbol{B}(\boldsymbol{v} \cdot \boldsymbol{B}) \right) = 0
\tag{能量守恒}
$$

$$
\frac{\partial \boldsymbol{B}}{\partial t} - \nabla \times (\boldsymbol{v} \times \boldsymbol{B}) = 0
\tag{感应方程}
$$

$$
\nabla \cdot \boldsymbol{B} = 0
\tag{无散度约束}
$$

其中总压 $P^* = P_{\text{gas}} + \frac{1}{2}|\boldsymbol{B}|^2$，总能量密度 $E = \dfrac{P_{\text{gas}}}{\gamma-1} + \dfrac{1}{2}\rho|\boldsymbol{v}|^2 + \dfrac{1}{2}|\boldsymbol{B}|^2$。

---

## 数值方法

### 1. 有限体积法（Godunov 型格式）

采用**单元中心**的有限体积法，将计算域划分为 $N \times N$ 个正方形网格单元（边长 $\Delta x$），每个单元上的物理量 $\boldsymbol{U}$ 代表该单元的平均值：

$$
\boldsymbol{U}_{i,j} = \frac{1}{\Delta x^2} \iint_{\text{cell}_{i,j}} \boldsymbol{u}(x,y) \,dx\,dy
$$

半离散形式：

$$
\frac{d\boldsymbol{U}_{i,j}}{dt} = -\frac{1}{\Delta x}\left( \boldsymbol{F}^{(x)}_{i+\frac12,j} - \boldsymbol{F}^{(x)}_{i-\frac12,j} \right) - \frac{1}{\Delta x}\left( \boldsymbol{F}^{(y)}_{i,j+\frac12} - \boldsymbol{F}^{(y)}_{i,j-\frac12} \right)
$$

其中 $\boldsymbol{F}^{(x)}$、$\boldsymbol{F}^{(y)}$ 分别为 $x$、$y$ 方向的数值通量。

### 2. 交错网格与约束输运（CT）

磁场分量定义在**单元面心**（staggered grid）：

- $\tilde{b}_x$：$x$ 方向磁场，位于单元**左右面心**（face-centered）
- $\tilde{b}_y$：$y$ 方向磁场，位于单元**上下面心**（face-centered）

单元中心的磁场 $B_x, B_y$ 由面心场平均得到：

$$
B_x(i,j) = \frac{1}{2}\bigl[\tilde{b}_x(i,j) + \tilde{b}_x(i+1,j)\bigr], \qquad
B_y(i,j) = \frac{1}{2}\bigl[\tilde{b}_y(i,j) + \tilde{b}_y(i,j+1)\bigr]
$$

磁矢势 $A_z$ 定义在**节点**（cell node，即网格角点）上。通过离散旋度算子从 $A_z$ 求得面心磁场（`getCurl()`）：

$$
\tilde{b}_x(i,j) = \frac{A_z(i,j) - A_z(i,j-1)}{\Delta x}
\quad\longleftrightarrow\quad B_x = \frac{\partial A_z}{\partial y}
$$

$$
\tilde{b}_y(i,j) = -\frac{A_z(i,j) - A_z(i-1,j)}{\Delta x}
\quad\longleftrightarrow\quad B_y = -\frac{\partial A_z}{\partial x}
$$

约束输运（`constrainedTransport()`）保证 $\nabla \cdot \boldsymbol{B} = 0$ 精确到机器精度：
1. 从 $x$ 方向 Riemann 通量 $F_{\tilde{B}_y}^{(x)}$ 和 $y$ 方向 Riemann 通量 $F_{\tilde{B}_x}^{(y)}$，在节点处平均出电场 $E_z$
2. 对 $E_z$ 取离散旋度得到面心磁场的修正量

从感应方程 $\partial \boldsymbol{B}/\partial t = \nabla \times (\boldsymbol{v} \times \boldsymbol{B})$，电场 $\boldsymbol{E} = -\boldsymbol{v} \times \boldsymbol{B}$ 的 $z$ 分量为：

$$
E_z = -(\boldsymbol{v} \times \boldsymbol{B})_z = v_y B_x - v_x B_y
$$

磁场的时间演化用 $E_z$ 表示为：

$$
\frac{\partial B_x}{\partial t} = -\frac{\partial E_z}{\partial y}, \qquad
\frac{\partial B_y}{\partial t} = \frac{\partial E_z}{\partial x}
$$

### 3. MUSCL 重构 + 斜率限制器

为达到**二阶空间精度**同时避免非物理振荡（Gibbs 现象），采用 MUSCL 方法：

1. **计算梯度**（`getGradient()`）：各变量中心差分

   $$
   \left(\frac{\partial f}{\partial x}\right)_{i,j} \approx \frac{f_{i+1,j} - f_{i-1,j}}{2\Delta x}, \qquad
   \left(\frac{\partial f}{\partial y}\right)_{i,j} \approx \frac{f_{i,j+1} - f_{i,j-1}}{2\Delta x}
   $$

2. **斜率限制**（`slopeLimit()`）：Minmod 型限制器

   $$
   \left(\frac{\partial f}{\partial x}\right)_{\text{limited}} = \min\!\left(1,\; \frac{f_{i,j} - f_{i-1,j}}{\Delta x \cdot f_{dx}}\right) \cdot f_{dx}, \quad \text{then symmetrically for } f_{i+1,j}
   $$

   保证重构值不超出相邻单元平均值的范围。

3. **空间外推**（`extrapolateInSpaceToFace()`）：

   $$
   f_{i+\frac12}^{(L)} = f_{i} + \frac{\Delta x}{2}\left(\frac{\partial f}{\partial x}\right)_i, \qquad
   f_{i+\frac12}^{(R)} = f_{i+1} - \frac{\Delta x}{2}\left(\frac{\partial f}{\partial x}\right)_{i+1}
   $$

   其中 $f_{i+\frac12}^{(L)}$ 为界面左侧状态（来自左单元），$f_{i+\frac12}^{(R)}$ 为界面右侧状态（来自右单元）。

### 4. 二阶 Predictor-Corrector 时间推进

采用两步格式：

**Step 1 — Predictor（时间外推半布）：**

$$
f' = f - \frac{\Delta t}{2} \cdot \mathcal{L}(f, \nabla f)
$$

其中 $\mathcal{L}$ 为 MHD 方程的空间算子（平流 + 磁压 + 洛伦兹力项），对密度、速度、压强、磁场分别做半时间步外推（L427-458）。

**Step 2 — Corrector（完整更新）：**

对半时间步的 $f'$ 做空间重构 → 计算 Riemann 通量 → 用 `applyFluxes()` 和 `constrainedTransport()` 完成 $\Delta t$ 的完整更新：

$$
\boldsymbol{U}^{n+1} = \boldsymbol{U}^n - \frac{\Delta t}{\Delta x}\left( \boldsymbol{F}_{i+\frac12}^{(x)} - \boldsymbol{F}_{i-\frac12}^{(x)} \right) - \frac{\Delta t}{\Delta x}\left( \boldsymbol{F}_{j+\frac12}^{(y)} - \boldsymbol{F}_{j-\frac12}^{(y)} \right)
$$

### 5. Rusanov（局部 Lax-Friedrichs）Riemann 求解器

`getFlux()` 函数（L268-345）计算相邻两个状态的数值通量。以一维 $x$ 方向为例，左右状态分别为 $\boldsymbol{U}_L$ 和 $\boldsymbol{U}_R$：

**Step 1 — Star 态平均：**

$$
\rho^* = \frac{1}{2}(\rho_L + \rho_R), \quad
(\rho v_x)^* = \frac{1}{2}(\rho_L v_{x,L} + \rho_R v_{x,R}), \quad
(\rho v_y)^* = \frac{1}{2}(\rho_L v_{y,L} + \rho_R v_{y,R})
$$

$$
E^* = \frac{1}{2}(E_L + E_R), \quad
B_x^* = \frac{1}{2}(B_{x,L} + B_{x,R}), \quad
B_y^* = \frac{1}{2}(B_{y,L} + B_{y,R})
$$

**Step 2 — 物理通量（以 $x$ 方向为例）：**

$$
F_{\rho} = \rho v_x
$$

$$
F_{\rho v_x} = \rho v_x^2 + P^* - B_x^2
$$

$$
F_{\rho v_y} = \rho v_x v_y - B_x B_y
$$

$$
F_{E} = (E + P^*)v_x - B_x(B_x v_x + B_y v_y)
$$

$$
F_{\tilde{B}_y} = B_y v_x - B_x v_y = -E_z
$$

**Step 3 — Rusanov 耗散项：**

$$
\boldsymbol{F}_{\text{num}} = \boldsymbol{F}(\boldsymbol{U}^*) - \frac{C}{2}(\boldsymbol{U}_R - \boldsymbol{U}_L)
$$

其中 $C$ 为界面最大波速，由快磁声速 $c_f$ 和流速决定：

$$
c_f = \sqrt{\frac{1}{2}\left(c_0^2 + c_a^2\right) + \frac{1}{2}\sqrt{(c_0^2 + c_a^2)^2}}
$$

- 声速：$\displaystyle c_0 = \sqrt{\frac{\gamma P_{\text{gas}}}{\rho}}$
- Alfvén 速度：$\displaystyle c_a = \sqrt{\frac{|\boldsymbol{B}|^2}{\rho}}$
- 最大波速：$\displaystyle C = \max\left(c_{f,L} + |v_{x,L}|,\; c_{f,R} + |v_{x,R}|\right)$

### 6. CFL 条件

自适应时间步长：

$$
\Delta t = C_{\text{CFL}} \cdot \frac{\Delta x}{\max(c_f + |\boldsymbol{v}|)}
$$

CFL 数设为 $C_{\text{CFL}} = 0.4$。

---

## 逐函数公式详解

### `getCurl(Az, dx)` — L13

从节点磁矢势 $A_z$ 计算面心磁场 $\tilde{b}_x, \tilde{b}_y$。

**公式：**

$$
\tilde{b}_x(i,j) = \frac{A_z(i,j) - A_z(i,j-1)}{\Delta x}
\quad\Longleftrightarrow\quad
B_x = \frac{\partial A_z}{\partial y} \;\;(\text{向后差分})
$$

$$
\tilde{b}_y(i,j) = -\frac{A_z(i,j) - A_z(i-1,j)}{\Delta x}
\quad\Longleftrightarrow\quad
B_y = -\frac{\partial A_z}{\partial x} \;\;(\text{向后差分})
$$

$A_z$ 位于节点 $(\text{node}_{i,j})$，$\tilde{b}_x$ 在面心 $(\text{face}_{i,j}^{(x)})$，$\tilde{b}_y$ 在面心 $(\text{face}_{i,j}^{(y)})$。

在代码中，`np.roll(arr, L, axis=...)` 其中 $L=1$ 表示数组沿指定轴正向偏移 1 格，即 `np.roll(arr, 1, axis=0)[i,j] = arr[i-1,j]`，因此：

```python
bx = (Az - np.roll(Az, 1, axis=1)) / dx   # Az(i,j) - Az(i,j-1)
by = -(Az - np.roll(Az, 1, axis=0)) / dx  # -(Az(i,j) - Az(i-1,j))
```

---

### `getDiv(bx, by, dx)` — L31

计算面心磁场的散度 $\nabla \cdot \boldsymbol{B}$，用于诊断约束输运效果。

**公式：**

$$
(\nabla \cdot \boldsymbol{B})_{i,j} \approx \frac{\tilde{b}_x(i,j) - \tilde{b}_x(i-1,j)}{\Delta x} + \frac{\tilde{b}_y(i,j) - \tilde{b}_y(i,j-1)}{\Delta x}
$$

这是 $\partial \tilde{b}_x/\partial x + \partial \tilde{b}_y/\partial y$ 的向后差分近似。

在理想情况下（CT 精确保持），此值应为机器精度零。

---

### `getBavg(bx, by)` — L47

将面心磁场 $\tilde{b}_x, \tilde{b}_y$ 平均到单元中心，得到 $B_x, B_y$。

**公式：**

$$
B_x(i,j) = \frac{1}{2}\left[\tilde{b}_x(i,j) + \tilde{b}_x(i+1,j)\right], \qquad
B_y(i,j) = \frac{1}{2}\left[\tilde{b}_y(i,j) + \tilde{b}_y(i,j+1)\right]
$$

面心 $\tilde{b}_x(i,j)$ 在左面，$\tilde{b}_x(i+1,j)$ 在右面，平均即得单元中心值。$\tilde{b}_y$ 同理。

代码中 `np.roll(bx, 1, axis=0)` 将 $\tilde{b}_x$ 在 $i$ 方向偏移 +1，使得：

```python
Bx = 0.5 * (bx + np.roll(bx, 1, axis=0))   # 0.5 * [bx(i,j) + bx(i-1,j)]
```

Wait — 这里需要仔细：`np.roll(bx, 1, axis=0)[i,j] = bx[i-1,j]`，所以 `bx[i,j] + np.roll(bx, 1, axis=0)[i,j] = bx[i,j] + bx[i-1,j]`。但这给出的是面心 $\tilde{b}_x$ 在 $i$ 和 $i-1$ 处的平均，对应的是单元 $(i-1,j)$ 的中心。

实际上，根据网格几何关系，单元 $(i,j)$ 的左面是 $\tilde{b}_x(i,j)$，右面是 $\tilde{b}_x(i+1,j)$，所以正确的中心平均应该是：

$$
B_x(i,j) = \frac{1}{2}[\tilde{b}_x(i,j) + \tilde{b}_x(i+1,j)]
$$

代码实现为：

```python
Bx = 0.5 * (np.roll(bx, -1, axis=0) + bx)
```

因为 `np.roll(bx, -1, axis=0)[i,j] = bx[i+1,j]`。

---

### `getConserved(rho, vx, vy, P, Bx, By, gamma, vol)` — L65

从原始变量 $(\rho, v_x, v_y, P_{\text{total}}, B_x, B_y)$ 计算守恒变量。

**公式（单元体积 $V = \Delta x^2$）：**

$$
\text{Mass} = \rho V
$$

$$
\text{Mom}_x = \rho v_x V
$$

$$
\text{Mom}_y = \rho v_y V
$$

$$
\text{Energy} = \left[ \frac{P_{\text{gas}}}{\gamma - 1} + \frac{1}{2}\rho(v_x^2 + v_y^2) + \frac{1}{2}(B_x^2 + B_y^2) \right] V
$$

其中热压 $P_{\text{gas}} = P_{\text{total}} - \frac{1}{2}(B_x^2 + B_y^2)$。
总能量三项分量依次为：内能、动能、磁能。

---

### `getPrimitive(Mass, Momx, Momy, Energy, Bx, By, gamma, vol)` — L93

从守恒变量反推原始变量，是 `getConserved()` 的逆运算。

**公式：**

$$
\rho = \frac{\text{Mass}}{V}
$$

$$
v_x = \frac{\text{Mom}_x}{\rho V}, \qquad v_y = \frac{\text{Mom}_y}{\rho V}
$$

$$
E_{\text{total}} = \frac{\text{Energy}}{V}
$$

$$
P_{\text{total}} = \left[ E_{\text{total}} - \frac{1}{2}\rho(v_x^2+v_y^2) - \frac{1}{2}(B_x^2+B_y^2) \right](\gamma-1) + \frac{1}{2}(B_x^2+B_y^2)
$$

---

### `getGradient(f, dx)` — L119

对任意标量场 $f$ 计算中心差分梯度。

**公式：**

$$
\left(\frac{\partial f}{\partial x}\right)_{i,j} \approx \frac{f_{i+1,j} - f_{i-1,j}}{2\Delta x}
$$

$$
\left(\frac{\partial f}{\partial y}\right)_{i,j} \approx \frac{f_{i,j+1} - f_{i,j-1}}{2\Delta x}
$$

代码中用 `np.roll(f, -1, axis=0)`（右移，即 $i \to i+1$）和 `np.roll(f, 1, axis=0)`（左移，即 $i \to i-1$）实现二阶中心差分。

---

### `slopeLimit(f, dx, f_dx, f_dy)` — L137

Minmod 型斜率限制器，防止 MUSCL 重构产生新的极值。

**公式（以 $x$ 方向为例，分两步限制）：**

**Step 1 — 限制左侧斜率：**

$$
r_L = \frac{f_{i,j} - f_{i-1,j}}{\Delta x \cdot f_{dx}}, \qquad
f_{dx} \gets \max(0,\; \min(1,\; r_L)) \cdot f_{dx}
$$

**Step 2 — 限制右侧斜率：**

$$
r_R = -\frac{f_{i,j} - f_{i+1,j}}{\Delta x \cdot f_{dx}}, \qquad
f_{dx} \gets \max(0,\; \min(1,\; r_R)) \cdot f_{dx}
$$

$y$ 方向同理。除数中添加小量 $10^{-8}$ 避免除零。

**物理含义：** 如果重构梯度会导致面心值超出相邻单元均值的范围，则缩小梯度至不越界的最大值。$r < 0$ 意味着梯度方向与局部单调性矛盾，此时将梯度截断为 0（一阶精度）。

---

### `extrapolateInSpaceToFace(f, f_dx, f_dy, dx)` — L189

将单元中心的物理量 $f$ 沿特征线外推到面心位置，得到界面两侧的 Riemann 状态。

**公式：**

对于 $x$ 方向的左右界面，在单元 $(i,j)$ 处：

$$
f_{i+\frac12}^{(L)} = f_{i,j} + \frac{\Delta x}{2}\left(\frac{\partial f}{\partial x}\right)_{i,j}
\quad\text{(右面，左状态)}
$$

$$
f_{i-\frac12}^{(R)} = f_{i,j} - \frac{\Delta x}{2}\left(\frac{\partial f}{\partial x}\right)_{i,j}
\quad\text{(左面，右状态)}
$$

代码实现分两步：
1. 计算未对齐的左右面心值：
   ```python
   f_XL = f - f_dx * dx/2   # 左面（本单元视角）
   f_XR = f + f_dx * dx/2   # 右面（本单元视角）
   ```
2. 对 `f_XL` 做 `np.roll(..., -1, axis=0)` 右移一格，将"左单元的左面"对齐到"本单元的右面"（Riemann 对偶面）：
   ```python
   f_XL = np.roll(f_XL, -1, axis=0)  # f_XL(i,j) = f(i+1,j) - f_dx(i+1,j)*dx/2
   ```
   最终 `f_XR(i,j)` = 界面 $i+\frac12$ 左状态，`f_XL(i,j)` = 界面 $i+\frac12$ 右状态。

$y$ 方向同理。

---

### `applyFluxes(F, flux_F_X, flux_F_Y, dx, dt)` — L216

将面心数值通量应用到守恒变量上，完成有限体积更新。

**公式：**

$$
F_{i,j} \gets F_{i,j} - \Delta t \cdot \Delta x \cdot \left[ \text{flux\_F\_X}_{i,j} - \text{flux\_F\_X}_{i-1,j} \right] - \Delta t \cdot \Delta x \cdot \left[ \text{flux\_F\_Y}_{i,j} - \text{flux\_F\_Y}_{i,j-1} \right]
$$

由于 $F = \boldsymbol{U} \cdot \Delta x^2$（守恒变量 = 通量密度 × 体积），上式等价于标准有限体积更新格式：

$$
\boldsymbol{U}_{i,j}^{n+1} = \boldsymbol{U}_{i,j}^n - \frac{\Delta t}{\Delta x}\left( \boldsymbol{F}_{i+\frac12}^{(x)} - \boldsymbol{F}_{i-\frac12}^{(x)} \right) - \frac{\Delta t}{\Delta x}\left( \boldsymbol{F}_{j+\frac12}^{(y)} - \boldsymbol{F}_{j-\frac12}^{(y)} \right)
$$

其中 $\text{flux\_F\_X}_{i,j} = \boldsymbol{F}_{i+\frac12}^{(x)}$（界面 $i+\frac12$ 处的 $x$ 方向通量），$\text{flux\_F\_X}_{i-1,j} = \boldsymbol{F}_{i-\frac12}^{(x)}$（界面 $i-\frac12$ 处的 $x$ 方向通量）。

代码实现：

```python
F += -dt * dx * flux_F_X              # 减去右面通量贡献
F +=  dt * dx * np.roll(flux_F_X, 1, axis=0)  # 加上左面通量贡献（np.roll(..., 1) 取 flux_F_X(i-1,j)）
F += -dt * dx * flux_F_Y              # 减去上面通量贡献（注：y 方向正方向向上）
F +=  dt * dx * np.roll(flux_F_Y, 1, axis=1)  # 加上下面通量贡献
```

---

### `constrainedTransport(bx, by, flux_By_X, flux_Bx_Y, dx, dt)` — L238

约束输运（Constrained Transport）更新面心磁场，保证 $\nabla \cdot \boldsymbol{B} = 0$ 精确成立。

**物理背景：**

理想的 MHD 感应方程 $\partial \boldsymbol{B}/\partial t = \nabla \times (\boldsymbol{v} \times \boldsymbol{B})$ 自动保持 $\nabla \cdot \boldsymbol{B} = 0$（因为 $\nabla \cdot (\nabla \times \cdot) = 0$）。CT 格式在离散层面上保留了这一性质。

**Step 1 — 节点电场平均：**

面心通量 $F_{\tilde{B}_y}^{(x)}$ 和 $F_{\tilde{B}_x}^{(y)}$ 分别给出了 $x$ 界面和 $y$ 界面处的电场 $E_z$：

$$
F_{\tilde{B}_y}^{(x)} = B_y v_x - B_x v_y = -E_z^{(x\text{-face})}, \qquad
F_{\tilde{B}_x}^{(y)} = B_x v_y - B_y v_x = E_z^{(y\text{-face})}
$$

在单元节点 $(\text{node}_{i,j})$ 处（即单元 $(i,j)$ 的右上角），$E_z$ 取周围四个界面电场值的算术平均：

$$
E_z(i,j) = \frac{1}{4}\Big[ E_z^{(x)}(i,j) + E_z^{(x)}(i,j+1) + E_z^{(y)}(i,j) + E_z^{(y)}(i+1,j) \Big]
$$

代入通量表达：

$$
E_z(i,j) = \frac{1}{4}\Big[ -F_{\tilde{B}_y}^{(x)}(i,j) - F_{\tilde{B}_y}^{(x)}(i,j+1) + F_{\tilde{B}_x}^{(y)}(i,j) + F_{\tilde{B}_x}^{(y)}(i+1,j) \Big]
$$

**Step 2 — 离散旋度更新磁场：**

对 $E_z$ 取离散旋度得到面心磁场修正：

$$
\Delta \tilde{b}_x(i,j) = -\frac{E_z(i,j) - E_z(i,j-1)}{\Delta x}
\quad\Longleftrightarrow\quad
\frac{\partial B_x}{\partial t} = -\frac{\partial E_z}{\partial y}
$$

$$
\Delta \tilde{b}_y(i,j) = \frac{E_z(i,j) - E_z(i-1,j)}{\Delta x}
\quad\Longleftrightarrow\quad
\frac{\partial B_y}{\partial t} = \frac{\partial E_z}{\partial x}
$$

**Step 3 — 时间推进：**

$$
\tilde{b}_x \gets \tilde{b}_x + \Delta t \cdot \Delta \tilde{b}_x, \qquad
\tilde{b}_y \gets \tilde{b}_y + \Delta t \cdot \Delta \tilde{b}_y
$$

**为什么保证 $\nabla \cdot \boldsymbol{B} = 0$：** 因为 $\Delta \boldsymbol{B}$ 是电场的旋度，$B^{n+1} = B^n + \Delta t (\nabla \times E_z)$，而 $\nabla \cdot (\nabla \times E_z) \equiv 0$，初始 $\nabla \cdot \boldsymbol{B} = 0$ 意味着 $\nabla \cdot \boldsymbol{B}^{n+1} = 0$。

---

### `getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, Bx_L, Bx_R, By_L, By_R, gamma)` — L268

Rusanov（局部 Lax-Friedrichs）Riemann 求解器，计算界面的数值通量。

**公式推导（以 $x$ 方向为例）：**

**1. 左右状态物理量：**

左侧：$\rho_L, v_{x,L}, v_{y,L}, P_L, B_{x,L}, B_{y,L}$
右侧：$\rho_R, v_{x,R}, v_{y,R}, P_R, B_{x,R}, B_{y,R}$

**2. Star 态（算术平均）：**

$$
\rho^* = \frac{1}{2}(\rho_L + \rho_R), \quad
(\rho v_x)^* = \frac{1}{2}(\rho_L v_{x,L} + \rho_R v_{x,R}), \quad
(\rho v_y)^* = \frac{1}{2}(\rho_L v_{y,L} + \rho_R v_{y,R})
$$

$$
E^* = \frac{1}{2}(E_L + E_R), \quad
B_x^* = \frac{1}{2}(B_{x,L} + B_{x,R}), \quad
B_y^* = \frac{1}{2}(B_{y,L} + B_{y,R})
$$

其中 $E_{L,R}$ 由状态方程计算：

$$
E = \frac{P_{\text{gas}}}{\gamma - 1} + \frac{1}{2}\rho(v_x^2+v_y^2) + \frac{1}{2}(B_x^2+B_y^2), \quad
P_{\text{gas}} = P_{\text{total}} - \frac{1}{2}(B_x^2+B_y^2)
$$

Star 态总压：

$$
P^* = (\gamma - 1)\left[E^* - \frac{1}{2}\frac{(\rho v_x)^*{}^2 + (\rho v_y)^*{}^2}{\rho^*} - \frac{1}{2}(B_x^{*2}+B_y^{*2})\right] + \frac{1}{2}(B_x^{*2}+B_y^{*2})
$$

**3. 物理通量（$x$ 方向）：**

$$
F_{\rho} = (\rho v_x)^*
$$

$$
F_{\rho v_x} = \frac{(\rho v_x)^*{}^2}{\rho^*} + P^* - B_x^{*2}
$$

$$
F_{\rho v_y} = \frac{(\rho v_x)^* (\rho v_y)^*}{\rho^*} - B_x^* B_y^*
$$

$$
F_{E} = \frac{(E^* + P^*)(\rho v_x)^*}{\rho^*} - B_x^* \frac{B_x^*(\rho v_x)^* + B_y^*(\rho v_y)^*}{\rho^*}
$$

$$
F_{\tilde{B}_y} = \frac{B_y^*(\rho v_x)^* - B_x^*(\rho v_y)^*}{\rho^*} = B_y^* v_x^* - B_x^* v_y^* = -E_z
$$

**4. 波速计算：**

左右两侧的热压：

$$
P_{\text{gas},L} = P_L - \frac{1}{2}(B_{x,L}^2 + B_{y,L}^2), \qquad
P_{\text{gas},R} = P_R - \frac{1}{2}(B_{x,R}^2 + B_{y,R}^2)
$$

声速、Alfvén 速度、快磁声速：

$$
c_{0,L} = \sqrt{\frac{\gamma P_{\text{gas},L}}{\rho_L}}, \qquad
c_{a,L} = \sqrt{\frac{B_{x,L}^2 + B_{y,L}^2}{\rho_L}}, \qquad
c_{f,L} = \sqrt{\frac{1}{2}(c_{0,L}^2 + c_{a,L}^2) + \frac{1}{2}\sqrt{(c_{0,L}^2 + c_{a,L}^2)^2}}
$$

右侧同理。界面最大波速：

$$
C = \max\left(c_{f,L} + |v_{x,L}|,\; c_{f,R} + |v_{x,R}|\right)
$$

**5. Rusanov 耗散修正：**

$$
F_{\rho}^{\text{(num)}} = F_{\rho} - \frac{C}{2}(\rho_L - \rho_R)
$$

$$
F_{\rho v_x}^{\text{(num)}} = F_{\rho v_x} - \frac{C}{2}(\rho_L v_{x,L} - \rho_R v_{x,R})
$$

$$
F_{\rho v_y}^{\text{(num)}} = F_{\rho v_y} - \frac{C}{2}(\rho_L v_{y,L} - \rho_R v_{y,R})
$$

$$
F_{E}^{\text{(num)}} = F_{E} - \frac{C}{2}(E_L - E_R)
$$

$$
F_{\tilde{B}_y}^{\text{(num)}} = F_{\tilde{B}_y} - \frac{C}{2}(B_{y,L} - B_{y,R})
$$

$y$ 方向的通量计算在调用时交换参数顺序：将 $y$ 分量视作"法向"，$x$ 分量视作"切向"，复用同一 `getFlux()` 函数。返回的顺序也相应调整（`flux_Momy_Y` 为法向动量通量，`flux_Momx_Y` 为切向动量通量，`flux_Bx_Y` 为磁场 $B_x$ 的通量）。

---

### `main()` — L348

主程序函数，包含完整的模拟循环流程。

**算法流程：**

```
1. 初始化参数 (N, boxsize, gamma, CFL, ...)
2. 建立网格 (xlin, Y, X, Xn, Yn)
3. 设置初始条件 (rho, vx, vy, P, Az)
4. 从 Az 计算面心磁场 bx, by: getCurl()
5. 从面心磁场计算中心磁场: getBavg()
6. 计算守恒变量: getConserved()
7. 时间循环 (t < tEnd):
   a. 面心→中心磁场平均: getBavg()
   b. 从守恒变量求原始变量: getPrimitive()
   c. 计算 CFL 步长 dt
   d. 计算各变量梯度: getGradient()
   e. 斜率限制: slopeLimit()
   f. Predictor 半步外推
   g. 空间外推到面心: extrapolateInSpaceToFace()
   h. x 方向 Riemann 通量: getFlux()
   i. y 方向 Riemann 通量: getFlux() (参数交换)
   j. 更新守恒变量: applyFluxes()
   k. 约束输运更新磁场: constrainedTransport()
   l. 输出诊断 (divB), 实时绘图
8. 保存最终图像
```

---

## 运行方法

### 环境要求

- Python 3.x
- NumPy
- Matplotlib

### 执行

```bash
cd MagFluDynSim/src
python woxmn.py
```

### 参数调整

在 `main()` 函数开头的参数区（L351-359）可调整：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `N` | 128 | 网格分辨率（每维） |
| `boxsize` | 1.0 | 计算域大小 |
| `gamma` | 5/3 | 绝热指数（理想气体） |
| `courant_fac` | 0.4 | CFL 数 |
| `tEnd` | 0.5 | 模拟总时间 |
| `tOut` | 0.01 | 输出帧间隔 |
| `useSlopeLimiting` | `True` | 是否启用斜率限制器 |
| `plotRealTime` | `True` | 是否实时绘图 |

---

## 输出

- **控制台**：每个时间步输出当前时间 $t$ 和平均 $|\nabla \cdot \boldsymbol{B}|$（用于监控无散度约束的保持情况）
- **图形**：实时显示密度 $\rho$ 的伪彩色图；模拟结束时保存为 `constrainedtransport.png`

### 预期物理结果

Orszag-Tang 涡旋的典型演化特征：

| 时间 | 物理现象 |
|---|---|
| $t \approx 0.1$ | 初始涡旋开始变形，激波开始形成 |
| $t \approx 0.2 \sim 0.3$ | 激波相互作用，电流片形成 |
| $t \approx 0.4 \sim 0.5$ | 湍流充分发展，小尺度结构丰富 |

---

## 与 `MHDcode.py` 的对比

| 特性 | `woxmn.py` | `MHDcode.py` |
|---|---|---|
| 物理模型 | 理想 MHD（无耗散） | 电阻 MHD（有耗散） |
| 数值格式 | 有限体积法 + CT | 伪谱法 + 有限差分 |
| 边界条件 | 周期性 | 周期性 |
| 模拟问题 | Orszag-Tang 涡旋 | 磁重联（Harris 电流片） |
| 网格 | $128 \times 128$ | $160 \times 160$ |
| 磁场处理 | 约束输运（精确 $\nabla\cdot\boldsymbol{B}=0$） | 矢势法（隐式满足） |
| 时间积分 | 二阶 Predictor-Corrector | 四阶 RK4 |
| 目的 | 通用 MHD 求解器 | 磁重联专门研究 |

---

## 完整函数公式一览

| 函数 | 核心数学公式 |
|---|---|
| `getCurl()` | $\tilde{b}_x = \frac{A_z(i,j) - A_z(i,j-1)}{\Delta x},\quad \tilde{b}_y = -\frac{A_z(i,j) - A_z(i-1,j)}{\Delta x}$ |
| `getDiv()` | $\nabla\cdot\boldsymbol{B} \approx \frac{\tilde{b}_x(i) - \tilde{b}_x(i-1)}{\Delta x} + \frac{\tilde{b}_y(j) - \tilde{b}_y(j-1)}{\Delta x}$ |
| `getBavg()` | $B_x = \frac{1}{2}(\tilde{b}_x(i) + \tilde{b}_x(i+1)),\quad B_y = \frac{1}{2}(\tilde{b}_y(j) + \tilde{b}_y(j+1))$ |
| `getConserved()` | $(\rho,\ \rho v_x,\ \rho v_y,\ E_{\text{total}}) \times V$ |
| `getPrimitive()` | $\rho = \text{Mass}/V,\quad v = \text{Mom}/(\rho V),\quad P = (\gamma-1)(E/V - \frac12\rho v^2 - \frac12 B^2) + \frac12 B^2$ |
| `getGradient()` | $\partial_x f \approx (f_{i+1}-f_{i-1})/(2\Delta x),\quad \partial_y f \approx (f_{j+1}-f_{j-1})/(2\Delta x)$ |
| `slopeLimit()` | $f_{dx} \gets \min(1,\ \frac{f_i-f_{i-1}}{\Delta x \cdot f_{dx}})\cdot f_{dx}$（分左右两侧限制） |
| `extrapolateInSpaceToFace()` | $f_{i+\frac12}^{(L)} = f_i + \frac{\Delta x}{2}f_{dx},\quad f_{i+\frac12}^{(R)} = f_{i+1} - \frac{\Delta x}{2}f_{dx,i+1}$ |
| `applyFluxes()` | $U \gets U - \frac{\Delta t}{\Delta x}\left(F_{i+\frac12} - F_{i-\frac12}\right) - \frac{\Delta t}{\Delta x}\left(F_{j+\frac12} - F_{j-\frac12}\right)$ |
| `constrainedTransport()` | $E_z = \frac14\sum\text{face }E_z,\quad \Delta\tilde{b}_x = -\partial_y E_z,\quad \Delta\tilde{b}_y = \partial_x E_z$ |
| `getFlux()` | $F_{\text{num}} = F(U^*) - \frac{C}{2}(U_R - U_L)$（Rusanov 格式） |

---

## 参考资料

1. Orszag, S. A. & Tang, C. M. (1979). *Small-scale structure of two-dimensional magnetohydrodynamic turbulence*. Journal of Fluid Mechanics, 90(1), 129-143.
2. Mocz, P. (2023). *Create Your Own Constrained Transport Magnetohydrodynamics Simulation (With Python)*. [GitHub: @PMocz](https://github.com/pmocz/constrainingtransport-python)
3. Stone, J. M. & Gardiner, T. A. (2009). *A simple unsplit Godunov method for multidimensional MHD*. New Astronomy, 14(2), 139-148.
4. Tóth, G. (2000). *The $\nabla \cdot \boldsymbol{B} = 0$ Constraint in Shock-Capturing Magnetohydrodynamics Codes*. Journal of Computational Physics, 161(2), 605-652.
5. Evans, C. R. & Hawley, J. F. (1988). *Simulation of magnetohydrodynamic flows — A constrained transport method*. The Astrophysical Journal, 332, 659-677.

---

*本文档由 Sisyphus 自动生成，代码版本基于 Philip Mocz (2023) 的原始实现。*
