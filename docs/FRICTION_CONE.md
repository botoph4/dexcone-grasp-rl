# 摩擦锥计算与防滑抓握求解

本文档整理项目中摩擦锥建模（`p24grasp/model/friction_cone.py`）与防滑抓握
角度求解（`scripts/antislip_grasp.py`）的公式与实现约定。

## 1. 库仑摩擦锥（点接触模型）

指尖与物体视为**点接触**。接触力 $f$ 在接触坐标系（法向 $n$、两个切向
$t_1, t_2$）下分解为法向力 $f_n$ 与切向力 $f_t$。不滑动的条件是切向力不超过
摩擦容量：

$$
\|f_t\| \le \mu \, f_n
$$

在力空间中这是一个以法向为轴、半顶角 $\alpha = \arctan(\mu)$ 的**圆锥**——
摩擦锥。抓取稳定的必要条件是物体的重力/惯性力能被接触力在摩擦锥内的组合
抵消。

### 安全系数

真实摩擦系数有不确定性，设计时使用**有效摩擦系数**：

$$
\mu_\mathrm{eff} = \frac{\mu}{\mathrm{safety\_factor}}
\qquad \text{（默认 } \mu=0.7,\ \mathrm{safety}=1.4 \ \Rightarrow\ \mu_\mathrm{eff}=0.5\text{）}
$$

即把摩擦锥收窄，留出安全裕度。

## 2. 力的分解与滑移裕度（`FrictionCone`）

### 分解（`decompose`）

$$
f_n = n \cdot f, \qquad f_t = f - f_n \, n
$$

### 滑移裕度（`slip_margin`）

切向力与摩擦容量的比值，$\rho < 1$ 表示不滑：

$$
\rho = \frac{\|f_t\|}{\mu_\mathrm{eff} \, f_n}
$$

### 期望法向力（`desired_normal_force`）

给定当前负载 $f$，为保持不滑所需的法向力：

$$
f_n^\mathrm{des} = \frac{\|f_t\|}{\mu_\mathrm{eff}} + f_\mathrm{margin},
\qquad
f_n^\mathrm{des} \in [f_\mathrm{min},\ f_\mathrm{max}]
\qquad \text{（默认 } [0.1,\ 8.0]\ \mathrm{N}\text{，margin } 0.3\ \mathrm{N}\text{）}
$$

$f_\mathrm{margin}$ 提供正压力余量（接触必须压着物体，而不是刚好贴着）。

## 3. 摩擦锥的多面体近似（`polyhedral_matrix`）

凸优化/力闭合计算中常用**内接多面体**近似圆锥，约束写成线性不等式：

$$
F \, f \le 0, \qquad f = [f_{t_1},\ f_{t_2},\ f_n]^\top \ \text{（接触坐标系）}
$$

- **4 边近似（默认）**：方锥，切向半径 $c = \mu_\mathrm{eff} / \sqrt{2}$：

$$
F =
\begin{bmatrix}
 1 &  0 & -c \\
-1 &  0 & -c \\
 0 &  1 & -c \\
 0 & -1 & -c \\
 0 &  0 & -1
\end{bmatrix}
$$

- **一般 $m$ 边**：内接正 $m$ 边形，每条边对应切向单位向量
  $u_k = (\cos\phi_k,\ \sin\phi_k)$，$\phi_k = 2\pi k/m + \pi/m$，行约束
  $u_k \cdot f_t \le \mu_\mathrm{eff} \cos(\pi/m)\, f_n$，外加 $f_n \ge 0$。

近似多面体完全在摩擦锥**内部**（内接），所以满足多面体约束的力一定满足
库仑条件——保守但安全。

## 4. 防滑抓握角度求解管线（`compute_antislip_grasp`）

把"重量需要多少抓力"与"抓力对应手指弯多少"串起来：

| 步骤 | 公式 | 说明 |
| --- | --- | --- |
| 物体重量 | $W = m\,g$ | |
| 总法向力需求 | $f_n^\mathrm{total} = W / \mu_\mathrm{eff}$ | 摩擦容量抵消重力：$\mu_\mathrm{eff}\sum f_n \ge W$ |
| 每指法向力 | $f_n^\mathrm{finger} = f_n^\mathrm{total}/n + f_\mathrm{margin}$ | $n$ = 手指数（5） |
| 指尖压入量 | $\delta = f_n^\mathrm{finger} / k$ | 线性接触刚度（默认 $k=500\ \mathrm{N/m}$） |
| 压入量上限 | $\delta_\mathrm{max} = 0.5\,r$ | 防止小物体被压穿 |
| 目标半径 | $r_\mathrm{target} = r - \delta$ | 指尖目标点向物体内部压入 $\delta$ |
| 指尖 IK | $\min_q \| \mathrm{tip}(q) - (\mathrm{center} + r_\mathrm{target}\,u) \|^2 + 10^{-4}\|q\|^2$ | 带关节限位的 L-BFGS-B |
| 实际摩擦容量 | $F_\mathrm{cap} = \mu_\mathrm{eff}\, n\, f_n^\mathrm{ach}$，$f_n^\mathrm{ach} = \delta\, k$ | |
| 滑移裕度 | $\mathrm{slip\_margin} = F_\mathrm{cap} / W$ | **$>1$ 表示足以托住** |

径向方向 $u$ 与 `sim_grasp` 的 IK 一致：圆柱取张开姿态指尖到轴线的投影方向，
球体取全 3D 方向。

### 运行与输出

```bash
python scripts/antislip_grasp.py --mu 0.7 --safety 1.4 --stiffness 500
```

输出 `outputs/antislip/antislip_grasp.json`（各配置的所需法向力、压入量、
滑移裕度、关节角）与 `r*.png` 姿态预览。

### 已知限制

- 这是**静态几何模型**：不跑动力学，滑移裕度是必要条件检查，不是充分条件。
- 该管线沿用几何 IK 的径向目标——受 FINDINGS 第 7 节的限制（开口式包裹
  抓不住东西），输出的是"防滑设计角度"而非实测可用的抓取姿态。真正的
  抓取由 RL（见 [RL.md](RL.md)）提供。
