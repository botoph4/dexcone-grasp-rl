# 腕装相机手部遥操作:实现文档

> 状态:全部实现并测试(79 个测试)。设计背景与调研见
> [HAND_TELEOP_RESEARCH.md](HAND_TELEOP_RESEARCH.md);本文档描述**实际实现**的
> 技术细节:相机如何识别手骨骼、角度如何提取、如何重定向映射到 P24 灵巧手。

## 1. 系统总览

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 相机         │→  │ 手部检测      │→  │ 关节角度      │→  │ 遮挡状态机    │→  │ 重定向映射    │
│ D405/305     │   │ MediaPipe    │   │ 掌平面屈曲    │   │ 1€滤波/EMA   │   │ Hybrid(默认) │
│ FrameSource  │   │ + 深度 lifting│   │ + 侧摆估计器  │   │ /KF外推/恢复  │   │ + 自标定坐标系 │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                                  ↓
                                                                           P24 关节命令(20 关节,度)
                                                                                  ↓
                                                              ┌───────────────┴───────────────┐
                                                              │ cv2 2×2 窗口 / MuJoCo 窗口    │
                                                              │ MJPEG 网页 / CSV 日志          │
                                                              └───────────────────────────────┘
```

代码布局([p24grasp/teleop/](../p24grasp/teleop/)):

| 模块 | 职责 |
|---|---|
| `camera.py` | `FrameSource` 抽象:RealSense D405 / Orbbec Gemini 305 / 录制回放 |
| `detector.py` | MediaPipe HandLandmarker(Tasks API,CPU delegate)+ 深度反投影 |
| `angles.py` | 21 关键点 → 屈曲角(掌平面 MCP)+ 原始侧摆 |
| `lateral.py` | 侧摆估计器:derotation 测量、标定、置信门控、持久化 |
| `filters.py` | One Euro、EMA、卡尔曼(速度有限差分初始化) |
| `state_machine.py` | TRACKING/DEGRADED/HOLD/LOST 遮挡状态机 |
| `retarget.py` | 5 种重定向后端(Hybrid 默认)+ 自标定手掌坐标系 |
| `geort.py` | GeoRT 式神经重定向器(仿真训练,MLP,numpy 推理) |
| `pipeline.py` | 全链路组装 |
| `viewer.py` | cv2 2×2 窗口 / MuJoCo 交互窗口 / MJPEG 网页 |
| `run.py` | CLI:`--camera realsense|orbbec`、`--retarget hybrid|fingertip|scaling|geort|dex`、`--local`、`--mujoco-view`、`--view` |

## 2. 相机如何识别手骨骼

### 2.1 相机选型与配置

| | D405 | Gemini 305 |
|---|---|---|
| 最小工作距离 | ~7 cm | **4 cm** |
| 分辨率 | 640×480@60 | 848×480@60(HW 深度-彩色对齐) |
| SDK | pyrealsense2(源码构建,无 macOS 轮子) | pyorbbecsdk2(PyPI 双端轮子) |

平台注意(实测):macOS ≥12 上 Orbbec 需 **sudo** 运行(UVC 直连绕过 TCC 授权机制);
mediapipe 1.0.1 在 macOS 26 崩溃 → 开发机用 0.10.31,ARM 部署用 1.0.1(唯一有
aarch64 轮子的版本)。相机输出统一为 `Frame`(RGB + 米制深度 + 内参),彩色流
启动比深度慢约 5 秒,`start()` 内做确定性热身等待。

### 2.2 检测:MediaPipe HandLandmarker → 21 关键点 3D

1. **2D 检测**:MediaPipe Tasks API `HandLandmarker`(VIDEO 模式,强制
   `Delegate.CPU`——ARM 目标无 GPU,实测 M4 Pro 上 94 FPS),输出 21 个归一化关键点;
2. **深度 lifting**:每关键点取其邻域(3×3)深度中位数(抗离群),按内参反投影:
   `x = (u − cx)·z/fx, y = (v − cy)·z/fy`(MediaPipe 自带的 z 是相对深度,不用);
3. **可见度**:该模型变体不输出 per-landmark visibility(属性恒为 NaN)→
   `sanitize_visibility` 把 NaN 视为可见,关节有效性回退到"深度反投影是否成功"。

### 2.3 角度提取(骨骼 → 关节角)

**四指 MCP 屈曲——掌平面法**(BEHAVIOR 式,消除掌内扇形偏置):

```
n  = 掌平面法向(腕 + 4 个 MCP 拟合,SVD,方向与前帧连续)
v  = 近节指骨方向(PIP − MCP)
θ_MCP = atan2(−⟨v, n⟩, ‖v − ⟨v, n⟩·n‖)
```

**PIP/DIP/拇指**:相邻指骨间夹角 `θ = arccos(⟨a, b⟩/(‖a‖‖b‖))`(无符号)。

**四指侧摆(MCP ab/adduction)——derotation 法**(屈曲下精确):
把近节指骨绕屈曲轴解旋回掌平面后再测带符号角;置信度 = 解旋后平面内分量占比
(深握拳时塌缩→冻结输出)。掌心参考 = 腕+4 MCP 均值(MediaPipe 无掌骨点)。

## 3. 遮挡处理:三态状态机

```
TRACKING ──presence<0.5 连续3帧──→ HOLD ──>3s──→ LOST(安全位)
    ↑                              │(0–0.5s:卡尔曼常速外推)
    │                              │(0.5–3s:向休息位松弛)
    └──presence≥0.7 连续5帧───────┘
恢复:不平滑重置滤波器(重置是跳变主因);|Δ|>30° 时限速追赶(300°/s),
否则指数融合(τ≈100ms);滤波器历史重新播种到融合输出。
DEGRADED:部分关节不可见时,逐关节可见度加权 EMA(隐藏关节自然保持)。
```

关键实现细节:卡尔曼速度用**前两次测量有限差分初始化**(靠过程噪声收敛需数百帧);
slew 限速从零位启动存在爬坡(安全特性);恢复条件要求 **≥60% 关节有效**
(指尖在画面边缘属常态,不能要求 100%)。

## 4. 重定向映射(核心)

### 4.1 P24 运动学约束(URDF 实测)

- 20 关节 = 5 指 × 4;**四指 DIP(joint_4)被 mimic 1:1 耦合到 PIP(joint_3)**,
  实际 16 独立自由度(拇指 4 + 四指 MCP/外展/PIP),MCP 独立;
- P2.4 零位非共线:FK 给出 ~10.03° 的中性远端弯曲。

### 4.2 自标定手掌坐标系(滚动角无关)

相机装腕上有未知滚动角,固定坐标系映射会把"张开"映射成"侧摆"。**每帧从
关键点现场构建** camera→palm 旋转:

```
x = 平面内(腕→中指 MCP)方向        # 手指方向
y = 平面内(腕→拇指 CMC)方向正交化   # 拇指侧
R = [x; y; x×y](一阶平滑 + SVD 极分解重正交化)
```

掌平面法向方向与**前帧连续**(手斜着重新入画时绝不翻面——翻面会导致侧摆角
全部变号、手指镜像,这是"手移出再回来就全乱"的根因)。

### 4.3 自适应尺度(伸直 → 张开)

人手伸展半径 ~0.16 m vs P24 ~0.22 m,固定 scale=1 会把"伸直"映射成"弯曲"。
运行时跟踪人手腕-指尖距离的运行最大值:`s = r_robot / r_human`(限幅 0.8–1.8),
伸直→张开由构造保证。

### 4.4 HybridRetargeter(默认)

```
min_q  w_tip·Σᵢ vᵢ‖tᵢ − tipᵢ(q)‖²                    # 指尖位置(拇/食指加权 2×)
     + w_pip·Σᵢ vᵢ‖pᵢ − pipᵢ(q)‖²                    # Vector10 近端半部:PIP 位置
     + w_pinch·(d_thumb_index(q) − d_human)²          # 捏取距离
     + Σⱼ ρ_c(j)·(qⱼ − q0ⱼ)²                          # 关节分类先验
s.t. URDF 关节限位
```

- **q0(解析先验)**:`DirectAngleScaling`——MCP→j1、侧摆估计器→j2(共享单一事实源)、
  PIP=DIP=0.5×((θ_PIP+θ_DIP) − 10.03° 零位补偿)、拇指三弯曲→j1/j2/j4(j3 中性);
- **先验分类权重**(参考项目配置比例):侧摆 0.05 > MCP/远端 0.01,拇指 0
  (拇指由优化器从指尖目标精修);`reg_weight=10` 补偿 tip-only 目标与 Vector10
  的尺度差异;
- 求解:scipy `least_squares` **dogbox**(trf 从零位——恰在 12/16 个关节的下界——
  启动会原地卡死,首帧生产级 bug);热启动前比较"旧解 vs 先验"的残差取优;
- 深度丢失 → 优雅退回纯关节映射(上一帧命令保持)。

### 4.5 其他后端

| 后端 | 说明 |
|---|---|
| `fingertip` | 纯指尖位置优化(DexPilot 式;拇指欠定,姿态可能漂移) |
| `scaling` | 纯关节角度映射(链路验证) |
| `geort` | MLP(20→16 DOF)仿真训练:"定制手"技巧生成虚拟人手目标 + Hybrid 教师蒸馏;holdout 指尖误差 8.5 mm;纯 numpy 推理(ARM 可跑) |
| `dex` | dex-retargeting 官方向量优化(5×掌心→指尖 + 2×拇指尖→食/中指尖,mimic 由官方适配器处理;pinocchio 需 conda-forge——**pip 上的 pinocchio 是同名无关包**) |

### 4.6 侧摆标定与持久化

侧摆中性偏移是逐用户的。`LateralEstimator(auto_calibrate=True)` 启动后收集
60 帧高置信开掌帧,中位数 = 该用户自然扇形;完成后写入
`~/.cache/p24grasp/lateral_calibration.json`,下次启动自动加载。收集期间输出
中性零位(绝不用他人标定——错标定会让中间三指塌缩重叠)。

## 5. 可视化

- `--local`:cv2 窗口,2×2 网格(相机+关键点 | 深度 / P24 渲染 | 状态文本);
- `--mujoco-view`:标准 MuJoCo 交互窗口(可旋转缩放),关节名映射实时驱动;
- `--view`:极简 HTTP MJPEG 页面(`<img src="/stream">`),无前端依赖
  (viser 的 GUI 面板在本机浏览器组合上反复失联,已弃用);
- P24 离屏渲染:`build/hand.xml` + 注入定向光 + MuJoCo 蓝灰渐变背景;
  **GL 上下文非线程安全——渲染器必须在使用它的线程内创建**。

## 6. 测试与验证

79 个测试覆盖:角度几何(掌平面屈曲/derotation 精确性)、滤波(1€/EMA/KF 门控)、
状态机(保持/恢复无跳变)、重定向(已知目标恢复、捏取保持、伸直→张开、
滚动不变性、遮挡后热启动恢复、侧摆再入不镜像)、GeoRT(训练+推理+权重配对)、
dex 适配器、相机(回放往返)。CI 跑 pylint(10/10)。

真机验证要点:启动后开掌 1 秒(侧摆标定)→ 任意滚动角下张开/弯曲/对捏;
手移出画面再返回,映射保持稳定。
