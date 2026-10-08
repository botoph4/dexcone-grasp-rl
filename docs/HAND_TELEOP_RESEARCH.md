# 腕装 RealSense 手部遥操作调研报告(P24 假肢手)

> 调研日期:2026-09-28。目标:RealSense 相机安装于手腕靠近掌心处,识别人手骨骼模型,
> 提取手指弯曲角度,映射到 P24 假肢手(20 自由度)做动作映射;要求实时、遮挡时动作
> 推测与保持、无跳变。

---

## 1. 系统定义与总体结论

```
RealSense(腕装) → 手部姿态估计(21关键点 / MANO网格)
              → 指关节角度提取(MCP/PIP/DIP 屈曲 + 外展,拇指 CMC/MP/IP)
              → 遮挡状态机(滤波 + 保持 + 恢复过渡)
              → 人手→P24 关节映射(直接缩放 / 向量优化 / GMR 蒸馏)
              → MuJoCo 仿真验证 → 真实 P24 手
```

**一句话结论**:
- 感知:**2D 关键点 + RealSense 深度 lifting + 几何角度提取**作为首版(CPU 实时 30–60 FPS);
  GPU 可用时升级到 **WiLoR / Fast-HaMeR** 直接输出 MANO 轴角(MANO 的 15 个手指自由度与
  P24 的 20 关节 1:1 对应,抗自遮挡更强)。
- 映射:**直接角度缩放**验证链路 → **向量优化重定向(dex-retargeting / AnyTeleop 方案)**
  作为正式映射(1–2 ms/帧、关节限位硬约束、时间平滑项)→ 数据积累后用 **GMR 蒸馏**加速
  (<0.1 ms/帧,可上嵌入式);备选 **GeoRT** 神经映射(MuJoCo 仿真里无监督训练 3–5 分钟,
  1 kHz 推理)。
- 遮挡:**One Euro 滤波(跟踪态)→ 卡尔曼 predict-only 外推(保持态)→ 限速+指数融合(恢复态)**
  的三态状态机,叠加人手运动学先验(DIP≈0.7×PIP 耦合、关节限位、骨长约束)。

### 1.1 决策记录(2026-09-28,已确认)

1. **感知层**:Python 可运行、ARM 架构、低 GPU 依赖 → 选 **MediaPipe Hand Landmarker
   (Tasks API) + RealSense 深度 lifting + 几何角度提取**。实测 PyPI:`mediapipe 1.0.1`
   提供 `manylinux_2_28_aarch64` 轮子(ARM64 Linux 直接 pip 安装,需 Ubuntu 22.04+/JetPack 6)
   与 `macosx_11_0_arm64` 轮子(开发机),纯 CPU 推理。备选 `ai-edge-litert 2.2.0`、
   `onnxruntime 1.30.0` 亦有 aarch64 轮子。
   **实机验证(2026-09-28,Apple M4 Pro / macOS 26)**:1.0.1 在 macOS 26 上崩溃
   (DrishtiMetalHelper `service_ Service is unavailable`,Metal 图服务初始化失败);
   **0.10.31 正常**,CPU delegate 实测 **~10.6 ms/帧(94 FPS)**。因此:开发机(macOS 26)
   装 0.10.31,ARM 部署机装 1.0.1(唯一有 aarch64 轮子的版本);代码只使用两版本
   共有的 Tasks API 面。备选 `ai-edge-litert 2.2.0`、`onnxruntime 1.30.0` 亦有 aarch64 轮子。
2. **相机**:**双后端适配**——RealSense **D405** 与 **Orbbec Gemini 305**(实测其产品页:
   最小工作距离 4 cm、68 g、42×42 mm、深度 848×480@60、HW 深度-彩色对齐,官方定位即
   "wrist-mounted robotic applications",比 D405 的 7 cm 更近)。
   - `pyrealsense2 2.58.4` 实测 PyPI **已有 `manylinux2014_aarch64` 轮子**(cp39/cp310/cp312,
     ARM 部署机可直接 pip 安装);macOS **无任何轮子** → 开发机源码构建
     (librealsense v2.58.4 + `-DBUILD_PYTHON_BINDINGS=ON`,已装进 base env)。
   - `pyorbbecsdk2 2.1.2`(2026-08)官方提供 **macOS-arm64 + manylinux_2_27_aarch64 轮子**
     (导入名 `pyorbbecsdk`),双端直接 pip 安装(已装进 base env)。
   - 代码层用 `FrameSource` 抽象(`RealsenseSource` / `OrbbecSource` / `ReplaySource`),
     CLI `--camera realsense|orbbec` 选择后端,开发机无相机时用录制数据回放调试。
3. **映射层(2026-09-30 更新)**:默认 **HybridRetargeter**(关节映射为主 + 指尖捏取
   修正);另实现 **FingertipRetargeter**(纯指尖优化)、**GeoRtRetargeter**(仿真训练的
   神经映射,holdout 指尖误差 8.5mm)、**DexRetargetingAdapter**(dex-retargeting 向量
   优化,pinocchio 经 conda-forge 安装——注意 pip 上的 pinocchio 是同名无关包)、
   DirectAngleScaling(纯关节缩放)。CLI `--retarget {hybrid,fingertip,scaling,geort,dex}`。
4. **遮挡处理**:按 §5.2 状态机原样实现(1€ 滤波 → EMA 退化 → KF 外推/松弛 → 限速恢复)。

---

## 2. P24 手运动学(来自本仓库 URDF 实测)

[p24_hand_right.urdf](../p24grasp/assets/p24_hand_right.urdf) 实测:

| 手指 | 关节 | 范围 | 备注 |
|---|---|---|---|
| 四指(食/中/环/小) | j1 MCP 屈曲 | 0 ~ +80° | ⚠️ **mimic 关节:强制 q1 = q3(PIP),1:1 耦合** |
| | j2 MCP 外展 | ±15° | |
| | j3 PIP | 0 ~ +80° | |
| | j4 DIP | 0 ~ +80° | ⚠️ **mimic 关节:强制 q4 = q3(PIP),1:1 耦合** |
| 拇指 | j1 CMC 屈曲 | 0 ~ +67° | |
| | j2 CMC 屈/伸 | −20° ~ +80° | |
| | j3 CMC 外展 | ±20° | |
| | j4 MP | 0 ~ +80° | 无独立 IP |

**关键推论**:

1. **实际独立自由度是 16 个**(拇指 4 + 四指 × 3),而非 20 个——四指 DIP 被硬件 1:1 耦合
   到 PIP(URDF 实测:joint_4 带 `<mimic joint=..._joint_3 multiplier=1>`,2026-09-30 复核
   确认,此前文档误记为 MCP-PIP 耦合)。人手 DIP≈0.7×PIP 的自然耦合被硬件强制为 1:1,
   人手 DIP 信息在耦合模式下被舍弃:
   - 直接角度缩放:`couple_dip_pip=True` 时机器人 DIP = 机器人 PIP;
   - 指尖优化重定向:优化变量按 16 个独立 DOF 建模(MCP/外展/PIP,拇指 4),DIP 由
     HandModel 的 mimic 处理自动跟随(FK 内建);
   - **硬件核实项**:需确认真实 P24 硬件是否确实 1:1 耦合(URDF 可能只是近似),若不是,
     修改 URDF 释放约束即可,映射层不需要改动。
2. 每指 4 关节(屈曲×3 + 外展×1)与 MANO 参数化完全一致(MANO:15 个手指关节,
   MCP 2DOF + PIP + DIP),是"人手→P24"结构化映射的有利条件。

---

## 3. 感知:手部姿态估计与关节角度提取

### 3.1 相机选型

腕装近景(手距相机 10–30 cm)必须注意最小工作距离:

- **Orbbec Gemini 305**:最小深度 **4 cm**、68 g、42×42 mm、深度 848×480@60(HW 深度-彩色
  对齐),官方定位即腕装机器人应用——**最近距离优势最大**。
- **RealSense D405**:最小深度 ≈7 cm、全局快门、87°×58° FOV、60/90 FPS。
- D435 系列最小深度 ≈28 cm、卷帘快门,**不适用于腕装近景**。
- 深度图需对齐到 RGB(RealSense 用 `rs.align`、Orbbec 用 HW_MODE),并按真实内参反投影
  (MediaPipe 的 z 是相对深度,不可用)。
- **⚠️ macOS 上 Orbbec 需 root 运行**(实测 2026-09-28):自 macOS 12 起,通过
  libuvc 类 SDK 打开 UVC 相机(绕过系统授权弹窗)需要 sudo,否则 `uvc_open failed
  Return Code: -3`(OrbbecSDK issue #9 / libuvc issue #194 确认)。命令:
  `sudo <conda-python> scripts/teleop_camera.py --camera orbbec`。
  永久免 sudo 方案:按 Apple 文档做一个覆盖默认 UVC 扩展的空 CoreMediaIO DAL
  插件(VID 0x2BC5/PID 0x0840)装入 `/Library/CoreMediaIO/Plug-Ins/DAL/` 后重插相机。

### 3.2 技术路线对比(2023–2026 SOTA)

**路线一:2D 关键点 + 深度 lifting(推荐首版)**

| 方法 | 精度 | 速度 | 部署 | 适配性 |
|---|---|---|---|---|
| **MediaPipe Hand Landmarker**(Tasks API) | 衍生研究:PIP 角度 RMSE ~5–8° | CPU 数十~上百 FPS | TFLite 全平台,~10MB | ★★★★★ 首版首选 |
| WiLoR 检测器(CVPR 2025) | FreiHAND PA-MPJPE 5.5mm(重建端) | 检测 130–175 FPS @4090 | PyTorch | ★★★★ GPU 升级首选 |
| OpenPose | 弱于 MediaPipe | GPU ~10 FPS | 停维护 | ✗ 不推荐 |

注意:MediaPipe 旧 API `mp.solutions.hands` 已在 2025–2026 新版移除,须用 Tasks API
`HandLandmarker`。其 egocentric 近景检出率有报告仅 77.3% 帧检到手,腕装域外分布,
后续可考虑微调或换 WiLoR 检测器。

**路线二:端到端 mesh 回归(升级路线,GPU)**

| 方法 | 精度 | 速度 | 说明 |
|---|---|---|---|
| **WiLoR**(CVPR 2025) | FreiHAND PA-MPJPE 5.5mm / F@15 0.993 | 检测+重建整流程 GPU 实时 | anchor-free 检测器很强,mesh 路线首选 |
| **HaMeR**(CVPR 2024) | 6.0mm / F@15 0.990 | 27 FPS @4060Ti(671M 参数) | in-the-wild 鲁棒性基准 |
| **Fast-HaMeR**(2026,蒸馏) | +0.4mm 损失 | ~40–50 FPS @4060Ti | 把 SOTA 压进实时档的关键支撑 |
| Hamba(NeurIPS 2024) | 5.3mm | 省算力 | Mamba 架构,工程化不如 HaMeR 系成熟 |
| EgoForce(SIGGRAPH 2026) | HOT3D 相机系 MPJPE −28% | 双手 ~14 FPS @3090 | **最贴近腕装场景**(前臂表示消除尺度歧义),但 FPS 不足 |
| HandDiff(CVPR 2024) | RGB-D 点云扩散 | 慢 | 思路可参考,实时不可行 |

**路线三:专用硬件对照** — Ultraleap LMC2(120 Hz,关节级输出)不能腕装照自己的手,
但可作为**角度标定/验证基准**:桌面固定 LMC2 与腕装相机同时测同一只手,校正估计偏差。

**egocentric/腕装相关数据集**(微调候选):HOT3D(Meta Aria 眼镜鱼眼,与腕装近景最接近)、
AEA(Meta 日常活动)、AssemblyHands、EgoExo4D、H2O、LightHand99K(专门为腕装相机控制
假肢手生成的合成数据集,同场景先例)。

### 3.3 关节角度提取

**方法 A:21 关键点几何(首版)**。屈曲角 = 相邻骨向量的夹角:
`θ = 180° − acos(v1·v2)`,v1 = PIP−MCP,v2 = DIP−PIP(MediaPipe 索引:食指 6-7-8、
中指 10-11-12、无名 14-15-16、小指 18-19-20);MCP 外展角用叉积在掌平面投影求解。
实测参考:与量角器对照 PIP 均值差 2.85°、RMSE 7.27°(Science Progress 2023)。
工程要点:**先时序滤波再求角**(滤波 3D 关键点或直接在角度域滤波),角度对关键点抖动敏感。

**方法 B:MANO 参数直接输出(升级)**。WiLoR/HaMeR 回归 MANO 45 个轴角参数;
MANO 局部系约定 z 轴为屈曲轴,屈曲角即轴角的 z 分量,MCP 外展即 x 分量——
**逐关节映射到 P24 无需重定义轴系,自由度 1:1 对应**。自带手部先验,抗自遮挡更强。

**方法 C:深度点云直接拟合(回退)**。关键点失效时用深度 ROI 分割手指,圆柱/骨骼模型
拟合指节方向;对遮挡最鲁棒但实现成本高,作为二期兜底。

---

## 4. 映射:人手角度 → P24 关节角

### 4.1 问题定义

人手(21 关键点或 MANO,每指 MCP/PIP/DIP + 拇指 CMC/MP/IP)→ P24(20 关节,实际 16 独立
DOF)是"相似但不相同"的重定向:自由度结构相近、轴系与尺寸不同、P24 有 MCP-PIP 耦合
硬约束、拇指自由度少于人手 CMC。评价指标:指尖位置对齐(mm)、对指/捏合语义、时间平滑。

### 4.2 方法对比

| 方法 | 代表工作 | 计算 | 优点 | 缺点 |
|---|---|---|---|---|
| 直接角度缩放 | — | <0.1ms | 单调、零奇异、无依赖 | 指尖不齐、尺寸失配 |
| **向量优化** | **AnyTeleop**(RSS 2023)、Open-TeleVision、DexPilot | 1–2ms/帧(SLSQP) | 指尖对齐+对指语义+限位硬约束+时间平滑项 | 需 URDF 向量对定义、调参(s, α) |
| 指尖 IK | DexCap | 每帧迭代 | 指尖毫米级对齐 | 冗余/奇异(全伸位),工程费劲 |
| **GMR 蒸馏** | Calinon 框架 | <0.1ms/帧 | 极快、C^∞ 连续、自动学习指间耦合 | 需成对数据、无硬限位、外推不可靠 |
| 神经映射 | **GeoRT**(Meta 2025) | 1 kHz | 仿真无监督训练 3–5 分钟、自碰撞感知 | 依赖仿真 FK 准确度 |
| 协同降维 | Santello 1998(2 个 PC 解释 >80% 方差)、PCHands(2025) | 低 | 维度低、外推稳 | 指尖精度受损,需混合修正 |

向量优化的核心公式(AnyTeleop):
```
min_q  Σ_i ‖v_i^human − s·FK_i(q)‖² + α‖q − q_prev‖²
s.t.  q_lb ≤ q ≤ q_ub          # 关节限位硬约束
```
其中 `v_i` 为腕系下的人手关键点向量:5×(腕→指尖)+ 2×(拇指尖→食指/中指尖),
可选 4×(相邻指尖向量,用于激活 j2 外展,否则外展欠定)。SLSQP 热启动自上一帧解,
实测 25–60 Hz 可用。开源实现:**[dex-retargeting](https://github.com/dexsuite/dex-retargeting)**
(pip 可装,Pinocchio 底层,给 P24 定义向量对即可接入)。

### 4.3 GMR 详解(Gaussian Mixture Regression)

**数学原理**(Calinon JIST 2015):
- 训练:对配对数据 `ξ = [ξᴵ; ξᴼ]`(人手角度 → P24 角度)用 EM 拟合 GMM;
- 推理:给定输入,输出为高斯条件分布 `P(ξᴼ|ξᴵ) = Σᵢ hᵢ(ξᴵ) N(μ̂ᵢᴼ(ξᴵ), Σ̂ᵢᴼ)`,
  点估计取均值 `μ̂ᴼ = Σᵢ hᵢ μ̂ᵢᴼ`;
- 关键性质:**K=1 退化为线性回归,K=N 退化为核回归**——GMR 是线性/非线性回归的连续谱,
  用 8–20 个高斯即可拟合人手→P24 的仿射映射;
- 推理复杂度 O(K),与样本数无关,20 维输出 <0.1 ms。

**输入/输出空间定义**(本场景):
- 输入:人手 15 个手指角度(或降维到 2–3 个协同系数);
- 输出:P24 16 个独立关节角(四指 MCP 由耦合约束事后施加);
- 也可输出速度(角度→速度 GMR + 积分,天然平滑)。

**训练数据需求**:成对 (θ_human, q_p24) 数据,几百~几千条覆盖"全张开/抓握/对指"三类
区域的姿态对即可;数据来源 = 向量优化器离线跑人手数据集(优化器天然是 GMR 的"标注器")
或 MuJoCo 仿真采集。

**GMR vs GPR**:GPR 非参数、方差与数据密度相关(适合 OOD 检测)但每查询 O(N);
GMR 参数化、查询 O(K)、支持多模态与缺维条件化,但输出方差不是"离数据多远"的度量。
**实时遥操作映射选 GMR**;需要"训练区外拒绝执行"时补一个 GPR/距离检测器或阈值化 hᵢ 熵。

**局限**:不保证关节限位(输出需 clamp);远离训练区外推失真;每台新手需重新标注。

### 4.4 推荐映射方案(分阶段)

1. **阶段 0(半天)——直接角度缩放 + 拇指特殊处理**
   - 四指:人手 PIP → P24 j3(线性缩放),DIP → j4,MCP 外展 → j2(±15° 钳位);
     耦合关节 j1 取人手 PIP 或 `0.5·(θ_MCP + θ_PIP)`(需实测选优);
   - 拇指:人手 CMC 屈曲/外展 → P24 j1/j2(2×2 缩放矩阵),MP → j3,IP → j4;
   - 用于快速验证全链路。
2. **阶段 1(1–2 天)——向量优化(正式映射)**
   - dex-retargeting + P24 向量对定义;注意 MCP-PIP 耦合按 16 独立 DOF 建模;
   - 1€ 滤波前置 + α‖q−q_prev‖² 平滑 + 限位硬约束。
3. **阶段 2(可选)——GMR 蒸馏**:优化器离线产数据 → 训练 K=8–20 GMR → 嵌入式实时推理;
   上线前对比指尖误差,>5 mm 则退回混合模式(GMR 预测 + 低频优化器精修)。
4. **备选——GeoRT**:P24 已有 MuJoCo 模型,仿真采样 (q, FK, 碰撞标签) 训练几何映射器,
   3–5 分钟训练、1 kHz 推理,适合大批量部署,代价是依赖仿真 FK 准确度。

---

## 5. 遮挡处理与时间平滑

### 5.1 滤波选型

| 滤波 | 角色 | 参数建议 | 要点 |
|---|---|---|---|
| **One Euro**(Casiez CHI 2012) | 跟踪态默认 | **f_cmin=3–4 Hz(不要用默认 1 Hz!)、β≈0.05–0.1(°/s 单位)、d_cutoff=1 Hz** | 实测:默认 1 Hz 会把近一半运动路径吃掉;Monado 默认参数使轨迹误差 +74~156% |
| EMA | 退化态兜底 | α=0.7 | 无效帧跳过更新即隐式 hold;实测比 1€ 均衡 |
| **卡尔曼 CV 模型** | 保持态外推 + 延迟预测 | R=(1.5°)²,Q_vel 按 σ_a≈60°/s² | predict-only 外推比渐近保持更合理;可预测 1–2 帧补偿管线延迟 |
| Savitzky-Golay | 离线后处理 | W=5, polyorder=3 | 因果版滞后 (W−1)/2 帧,实时预算偏紧 |
| 滑动平均 | ✗ 不推荐 | — | 实测误差差一个数量级 |

经验教训:①在**角度域**滤波(不是像素/2D 关键点),按关节独立设参数(远端指节抖动最大);
②**先预测后滤波**:死推预测补偿延迟 −31%,重滤波又 +74%,别在预测输出上堆重滤波;
③用时间戳算 dt,帧率抖动时不要按帧数。

### 5.2 遮挡处理状态机(可直接落地)

```
参数(起始值,30–90 fps):
  T_hi=0.7, T_lo=0.5          # 手置信度滞回阈值
  N_enter=3, N_exit=5 帧      # 连续帧计数去毛刺(进保持更保守)
  T_hold_fast=0.5s, T_max=3s(仅状态降级)   # 保持两阶段超时
  slew_max=300°/s             # 恢复限速(假肢安全上限)
  τ_blend=80–150ms            # 恢复指数融合时间常数

状态:
  S0 TRACKING  置信度≥T_hi 且关节可见性≥0.5:
      θ̂ = OneEuro(θ_raw); clamp ROM; slew limit; KF 持续吸收正常观测
  S1 DEGRADED  T_lo≤置信度<T_hi 或部分关节不可见:
      α_eff = α·mean(v_i)          # 可见性加权,不可见关节少更新
      θ̂ = α_eff·θ_raw + (1−α_eff)·θ̂_prev   # EMA 自然偏向保持
      + 运动学修正(DIP←0.7·PIP、ROM)
  S2 HOLD      置信度<T_lo 连续 N_enter 帧:
      0–0.5s:  θ̂ = KF.predict_only(dt)          # CV 模型外推
      0.5s后:   冻结保持(实施修正:原计划向休息位松弛+放弃,实测每次
                摘手后映射都像"重新开始",改为保持最后位姿;安全释放
                留待真机部署时评估)
      >3s:     状态 LOST(仅状态降级,命令仍保持冻结位姿)
  S3 RECOVERY  置信度≥T_hi 连续 N_exit 帧:
      ⚠️ 不平滑重置滤波器(重置是恢复跳变主因)
      |θ_raw−θ̂|>30° → slew 限速追赶(≤300°/s,~150ms 到位)
      否则 → 指数融合 θ̂ += (θ_raw−θ̂)(1−e^(−dt/τ_blend))
```

要点:①进出保持用滞回+连续帧计数防单帧抖动;②保持分两段(外推→冻结),摘手后映射不重置;
③延迟预算:检测+FK(10–30ms)+ 滤波(10–40ms)+ 状态机(0 额外) <100ms ✓;
④先验叠加顺序:**滤波 < 运动学耦合(DIP-PIP/ROM/骨长)< 模型外推 < 学习先验**
(如蒸馏 UmeTrack 时序模块/HMP 手部运动先验),按可靠性与成本递增。

### 5.3 运动学先验(廉价而有效)

- **DIP≈k·PIP 指内耦合**(最新量化:k = 食指 0.96、中指 0.76、无名 0.69、小指 0.64,
  工程统一 0.7;拇指单独处理)——遮挡时单指部分可见即可推测整指;
- **PCA 协同子空间 + 低维卡尔曼平滑**(Federolf 2016 范式):采本人手部数据求 PCA 基
  (3–5 个协同解释 80%+ 方差),遮挡时把可观测关节投影到低维空间重构全部角度;
- **骨长一致性**:3D 关键点投影回骨长约束(权重 0.5 量级),骨长标准差 2.07→1.20 mm;
- **关节限位 + 速度限幅**:每帧 clamp 到解剖 ROM,|Δθ| ≤ slew_max·dt。

---

## 6. 推荐整体架构(v1 落地版)

```
┌─────────────────────────────────────────────────────────┐
│ RealSense D405 / Orbbec Gemini 305(60 FPS,RGB-D aligned) │
│   → MediaPipe HandLandmarker(Tasks API,21 关键点)       │
│   → 深度反投影 → 3D 关键点(带可见性)                    │
│   → 几何角度提取(4×MCP/PIP/DIP + 外展;拇指 CMC/MP/IP)   │
│   → 遮挡状态机(§5.2:1€ 滤波/EMA/KF 外推/恢复融合)       │
│   → 人手 15 角 → P24 映射(§4.4)                         │
│   → MuJoCo 仿真验证(本仓库 p24grasp)→ 真实 P24          │
└─────────────────────────────────────────────────────────┘
升级路径:WiLoR/Fast-HaMeR 换掉 MediaPipe(输出 MANO 轴角,1:1 映射)→
         dex-retargeting 向量优化替换直接缩放 → GMR/GeoRT 蒸馏 →
         HOT3D/AEA/LightHand99K 风格腕装数据微调
```

---

## 7. 分阶段实施计划

| 阶段 | 内容 | 工作量 | 依赖 |
|---|---|---|---|
| 0 | 直接角度缩放 + 拇指特处理 + 1€ 滤波 + 基本 hold,打通"相机→仿真"链路 | ~1 天 | RealSense SDK + MediaPipe Tasks |
| 1 | 遮挡状态机完整实现(KF 外推、松弛、恢复过渡)+ 运动学先验 | 2–3 天 | 阶段 0 |
| 2 | dex-retargeting 向量优化(16 独立 DOF 建模,MCP-PIP 约束) | 1–2 天 | P24 URDF 向量对定义 |
| 3 | 真实 P24 联调(关节限位/速度限幅/安全位)+ 角度标定(LMC2 对照或量角器) | 2–3 天 | 硬件 |
| 4 | GMR 蒸馏(优化器离线产数据 + K=8–20 GMR + 嵌入式部署) | 2–3 天 | 阶段 2 数据积累 |
| 5 | WiLoR/Fast-HaMeR 升级 + 腕装数据微调 + GeoRT 备选 | 1–2 周 | GPU |

---

## 7.5 RL 接入的必要性分析(2026-09-30)

**结论:当前映射阶段不需要 RL;RL 的正确位置是"抓取接触后"的残差策略。** 理由:

1. **映射是确定性的监督/优化问题**:目标函数明确(复现人手姿态),优化式/神经式
   都能直接求解,RL 的"探索-试错"只会降低精度、增加训练成本——这是行业共识
   (AnyTeleop/GeoRT 均不用 RL 做映射)。
2. **RL 的真实价值**:①**接触后残差修正**——手碰到物体后,纯运动学映射不感知力,
   可能捏滑/推倒物体;用本仓库现有 PPO/SAC 抓取环境训练一个"以遥操作命令为参考
   动作的残差策略"(residual policy),只在接触阶段做小幅度调整(Contact-Anchored
   Retargeting + Residual Policy Learning 路线,2026 年多篇工作验证);②**任务级
   语义**——"抓稳后保持"这类长期行为。
3. **接入点**:遥操作命令 → RL 策略的参考动作/初始位形 → MuJoCo 仿真验证 → 实机。
   本仓库的 RL 基础设施(grasp env + PPO/SAC)可直接复用。
4. **不建议**:用 RL 端到端替换映射(训练慢、难调试、无仿真-实机迁移保证)。

## 8. 风险与开放问题

1. **P24 DIP-PIP 1:1 耦合是否真实**——需硬件核实(URDF 实测 joint_4 mimic joint_3);
   若硬件独立,修改 URDF 后映射层直接释放约束(人手 DIP 可用了)。
2. **腕装域外分布**——公开模型训练数据多为头戴/桌面视角;腕装近景检出率可能偏低,
   首版需实测 MediaPipe 在目标安装位姿下的检出率,必要时第 0 阶段就采集少量自有数据。
3. **RealSense 最小工作距离**——D405 标称 7 cm,安装位置需满足;手部快速运动时全局快门
   优势明显,避免 D435 的卷帘畸变。
4. **人手角度→角度映射的尺度问题**——P24 屈曲仅 0–80°(人手 MCP ~90°/PIP ~110°),
   直接缩放会改变抓握几何;向量优化方案天然规避(指尖对齐),阶段 0 缩放需验证手势语义。
5. **恢复期手势已改变**——HOLD 期间用户换了手势,恢复时 |Δθ|>30° 触发限速追赶,
   可能产生 ~150 ms 的"快速但受控"过渡;若需更平滑,可加"手势一致性校验"(恢复时先比对
   当前观测与 hold 前观测的相关性)。
6. **滤波与延迟预算**——60 FPS 下 1€(f_cmin=3–4 Hz)时间常数 40–53 ms,叠加检测延迟
   可能逼近 100 ms 上限;必要时用 KF 前向预测 1–2 帧补偿,换向过冲由 slew_limit 兜底。

---

## 9. 主要参考文献

**手部姿态估计**:HaMeR (arXiv:2312.05251) · WiLoR (arXiv:2409.12259) ·
Fast-HaMeR (arXiv:2603.16444) · Hamba (arXiv:2407.09646) · HandDiff (CVPR 2024) ·
EgoForce (arXiv:2605.12498) · UmeTrack (arXiv:2211.00099, SIGGRAPH Asia 2022) ·
HOT3D (arXiv:2411.19167) · LightHand99K (arXiv:2412.07105) · MediaPipe Tasks API

**映射**:AnyTeleop (arXiv:2307.04577) · Qin et al. (arXiv:2204.12490) ·
DexPilot (arXiv:1910.03135) · DexCap (arXiv:2403.07788) ·
Open-TeleVision (arXiv:2407.01512) · GeoRT (arXiv:2503.07541) ·
dex-retargeting (github.com/dexsuite/dex-retargeting) ·
Calinon JIST 2015 (GMR/TP-GMM 教程) · Santello 1998(姿态协同)

**遮挡与平滑**:One Euro (DOI 10.1145/2207676.2208639) · Federolf 2016(低维卡尔曼平滑) ·
SmoothNet (arXiv:2112.13715) · TUM 滤波/预测量化(Monado 论文) ·
Peng et al. Sensors 2026(1€ vs EMA + 骨长约束,mdpi.com/1424-8220/26/12/3730) ·
PIP-DIP 耦合量化(frontiersin.org/fnbot.2026.1775834)
