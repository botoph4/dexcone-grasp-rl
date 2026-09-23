# P24 手抓取仿真与强化学习

P24 假肢手的 MuJoCo 抓取仿真项目：几何 IK 与 PyRoki 运动学求解、PPO/SAC 抓取
策略训练、桌面/网页双端交互查看器（可拖拽物体看握持反应）。

## 安装

```bash
pip install -e .                 # 核心依赖（numpy/scipy/mujoco/matplotlib/pillow）
pip install -e ".[rl,web,test]"  # + RL 训练 / 网页查看器 / 测试
```

> **torch 必须用 conda-forge 安装**：macOS 上 pip 的 torch wheel 自带 libomp，
> 与 conda 的 llvm-openmp 冲突，`import torch` 直接崩溃：
> `conda install -c conda-forge pytorch`。

## 目录结构

```
├── p24grasp/                  # 主包
│   ├── paths.py               # 资源/构建/输出路径统一解析
│   ├── assets/                # URDF + STL 网格（包数据）
│   ├── model/                 # 模型层：urdf.py（HandModel）、mjcf.py（生成器）、friction_cone.py
│   ├── kinematics/            # 运动学：ik.py（几何 IK 求解器）
│   ├── sim/                   # 仿真：scene.py（交互场景/接触位掩码）、demo.py（静态渲染）
│   ├── env/                   # RL：obs.py（观测编码）、grasp.py（Gymnasium 环境）
│   ├── rl/                    # RL：train.py、eval.py
│   ├── pyroki/                # PyRoki 后端：probe.py（可行性验证）、ik.py（LM IK）
│   └── viewers/               # desktop.py（mjpython）、web.py（viser 浏览器）
├── scripts/                   # 薄启动脚本（兼容 python scripts/xxx.py 用法）
├── tests/                     # pytest 冒烟测试
├── docs/FINDINGS.md           # 根因分析与实测结论（长文）
├── build/                     # 生成物：hand.xml、场景 XML、缓存（gitignore）
└── outputs/                   # 产物：render/ eval/ antislip/ runs/（gitignore）
```

## 快速开始

### 桌面交互查看器（mjpython）

```bash
mjpython scripts/sim_viewer.py --adjust          # 摆位模式：滑块调位姿，g 释放抓握
mjpython scripts/sim_viewer.py --policy outputs/runs/cylinder/sac/best/best_model.zip
```

### 网页查看器（浏览器拖拽）

```bash
python scripts/sim_web.py                        # http://localhost:8080
python scripts/sim_web.py --shape sphere --auto-run
```

浏览器里：拖拽手柄（摆位=钉住移动 / 抓握=弹簧力拉拽）、「摆位」「抓取」「关节角度」
三个标签页的滑块与实时状态。

### 几何 IK 演示与 PyRoki 后端

```bash
python scripts/sim_grasp.py --preview            # 静态 IK 姿态预览
python scripts/pyroki_probe.py                   # PyRoki 可行性验证（FK 对比）
python scripts/pyroki_grasp.py --collision-aware --tilt-x -90
```

### RL 训练与评估

```bash
python scripts/train_grasp_rl.py --algo both --shape cylinder --timesteps 1000000
python scripts/eval_grasp_rl.py --shape cylinder --algo both --episodes 50 --media 2
```

也可以用 console 入口：`p24-viewer`、`p24-web`、`p24-train`、`p24-eval`、
`p24-pyroki`、`p24-build`。

### 测试

```bash
python -m pytest tests/
```

## 结果摘要

| 模型 | 成功率 | 保持时长 | 平均接触 | 末态漂移 |
| --- | --- | --- | --- | --- |
| cylinder / SAC | **0.52–0.77** | 3.1 s | 6.5 | 70.9 mm |
| cylinder / PPO | 0.46 | 3.01 s | 5.4 | 64.9 mm |
| sphere / PPO | 0.33–0.37 | 2.3–2.7 s | 4.4 | 62.6 mm |
| 几何 IK（基线） | **0.00** | 0 s | 0 | 物体 0.3 s 内飞出 |

**无策略的标称握姿对 26 mm 横向圆柱可保持 ~35-40 s**（包裹对齐摆放；更大直径
的物体会在 ~14 s 内被挤出，见 docs/FINDINGS.md 第 10 节——这是指尖可达性的
几何限制）。球体需要策略。关键设计结论（物体轴向必须垂直于手指屈曲平面、
指节胶囊碰撞体、接触位掩码、IK 目标半径符号等）见 [docs/FINDINGS.md](docs/FINDINGS.md)。

## 已知限制

- 指尖接触球 8 mm 为建模设定；胶囊碰撞体是视觉网格的近似
- 物体轴默认横向（+y）；`--tilt-x/--tilt-y` 可改
- 抓取判据为"物体留在手中"，不是举起
- 桌面查看器在 macOS 上必须用 `mjpython` 启动
