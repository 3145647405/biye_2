# 🚛 多AGV调度MAPPO-Attention系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于多智能体近端策略优化(MAPPO)和多层次注意力机制的多AGV调度系统，用于解决复杂仓储环境下的任务分配与路径规划问题。

## 🎯 项目概述

本项目实现了一个完整的多AGV调度系统，具有以下核心特性：
- **修复的奖励机制**: 解决了训练不稳定和奖励分配问题
- **时空预留系统**: 实现多AGV路径协调，避免冲突和死锁
- **碰撞豁免机制**: 起点和终点允许多AGV共存
- **课程学习**: 9个阶段的渐进式训练，从简单到复杂
- **实时可视化**: 完整的训练监控和性能分析

## 🏗️ 系统架构

```
biye_augment/
├── src/                    # 核心源代码
│   ├── environment.py      # AGV环境实现（已修复）
│   ├── models.py          # MAPPO模型和注意力机制
│   ├── trainer.py         # 训练器
│   ├── utils.py           # 工具函数
│   └── visualization.py   # 可视化模块
├── config.yaml           # 配置文件（已优化）
├── complete_training.py   # 完整训练脚本
├── test_env.py           # 环境测试
├── results/              # 训练结果
└── checkpoints/          # 模型检查点
```

## ✨ 核心特性

### 已修复的关键问题
- **奖励机制**: 步数惩罚从-0.1降至-0.01，减少90%
- **奖励分配**: 修复碰撞、任务完成、死锁奖励分配逻辑
- **时空预留**: 实现多AGV协调路径规划系统
- **碰撞检测**: 起点和终点允许多AGV共存

### 技术亮点
- **多层次注意力机制**: 任务级、智能体级、全局环境注意力
- **MAPPO算法**: 集中式训练、分散式执行
- **课程学习**: 9个阶段渐进式训练
- **实时监控**: 每100个episode更新可视化
- **模块化设计**: 清晰的代码结构，便于扩展

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA (推荐，用于GPU加速)
- NumPy, Matplotlib, TensorBoard等

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/biye_augment.git
cd biye_augment
```

2. **创建conda环境**
```bash
conda create -n biye_RL python=3.8
conda activate biye_RL
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **验证安装**
```bash
python test_env.py
```

### 运行训练

**完整训练（推荐）**
```bash
python complete_training.py
```

**其他训练选项**
```bash
# 基础训练
python train.py

# 增强训练
python train_enhanced.py

# 改进训练
python train_improved.py
```

### 测试系统
```bash
# 环境功能测试
python test_env.py

# 可视化测试
python test_visualization.py
python test_env.py
```

## 项目结构

```
biye_augment/
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── environment.py      # AGV环境实现
│   ├── models.py          # 注意力增强的神经网络
│   ├── trainer.py         # MAPPO训练器
│   └── utils.py           # 工具函数
├── config.yaml            # 配置文件
├── train.py              # 训练主脚本
├── test_env.py           # 环境测试脚本
├── requirements.txt      # 依赖列表
└── README.md            # 项目说明
```

## 快速开始

### 1. 环境测试
```bash
python test_env.py
```
这将测试环境的基本功能、路径规划和神经网络模型。

### 2. 开始训练
```bash
python train.py --config config.yaml --render
```

参数说明：
- `--config`: 配置文件路径（默认：config.yaml）
- `--resume`: 恢复训练的检查点路径
- `--seed`: 随机种子（默认：42）
- `--device`: 设备选择（cpu/cuda/auto，默认：auto）
- `--render`: 启用环境渲染

### 3. 监控训练
使用TensorBoard查看训练进度：
```bash
tensorboard --logdir logs
```

## 配置说明

主要配置项在`config.yaml`中：

### 环境配置
```yaml
environment:
  map_width: 26          # 地图宽度
  map_height: 10         # 地图高度
  num_agvs: 4           # AGV数量
  num_tasks: 16         # 任务数量
  max_load: 25          # 最大载重
```

### 模型配置
```yaml
model:
  d_model: 64                    # 注意力维度
  num_attention_heads: 4         # 注意力头数
  enable_task_attention: true    # 启用任务级注意力
  enable_agent_attention: true   # 启用智能体级注意力
```

### 训练配置
```yaml
training:
  total_timesteps: 1000000  # 总训练步数
  batch_size: 256          # 批次大小
  learning_rate: 3e-4      # 学习率
  clip_epsilon: 0.2        # PPO裁剪参数
```

### 课程学习配置
```yaml
curriculum:
  enable: true
  stages:
    - name: "stage_1_basic"
      num_agvs: 2
      num_tasks: 8
      until:
        min_episodes: 1000
        min_return: 100.0
```

## 核心算法

### 多层次注意力机制

1. **任务级自注意力**: 识别任务间的相关性和优先级
2. **智能体级交叉注意力**: 学习AGV间的协作和避让策略  
3. **全局环境注意力**: 捕获全局环境动态和长期策略

### MAPPO算法

- **Actor网络**: 基于局部观测生成动作策略
- **Critic网络**: 基于全局状态评估价值函数
- **PPO更新**: 使用裁剪代理目标函数稳定训练

### 课程学习

从简单场景（少量AGV和任务）逐步过渡到复杂场景，提升训练效率和最终性能。

## 实验结果

训练过程中会记录以下指标：

- **效率指标**: 平均完工时间、路径总长度
- **协作指标**: 碰撞率、死锁率、协作成功率
- **学习指标**: 奖励曲线、损失函数、策略熵

## 扩展功能

### 添加新的注意力机制
在`src/models.py`中的`MultiLevelAttentionEncoder`类中添加新的注意力层。

### 修改奖励函数
在`config.yaml`的`rewards`部分调整奖励权重，或在`src/environment.py`中修改奖励计算逻辑。

### 自定义环境
继承`AGVEnv`类并重写相关方法来创建新的环境变体。

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小`batch_size`或`d_model`
   - 使用CPU训练：`--device cpu`

2. **训练不收敛**
   - 调整学习率和PPO参数
   - 检查奖励函数设计
   - 启用课程学习

3. **环境渲染问题**
   - 确保安装了matplotlib
   - 在无GUI环境中禁用渲染

### 调试技巧

1. 使用`test_env.py`验证环境功能
2. 检查TensorBoard日志分析训练过程
3. 启用`debug`模式获取详细信息

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

- 作者：Zhang Chao
- 项目：多AGV调度毕业设计
- 环境：biye_RL conda环境

## 致谢

感谢相关开源项目和研究工作的启发，特别是：
- OpenAI Gym/Gymnasium
- PyTorch
- MAPPO算法相关研究
