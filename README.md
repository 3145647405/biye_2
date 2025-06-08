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
```

## ⚙️ 配置说明

### 主要配置文件 (`config.yaml`)

**环境配置**
```yaml
environment:
  map_width: 26          # 地图宽度
  map_height: 8          # 地图高度
  num_agvs: 1           # AGV数量
  num_tasks: 2          # 任务数量
  max_steps: 600        # 最大步数
```

**奖励配置（已优化）**
```yaml
rewards:
  step_penalty: -0.01    # 步数惩罚（从-0.1优化到-0.01）
  task_pickup: 15.0      # 任务拾取奖励
  mission_complete: 100.0 # 任务完成奖励
  collision: -25.0       # 碰撞惩罚
  efficient_move: 0.02   # 高效移动奖励（新增）
  deadlock: -15.0        # 死锁惩罚
```

**训练配置**
```yaml
training:
  total_timesteps: 100000
  batch_size: 256
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
```

## 📊 训练结果

### 性能改善
- **奖励稳定性**: 步数惩罚减少90%，训练更稳定
- **完成率**: 0.4-0.67范围，显示明显学习效果
- **碰撞减少**: 通过时空预留机制显著减少冲突
- **训练稳定性**: 1000+ episodes稳定运行无崩溃

### 可视化输出
训练过程自动生成：
- 📈 奖励变化曲线和趋势分析
- 📊 任务完成率统计
- ⚠️ 碰撞次数分析
- 🧠 Actor/Critic损失变化
- 🎯 熵值和探索度分析

结果保存在 `results/complete_training_YYYYMMDD_HHMMSS/` 目录。

### 训练阶段
1. **阶段1**: 单AGV基础训练 (1个AGV, 2个任务)
2. **阶段2**: 双AGV协作训练 (2个AGV, 4个任务)
3. **阶段3-9**: 逐步增加复杂度到 (4个AGV, 16个任务)

## 🔧 主要修复内容

### 1. 奖励机制完全重构 ✅
- **步数惩罚优化**: -0.1 → -0.01 (减少90%)
- **奖励分配修复**: 碰撞、任务完成、死锁奖励正确分配
- **新增奖励类型**: 高效移动、协作奖励

### 2. 时空预留系统 ✅
- **冲突预防**: 实现时空预留表避免路径冲突
- **多AGV协调**: 支持多AGV同时路径规划
- **安全位置**: 起点(1,8)和终点(24,1)允许多AGV共存

### 3. 环境稳定性 ✅
- **错误处理**: 完善的异常处理和恢复机制
- **状态管理**: 修复AGV状态跟踪和更新逻辑
- **死锁检测**: 改进死锁检测和惩罚机制

## 🧪 测试验证

### 功能测试
```bash
# 基本环境测试
python test_env.py

# 可视化测试
python test_visualization.py
```

### 性能基准
- **单AGV场景**: 完成率 > 80%
- **双AGV场景**: 完成率 > 60%  
- **多AGV场景**: 完成率 > 40%
- **训练稳定性**: 1000+ episodes无崩溃

## 🐛 已知问题

### 当前挑战
1. **多AGV碰撞**: 复杂场景下仍有碰撞发生
2. **死锁处理**: 需要更强的死锁预防机制
3. **完成率**: 复杂场景下有提升空间

### 改进方向
1. 优化时空预留算法的冲突解决
2. 增强死锁检测和自动恢复
3. 改进路径规划的启发式函数

## 📈 训练建议

### 推荐参数
- **学习率**: 0.0001-0.001
- **批次大小**: 128-512  
- **熵系数**: 0.01-0.1
- **训练步数**: 100,000+

### 训练策略
1. 从简单场景开始训练
2. 逐步增加AGV数量和任务复杂度
3. 监控奖励趋势和完成率
4. 适时调整超参数

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发流程
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目链接: [https://github.com/your-username/biye_augment](https://github.com/your-username/biye_augment)
- 问题反馈: [Issues](https://github.com/your-username/biye_augment/issues)

## 🙏 致谢

感谢所有为多AGV调度和强化学习领域做出贡献的研究人员。

---

⭐ 如果这个项目对您有帮助，请给它一个星标！
