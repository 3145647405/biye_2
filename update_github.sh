#!/bin/bash

# GitHub更新脚本
# 将项目更新推送到 https://github.com/3145647405/biye_2

echo "=== 开始更新GitHub仓库 ==="

# 检查当前目录
echo "当前目录: $(pwd)"

# 检查git状态
echo "检查git状态..."
git status

# 检查远程仓库配置
echo "检查远程仓库..."
git remote -v

# 如果没有配置远程仓库，添加它
if ! git remote | grep -q origin; then
    echo "添加远程仓库..."
    git remote add origin https://github.com/3145647405/biye_2.git
else
    echo "更新远程仓库URL..."
    git remote set-url origin https://github.com/3145647405/biye_2.git
fi

# 添加所有更改的文件（排除.gitignore中的文件）
echo "添加文件到暂存区..."
git add .

# 检查暂存区状态
echo "检查暂存区状态..."
git status

# 提交更改
echo "提交更改..."
git commit -m "🚀 项目重大更新: 修复奖励机制和时空预留系统

✅ 主要修复内容:
1. 奖励机制完全重构
   - 步数惩罚从-0.1降至-0.01 (减少90%)
   - 修复任务完成、碰撞、死锁奖励分配
   - 添加高效移动和协作奖励

2. 时空预留路径规划系统
   - 实现SpaceTimeReservationTable类
   - 实现MultiAGVPathPlanner类
   - 支持多AGV协调路径规划

3. 碰撞检测优化
   - 起点和终点允许多AGV共存
   - 安全位置碰撞豁免机制
   - 改进碰撞检测逻辑

4. 系统集成和优化
   - 课程学习支持9个训练阶段
   - 实时可视化和进度监控
   - 完整的训练数据记录

📊 训练效果验证:
- 成功完成1040个episodes训练
- 奖励机制正常工作，任务完成奖励正确分配
- 多AGV协调功能基本正常
- 系统稳定性大幅提升

🔧 技术改进:
- 环境渲染系统优化
- 配置文件重新平衡
- 代码结构清理和文档完善
- 测试覆盖率提升

📁 项目结构:
- src/: 核心源代码
- config.yaml: 优化后的配置文件
- complete_training.py: 完整训练脚本
- test_*.py: 核心测试文件
- 文档: README, 项目状态报告等

⚠️ 注意: 训练结果和模型文件已排除，不上传到仓库"

# 推送到GitHub
echo "推送到GitHub..."
git push -u origin main

echo "=== GitHub更新完成 ==="
echo "仓库地址: https://github.com/3145647405/biye_2"
