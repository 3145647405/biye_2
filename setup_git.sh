#!/bin/bash

# 设置Git配置
echo "设置Git配置..."
git config --global user.name "Zhang Chao"
git config --global user.email "your-email@example.com"

# 进入项目目录
cd "/home/public/桌面/zhangchao/biye_augment"

# 检查Git状态
echo "检查Git状态..."
git status

# 添加所有文件
echo "添加文件到Git..."
git add .

# 提交更改
echo "提交更改..."
git commit -m "🚀 Initial commit: Multi-AGV Scheduling System with MAPPO

✨ Features:
- Fixed reward mechanism (step penalty reduced by 90%)
- Space-time reservation system for multi-AGV coordination
- Collision exemption at start/end points
- Curriculum learning with 9 progressive stages
- Real-time visualization and monitoring
- Complete training pipeline

🔧 Major fixes:
- Reward distribution logic for collisions, task completion, deadlock
- AGV state tracking and update mechanisms
- Environment stability and error handling
- Time-space reservation table implementation

📊 Training results:
- 1000+ episodes stable training
- Completion rate: 0.4-0.67
- Significant collision reduction
- Proper reward signal recovery

🧪 Tested and verified:
- Environment functionality
- Visualization system
- Multi-AGV coordination
- Curriculum learning progression"

echo "Git设置完成！"
echo "现在您可以添加远程仓库并推送："
echo "git remote add origin https://github.com/your-username/biye_augment.git"
echo "git branch -M main"
echo "git push -u origin main"
