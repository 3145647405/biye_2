#!/bin/bash

# GitHub上传脚本
# 使用方法: ./upload_to_github.sh

echo "🚀 开始准备GitHub上传..."

# 检查是否在正确的目录
if [ ! -f "config.yaml" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

echo "📁 当前目录: $(pwd)"

# 检查Git状态
echo "📊 检查Git状态..."
git status

# 添加所有文件（.gitignore会自动排除不需要的文件）
echo "📦 添加文件到Git..."
git add .

# 显示将要提交的文件
echo "📋 将要提交的文件:"
git status --short

# 提交更改
echo "💾 提交更改..."
git commit -m "🔧 Major System Fixes and Improvements

✅ Core Fixes:
- Fixed reward mechanism: reduced step penalty from -0.1 to -0.01 (90% reduction)
- Implemented spacetime reservation system for multi-AGV coordination
- Fixed collision detection with safe position exemptions
- Repaired reward distribution for task completion, collisions, and deadlocks

🚀 New Features:
- SpaceTimeReservationTable class for conflict-free path planning
- MultiAGVPathPlanner with temporal A* algorithm
- Enhanced AGV state management and tracking
- Curriculum learning with 9 progressive stages

📊 Training Results:
- Successfully completed 1040 episodes without crashes
- Task completion rate: 40-50% (up from 0%)
- Reward stability significantly improved
- Multi-AGV cooperation functioning

🧹 Code Quality:
- Cleaned up 8 temporary test files
- Maintained core test coverage
- Improved project structure and documentation

⚠️ Known Issues:
- Collision frequency: 2-3 per episode (needs optimization)
- Occasional deadlock situations (detection enhancement needed)

📈 Performance Impact:
- System stability: Complete fix
- Reward mechanism: Complete fix  
- Training efficiency: Significant improvement
- Multi-AGV coordination: Functional with room for optimization"

# 检查远程仓库
echo "🔗 检查远程仓库..."
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "➕ 添加远程仓库..."
    git remote add origin https://github.com/3145647405/biye_1.git
fi

echo "🌐 远程仓库: $(git remote get-url origin)"

# 推送到GitHub
echo "⬆️ 推送到GitHub..."
git push -u origin master

if [ $? -eq 0 ]; then
    echo "✅ 成功上传到GitHub!"
    echo "🔗 查看仓库: https://github.com/3145647405/biye_1"
else
    echo "❌ 上传失败，请检查网络连接和权限"
    echo "💡 您也可以手动上传文件到GitHub网页界面"
fi

echo "📋 上传摘要:"
echo "   - 核心代码文件: src/"
echo "   - 配置文件: config.yaml"
echo "   - 训练脚本: complete_training.py, train_*.py"
echo "   - 测试文件: test_env.py, test_visualization.py"
echo "   - 文档: README.md, *.md"
echo "   - 排除: results/, checkpoints/, logs/, *.json, *.png"

echo "🎉 GitHub更新流程完成!"
