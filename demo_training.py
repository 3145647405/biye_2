#!/usr/bin/env python3
"""
完整训练演示脚本
展示优化后的可视化系统：单一PNG图片，每100个episode更新，平滑的loss曲线
"""

import os
import sys
import time
import yaml
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加src目录到路径
sys.path.append('src')

from visualization import TrainingVisualizer

def demo_complete_training():
    """演示完整的训练过程"""
    
    print("🎯 开始完整训练过程演示")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(f"results/demo_complete_training_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 结果目录: {result_dir}")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化可视化器 - 每100个episode更新一次
    visualizer = TrainingVisualizer(config, str(result_dir / 'plots'))
    visualizer.start_realtime_visualization(update_frequency=100)
    
    print("📊 可视化系统已启动 - 每100个episode更新一次，只生成一张PNG图片")
    
    # 课程学习阶段
    stages = config['curriculum']['stages']
    print(f"🎓 课程学习已启用，共 {len(stages)} 个阶段")
    
    total_episodes = 0
    
    try:
        for stage_idx, stage in enumerate(stages, 1):
            stage_name = stage['name']
            num_agvs = stage['num_agvs']
            num_tasks = stage['num_tasks']
            
            print(f"\n🚀 开始训练阶段 {stage_idx}/9: {stage_name}")
            print(f"   配置: {num_agvs}个AGV, {num_tasks}个任务")
            
            # 阶段训练参数
            stage_episodes = stage['until']['min_episodes']
            target_completion_rate = stage['until']['min_completion_rate']
            target_return = stage['until']['min_return']
            
            print(f"   目标: {stage_episodes}回合, 完成率≥{target_completion_rate}, 奖励≥{target_return}")
            
            # 模拟训练过程
            for episode in range(stage_episodes):
                total_episodes += 1
                
                # 模拟学习进度
                progress = episode / stage_episodes
                
                # 模拟奖励改善（更平滑的曲线）
                base_reward = target_return - 15
                reward_improvement = 15 * (1 - np.exp(-progress * 3))  # 指数增长
                noise = np.random.normal(0, 1)  # 减少噪声
                episode_reward = base_reward + reward_improvement + noise
                
                # 模拟完成率改善
                completion_rate = min(target_completion_rate + 0.1, 
                                    progress * (target_completion_rate + 0.2) + np.random.normal(0, 0.02))
                completion_rate = max(0, completion_rate)
                
                # 模拟平滑的loss曲线
                actor_loss = 0.8 * np.exp(-progress * 2) + 0.1 + np.random.normal(0, 0.05)
                critic_loss = 0.5 * np.exp(-progress * 1.8) + 0.05 + np.random.normal(0, 0.03)
                entropy = 1.2 * np.exp(-progress * 1.5) + 0.2 + np.random.normal(0, 0.05)
                
                # 确保loss值为正
                actor_loss = max(0.01, actor_loss)
                critic_loss = max(0.01, critic_loss)
                entropy = max(0.01, entropy)
                
                # 模拟AGV特定指标
                load_utilization = min(0.95, progress * 0.7 + 0.2 + np.random.normal(0, 0.05))
                path_length = max(8, 40 - progress * 15 + np.random.normal(0, 2))
                collision_count = max(0, int(8 * (1 - progress) + np.random.normal(0, 0.5)))
                
                # 记录数据到可视化器
                visualizer.add_episode_data(
                    episode=total_episodes,
                    reward=episode_reward,
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    entropy=entropy,
                    completion_rate=completion_rate,
                    load_utilization=load_utilization,
                    path_length=path_length,
                    collision_count=collision_count
                )
                
                # 每10个回合显示进度
                if (episode + 1) % 10 == 0:
                    print(f"   阶段{stage_idx} 回合{episode+1}/{stage_episodes}: "
                          f"奖励={episode_reward:.2f}, 完成率={completion_rate:.2f}, "
                          f"Actor Loss={actor_loss:.3f}, Critic Loss={critic_loss:.3f}")
                
                # 检查晋级条件
                if (completion_rate >= target_completion_rate and 
                    episode_reward >= target_return and
                    episode >= stage_episodes * 0.6):  # 至少完成60%训练
                    print(f"✅ 阶段 {stage_idx} 提前完成! "
                          f"完成率={completion_rate:.2f}, 奖励={episode_reward:.2f}")
                    break
                
                # 模拟训练延迟
                time.sleep(0.01)  # 很短的延迟以便观察
            
            print(f"🎉 阶段 {stage_idx} 训练完成! 总回合数: {total_episodes}")
            time.sleep(1)  # 阶段间短暂休息
    
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    finally:
        # 停止可视化
        visualizer.stop_realtime_visualization()
        
        # 生成最终报告
        final_report = {
            "training_completed": True,
            "total_episodes": total_episodes,
            "total_stages_completed": stage_idx,
            "result_directory": str(result_dir),
            "single_visualization_file": str(visualizer.single_plot_file),
            "improvements": [
                "单一PNG图片可视化",
                "每100个episode更新一次",
                "平滑的Actor/Critic loss曲线",
                "完整的9阶段课程学习"
            ]
        }
        
        # 保存最终报告
        import json
        with open(result_dir / 'final_training_report.json', 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        # 保存训练数据
        visualizer.save_data_to_json(result_dir / 'training_data.json')
        
        # 生成最终可视化
        visualizer.create_training_plots(result_dir / 'final_training_summary.png')
        
        print("\n🎉 完整训练过程演示完成!")
        print(f"📊 总回合数: {total_episodes}")
        print(f"📈 完成阶段数: {stage_idx}/{len(stages)}")
        print(f"📁 所有结果已保存到: {result_dir}")
        print(f"🖼️  单一训练可视化图片: {visualizer.single_plot_file}")
        print("\n✨ 主要改进:")
        print("   ✅ 只生成一张PNG图片，每100个episode更新")
        print("   ✅ Actor和Critic loss曲线更加平滑")
        print("   ✅ 完整的9阶段课程学习演示")
        print("   ✅ 实时可视化系统优化")
        
        return result_dir

if __name__ == "__main__":
    demo_complete_training()
