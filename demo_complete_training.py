#!/usr/bin/env python3
"""
完整训练过程演示脚本
展示从第一阶段到最后阶段的完整课程学习过程
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

from environment import MultiAGVEnvironment
from models import MAPPOAgent
from trainer import MAPPOTrainer
from visualization import TrainingVisualizer
from utils import setup_logging, save_config

def demo_complete_training():
    """演示完整的训练过程"""
    
    # 设置日志
    logger = setup_logging("MAPPO-AGV-DEMO")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(f"results/demo_complete_training_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🎯 开始完整训练过程演示")
    logger.info(f"📂 结果目录: {result_dir}")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 保存配置副本
    save_config(config, result_dir / 'config.yaml')
    
    # 初始化可视化器
    visualizer = TrainingVisualizer(result_dir / 'plots')
    
    # 课程学习阶段
    stages = config['curriculum']['stages']
    logger.info(f"课程学习已启用，共 {len(stages)} 个阶段")
    
    total_episodes = 0
    
    for stage_idx, stage in enumerate(stages, 1):
        stage_name = stage['name']
        num_agvs = stage['num_agvs']
        num_tasks = stage['num_tasks']
        max_steps = stage['max_steps']
        map_width = stage['map_width']
        map_height = stage['map_height']
        
        logger.info(f"🚀 开始训练阶段 {stage_idx}/9: {stage_name}")
        logger.info(f"   配置: {num_agvs}个AGV, {num_tasks}个任务, 地图{map_width}x{map_height}")
        
        # 创建环境
        env = MultiAGVEnvironment(
            num_agvs=num_agvs,
            num_tasks=num_tasks,
            map_width=map_width,
            map_height=map_height,
            max_steps=max_steps
        )
        
        # 模拟训练过程
        stage_episodes = stage['until']['min_episodes']
        target_completion_rate = stage['until']['min_completion_rate']
        target_return = stage['until']['min_return']
        
        logger.info(f"   目标: {stage_episodes}回合, 完成率≥{target_completion_rate}, 奖励≥{target_return}")
        
        # 模拟训练数据
        for episode in range(stage_episodes):
            total_episodes += 1
            
            # 模拟学习进度
            progress = episode / stage_episodes
            
            # 模拟奖励改善
            base_reward = target_return - 20
            reward_improvement = 20 * progress
            episode_reward = base_reward + reward_improvement + np.random.normal(0, 2)
            
            # 模拟完成率改善
            completion_rate = min(target_completion_rate + 0.1, progress * (target_completion_rate + 0.2))
            completion_rate = max(0, completion_rate + np.random.normal(0, 0.05))
            
            # 模拟其他指标
            actor_loss = 0.5 * np.exp(-progress * 2) + np.random.normal(0, 0.1)
            critic_loss = 0.3 * np.exp(-progress * 1.5) + np.random.normal(0, 0.05)
            entropy = 1.0 * np.exp(-progress * 1) + np.random.normal(0, 0.1)
            
            # 模拟AGV特定指标
            load_utilization = min(0.9, progress * 0.8 + np.random.normal(0, 0.1))
            path_length = max(10, 50 - progress * 20 + np.random.normal(0, 3))
            collision_count = max(0, 10 * (1 - progress) + np.random.normal(0, 1))
            
            # 记录数据
            visualizer.record_episode(
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
            
            # 每5个回合更新一次可视化
            if (episode + 1) % 5 == 0:
                visualizer.update_plots()
                logger.info(f"   回合 {episode+1}/{stage_episodes}: 奖励={episode_reward:.2f}, 完成率={completion_rate:.2f}")
        
        # 检查晋级条件
        current_completion_rate = completion_rate
        current_reward = episode_reward
        
        if current_completion_rate >= target_completion_rate and current_reward >= target_return:
            logger.info(f"✅ 阶段 {stage_idx} 完成! 完成率={current_completion_rate:.2f}, 奖励={current_reward:.2f}")
        else:
            logger.info(f"⚠️  阶段 {stage_idx} 未完全达标，但继续下一阶段进行演示")
        
        # 保存阶段检查点
        checkpoint_path = result_dir / 'checkpoints' / f'stage_{stage_idx}_checkpoint.pt'
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        # 模拟保存检查点
        logger.info(f"💾 阶段检查点已保存: {checkpoint_path}")
        
        time.sleep(1)  # 短暂暂停以便观察
    
    # 生成最终报告
    final_report = {
        "training_completed": True,
        "total_episodes": total_episodes,
        "total_stages": len(stages),
        "final_performance": {
            "reward": float(episode_reward),
            "completion_rate": float(completion_rate),
            "load_utilization": float(load_utilization)
        },
        "training_time": f"{time.time():.2f} seconds (demo)",
        "result_directory": str(result_dir)
    }
    
    # 保存最终报告
    import json
    with open(result_dir / 'final_training_report.json', 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    # 生成最终可视化
    visualizer.create_training_plots(result_dir / 'final_training_summary.png')
    
    logger.info("🎉 完整训练过程演示完成!")
    logger.info(f"📊 总回合数: {total_episodes}")
    logger.info(f"📈 最终性能: 奖励={episode_reward:.2f}, 完成率={completion_rate:.2f}")
    logger.info(f"📁 所有结果已保存到: {result_dir}")
    
    return result_dir

if __name__ == "__main__":
    demo_complete_training()
