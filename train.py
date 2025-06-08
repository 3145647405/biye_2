#!/usr/bin/env python3
"""
多AGV调度MAPPO-Attention训练主脚本
支持课程学习和模型保存/加载
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment import AGVEnv
from src.trainer import MAPPOTrainer
from src.utils import load_config, setup_logging, set_seed, create_directories


class CurriculumLearning:
    """课程学习管理器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.curriculum_config = config.get('curriculum', {})
        self.enable = self.curriculum_config.get('enable', False)
        self.stages = self.curriculum_config.get('stages', [])
        self.current_stage = 0
        
        if self.enable:
            self.logger.info(f"课程学习已启用，共 {len(self.stages)} 个阶段")
        else:
            self.logger.info("课程学习已禁用")
    
    def get_current_stage_config(self):
        """获取当前阶段的配置"""
        if not self.enable or self.current_stage >= len(self.stages):
            return self.config
        
        # 复制基础配置
        stage_config = self.config.copy()
        current_stage = self.stages[self.current_stage]
        
        # 更新环境配置
        for key, value in current_stage.items():
            if key != 'until' and key != 'name':
                if key in stage_config['environment']:
                    stage_config['environment'][key] = value
        
        self.logger.info(f"当前阶段: {current_stage.get('name', f'stage_{self.current_stage}')}")
        return stage_config
    
    def should_advance(self, metrics):
        """检查是否应该进入下一阶段"""
        if not self.enable or self.current_stage >= len(self.stages):
            return False
        
        current_stage = self.stages[self.current_stage]
        until_conditions = current_stage.get('until', {})
        
        # 检查所有条件
        for condition, threshold in until_conditions.items():
            if condition == 'min_episodes':
                continue  # 这个在外部检查
            
            metric_value = metrics.get(condition.replace('min_', ''), 0)
            if metric_value < threshold:
                return False
        
        return True
    
    def advance_stage(self):
        """进入下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            stage_name = self.stages[self.current_stage].get('name', f'stage_{self.current_stage}')
            self.logger.info(f"进入下一阶段: {stage_name}")
            return True
        return False
    
    def is_completed(self):
        """检查课程学习是否完成"""
        return self.current_stage >= len(self.stages) - 1


def train_stage(trainer, stage_config, curriculum, writer, stage_episodes=0):
    """训练单个阶段"""
    logger = trainer.logger
    train_config = stage_config['training']
    
    # 阶段参数
    buffer_size = train_config['buffer_size']
    eval_frequency = train_config['eval_frequency']
    save_frequency = train_config['save_frequency']
    
    # 当前阶段的最小回合数
    if curriculum.enable and curriculum.current_stage < len(curriculum.stages):
        current_stage = curriculum.stages[curriculum.current_stage]
        min_episodes = current_stage.get('until', {}).get('min_episodes', 1000)
    else:
        min_episodes = 0
    
    logger.info(f"开始训练阶段，最小回合数: {min_episodes}")
    
    step_count = 0
    last_eval_step = 0
    last_save_step = 0
    
    while True:
        # 收集经验
        start_time = time.time()
        rollout_metrics = trainer.collect_rollouts(buffer_size)
        collect_time = time.time() - start_time
        
        # 更新网络
        start_time = time.time()
        update_metrics = trainer.update()
        update_time = time.time() - start_time
        
        step_count += buffer_size
        stage_episodes += rollout_metrics['episodes_completed']
        
        # 记录指标
        all_metrics = {**rollout_metrics, **update_metrics}
        all_metrics.update({
            'collect_time': collect_time,
            'update_time': update_time,
            'stage_episodes': stage_episodes
        })
        
        # TensorBoard记录
        for key, value in all_metrics.items():
            writer.add_scalar(f'train/{key}', value, trainer.total_steps)
        
        # 日志输出
        if step_count % 1000 == 0:
            logger.info(
                f"步数: {trainer.total_steps}, 回合: {trainer.episode_count}, "
                f"平均奖励: {rollout_metrics['avg_episode_reward']:.2f}, "
                f"完成率: {rollout_metrics['avg_completion_rate']:.2f}, "
                f"Actor损失: {update_metrics['actor_loss']:.4f}"
            )
        
        # 评估
        if trainer.total_steps - last_eval_step >= eval_frequency:
            eval_metrics = trainer.evaluate(train_config['eval_episodes'])
            
            for key, value in eval_metrics.items():
                writer.add_scalar(f'eval/{key}', value, trainer.total_steps)
            
            logger.info(
                f"评估结果 - 平均奖励: {eval_metrics['eval_mean_reward']:.2f}, "
                f"完成率: {eval_metrics['eval_mean_completion_rate']:.2f}"
            )
            
            last_eval_step = trainer.total_steps
        
        # 保存检查点
        if trainer.total_steps - last_save_step >= save_frequency:
            checkpoint_path = Path(train_config['checkpoint_dir']) / f"checkpoint_{trainer.total_steps}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            last_save_step = trainer.total_steps
        
        # 检查课程学习进度
        if curriculum.enable:
            # 检查最小回合数
            if stage_episodes >= min_episodes:
                # 检查其他条件
                recent_metrics = {
                    'return': trainer.metrics.get_episode_mean('episode_reward', 100),
                    'completion_rate': trainer.metrics.get_episode_mean('completion_rate', 100)
                }
                
                if curriculum.should_advance(recent_metrics):
                    logger.info(f"阶段完成，已训练 {stage_episodes} 回合")
                    return stage_episodes
        
        # 检查总训练步数限制
        if trainer.total_steps >= train_config['total_timesteps']:
            logger.info("达到最大训练步数，训练结束")
            return stage_episodes


def main():
    parser = argparse.ArgumentParser(description='多AGV调度MAPPO-Attention训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='设备选择 (cpu/cuda/auto)')
    parser.add_argument('--render', action='store_true', help='启用环境渲染')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建目录
    create_directories(config)
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("开始多AGV调度MAPPO-Attention训练")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"设备: {device}")
    logger.info(f"随机种子: {args.seed}")
    
    # 初始化课程学习
    curriculum = CurriculumLearning(config, logger)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    # 训练循环
    total_stage_episodes = 0
    
    while True:
        # 获取当前阶段配置
        stage_config = curriculum.get_current_stage_config()
        
        # 创建环境
        env = AGVEnv(stage_config)
        if args.render:
            env.render()
        
        # 创建训练器
        if 'trainer' not in locals():
            trainer = MAPPOTrainer(stage_config, env, device)
            
            # 恢复训练
            if args.resume:
                trainer.load_checkpoint(args.resume)
                logger.info(f"从检查点恢复训练: {args.resume}")
        else:
            # 更新环境（保持训练器状态）
            trainer.env = env
            trainer.buffer.num_agents = stage_config['environment']['num_agvs']
        
        # 训练当前阶段
        stage_episodes = train_stage(trainer, stage_config, curriculum, writer, total_stage_episodes)
        total_stage_episodes = stage_episodes
        
        # 检查是否继续下一阶段
        if curriculum.enable and not curriculum.is_completed():
            if curriculum.advance_stage():
                continue
        
        # 训练完成
        break
    
    # 保存最终模型
    final_checkpoint = Path(config['training']['checkpoint_dir']) / "final_model.pt"
    trainer.save_checkpoint(str(final_checkpoint))
    logger.info(f"最终模型已保存到: {final_checkpoint}")
    
    # 关闭
    writer.close()
    env.close()
    logger.info("训练完成")


if __name__ == "__main__":
    main()
