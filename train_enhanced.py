#!/usr/bin/env python3
"""
增强版多AGV调度MAPPO-Attention训练脚本
支持动态可视化、时间戳文件夹和完整的指标收集
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time
import shutil
from datetime import datetime
from typing import Dict, Any

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment import AGVEnv
from src.trainer import MAPPOTrainer
from src.utils import load_config, setup_logging, set_seed, create_directories
from src.visualization import TrainingVisualizer, EnhancedMetricsCollector
import matplotlib.pyplot as plt


def create_timestamped_results_dir(config: Dict[str, Any], script_name: str) -> Path:
    """创建带时间戳的结果目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path(config['training']['results_dir'])
    results_dir = results_root / f"{script_name}_{timestamp}"
    
    # 创建目录结构
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    (results_dir / "data").mkdir(exist_ok=True)
    
    # 复制配置文件
    if config['training'].get('save_config_copy', True):
        shutil.copy2('config.yaml', results_dir / 'config.yaml')
    
    return results_dir


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
            for i, stage in enumerate(self.stages):
                self.logger.info(f"  阶段{i+1}: {stage['name']} - {stage['num_agvs']}个AGV, {stage['num_tasks']}个任务")
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
            if key not in ['until', 'name']:
                if key in stage_config['environment']:
                    stage_config['environment'][key] = value
        
        self.logger.info(f"当前阶段: {current_stage.get('name', f'stage_{self.current_stage}')}")
        self.logger.info(f"  环境配置: {current_stage['num_agvs']}个AGV, {current_stage['num_tasks']}个任务, 地图{current_stage.get('map_width', 26)}x{current_stage.get('map_height', 10)}")
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
                self.logger.info(f"阶段晋级条件未满足: {condition} = {metric_value:.3f} < {threshold}")
                return False
        
        self.logger.info("所有阶段晋级条件已满足")
        return True
    
    def advance_stage(self):
        """进入下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            stage_name = self.stages[self.current_stage].get('name', f'stage_{self.current_stage}')
            self.logger.info(f"🎉 进入下一阶段: {stage_name}")
            return True
        return False
    
    def is_completed(self):
        """检查课程学习是否完成"""
        return self.current_stage >= len(self.stages) - 1


def train_stage(trainer, stage_config, curriculum, writer, visualizer, metrics_collector, 
                results_dir, stage_episodes=0):
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
        stage_name = current_stage.get('name', f'stage_{curriculum.current_stage}')
    else:
        min_episodes = 0
        stage_name = "final_stage"
    
    logger.info(f"🚀 开始训练阶段: {stage_name}")
    logger.info(f"   最小回合数: {min_episodes}")
    logger.info(f"   环境配置: {stage_config['environment']['num_agvs']}个AGV, {stage_config['environment']['num_tasks']}个任务")
    
    step_count = 0
    last_eval_step = 0
    last_save_step = 0
    last_plot_update = 0
    
    # 可视化配置
    plot_update_freq = stage_config.get('visualization', {}).get('plot_update_frequency', 20)
    
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
        
        # 收集详细指标
        if rollout_metrics['episodes_completed'] > 0:
            # 获取最新的环境信息
            env_info = trainer.env._get_info()
            
            # 收集回合指标
            episode_metrics = metrics_collector.collect_episode_metrics(
                trainer.env, 
                trainer.episode_count,
                rollout_metrics['avg_episode_reward'],
                rollout_metrics['avg_episode_length'],
                env_info
            )
            
            # 添加到可视化器
            visualizer.add_episode_data(
                episode=trainer.episode_count,
                reward=rollout_metrics['avg_episode_reward'],
                actor_loss=update_metrics['actor_loss'],
                critic_loss=update_metrics['critic_loss'],
                entropy=update_metrics['entropy'],
                completion_rate=rollout_metrics['avg_completion_rate'],
                load_utilization=episode_metrics.get('load_utilization', 0.0),
                path_length=episode_metrics.get('path_length', 0.0),
                collision_count=episode_metrics.get('collision_count', 0),
                episode_length=rollout_metrics['avg_episode_length']
            )
        
        # 收集步级指标
        step_metrics = metrics_collector.collect_step_metrics(
            trainer.total_steps,
            update_metrics['actor_loss'],
            update_metrics['critic_loss'],
            update_metrics['entropy'],
            train_config['learning_rate']
        )
        
        # 记录指标
        all_metrics = {**rollout_metrics, **update_metrics}
        all_metrics.update({
            'collect_time': collect_time,
            'update_time': update_time,
            'stage_episodes': stage_episodes,
            'current_stage': curriculum.current_stage,
            'stage_name': stage_name
        })
        
        # TensorBoard记录
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):  # 只记录数值类型
                writer.add_scalar(f'train/{key}', value, trainer.total_steps)
        
        # 更新可视化图表
        if (trainer.episode_count - last_plot_update >= plot_update_freq and 
            trainer.episode_count > 5):
            try:
                # 创建并保存最新的训练图表
                fig = visualizer.create_training_plots()
                plot_path = results_dir / "plots" / "training_progress_live.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # 保存数据
                data_path = results_dir / "data" / "training_data.json"
                visualizer.save_data_to_json(str(data_path))
                
                last_plot_update = trainer.episode_count
                logger.info(f"📊 训练图表已更新 (回合: {trainer.episode_count})")
                
            except Exception as e:
                logger.warning(f"可视化更新失败: {e}")
        
        # 日志输出
        if step_count % 1000 == 0:
            logger.info(
                f"📈 步数: {trainer.total_steps:,}, 回合: {trainer.episode_count:,}, "
                f"平均奖励: {rollout_metrics['avg_episode_reward']:.2f}, "
                f"完成率: {rollout_metrics['avg_completion_rate']:.2f}, "
                f"Actor损失: {update_metrics['actor_loss']:.4f}, "
                f"阶段进度: {stage_episodes}/{min_episodes}"
            )
        
        # 评估
        if trainer.total_steps - last_eval_step >= eval_frequency:
            eval_metrics = trainer.evaluate(train_config['eval_episodes'])
            
            for key, value in eval_metrics.items():
                writer.add_scalar(f'eval/{key}', value, trainer.total_steps)
            
            logger.info(
                f"🎯 评估结果 - 平均奖励: {eval_metrics['eval_mean_reward']:.2f}, "
                f"完成率: {eval_metrics['eval_mean_completion_rate']:.2f}"
            )
            
            last_eval_step = trainer.total_steps
        
        # 保存检查点
        if trainer.total_steps - last_save_step >= save_frequency:
            checkpoint_path = results_dir / "checkpoints" / f"checkpoint_{trainer.total_steps}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            last_save_step = trainer.total_steps
        
        # 检查课程学习进度
        if curriculum.enable:
            # 检查最小回合数
            if stage_episodes >= min_episodes:
                # 检查其他条件
                recent_metrics = metrics_collector.get_recent_metrics(100)
                
                if curriculum.should_advance(recent_metrics):
                    logger.info(f"✅ 阶段 {stage_name} 完成，已训练 {stage_episodes} 回合")
                    
                    # 保存阶段总结
                    stage_summary_path = results_dir / f"stage_{curriculum.current_stage}_summary.json"
                    visualizer.create_summary_report(str(stage_summary_path))
                    
                    return stage_episodes
        
        # 检查总训练步数限制
        if trainer.total_steps >= train_config['total_timesteps']:
            logger.info("⏰ 达到最大训练步数，训练结束")
            return stage_episodes


def main():
    parser = argparse.ArgumentParser(description='增强版多AGV调度MAPPO-Attention训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='设备选择 (cpu/cuda/auto)')
    parser.add_argument('--render', action='store_true', help='启用环境渲染')
    parser.add_argument('--no-visualization', action='store_true', help='禁用实时可视化')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"🚀 使用设备: {device}")

    # 设置随机种子
    set_seed(args.seed)

    # 创建带时间戳的结果目录
    script_name = "mappo_attention_training"
    results_dir = create_timestamped_results_dir(config, script_name)
    print(f"📁 结果将保存到: {results_dir}")

    # 更新配置中的路径
    config['training']['checkpoint_dir'] = str(results_dir / "checkpoints")
    config['training']['log_dir'] = str(results_dir / "logs")

    # 创建目录
    create_directories(config)

    # 设置日志
    logger = setup_logging("MAPPO-AGV")
    logger.info("🎯 开始增强版多AGV调度MAPPO-Attention训练")
    logger.info(f"📋 配置文件: {args.config}")
    logger.info(f"💻 设备: {device}")
    logger.info(f"🎲 随机种子: {args.seed}")
    logger.info(f"📂 结果目录: {results_dir}")

    # 初始化课程学习
    curriculum = CurriculumLearning(config, logger)

    # 初始化可视化器
    visualizer = TrainingVisualizer(config, save_dir=str(results_dir / "plots"))
    if not args.no_visualization:
        plot_update_freq = config.get('visualization', {}).get('plot_update_frequency', 20)
        visualizer.start_realtime_visualization(plot_update_freq)

    # 初始化指标收集器
    metrics_collector = EnhancedMetricsCollector(config)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(results_dir / "logs"))

    # 训练循环
    total_stage_episodes = 0
    training_start_time = time.time()

    try:
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
                    logger.info(f"📥 从检查点恢复训练: {args.resume}")
            else:
                # 更新环境（保持训练器状态）
                trainer.env = env
                trainer.buffer.num_agents = stage_config['environment']['num_agvs']

            # 训练当前阶段
            stage_episodes = train_stage(
                trainer, stage_config, curriculum, writer,
                visualizer, metrics_collector, results_dir, total_stage_episodes
            )
            total_stage_episodes = stage_episodes

            # 检查是否继续下一阶段
            if curriculum.enable and not curriculum.is_completed():
                if curriculum.advance_stage():
                    continue

            # 训练完成
            break

    except KeyboardInterrupt:
        logger.info("⚠️ 训练被用户中断")
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 停止实时可视化
        if not args.no_visualization:
            visualizer.stop_realtime_visualization()

        # 保存最终模型
        final_checkpoint = results_dir / "checkpoints" / "final_model.pt"
        if 'trainer' in locals():
            trainer.save_checkpoint(str(final_checkpoint))
            logger.info(f"💾 最终模型已保存到: {final_checkpoint}")

        # 生成最终训练图表
        try:
            final_plot = visualizer.create_training_plots()
            final_plot_path = results_dir / "plots" / "final_training_results.png"
            final_plot.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(final_plot)
            logger.info(f"📊 最终训练图表已保存到: {final_plot_path}")
        except Exception as e:
            logger.warning(f"生成最终图表失败: {e}")

        # 保存最终数据和报告
        try:
            final_data_path = results_dir / "data" / "final_training_data.json"
            visualizer.save_data_to_json(str(final_data_path))

            final_report_path = results_dir / "training_summary_report.json"
            visualizer.create_summary_report(str(final_report_path))

            logger.info(f"📋 训练数据已保存到: {final_data_path}")
            logger.info(f"📄 训练报告已保存到: {final_report_path}")
        except Exception as e:
            logger.warning(f"保存最终数据失败: {e}")

        # 计算总训练时间
        total_training_time = time.time() - training_start_time
        hours = int(total_training_time // 3600)
        minutes = int((total_training_time % 3600) // 60)
        seconds = int(total_training_time % 60)

        logger.info(f"⏱️ 总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
        logger.info(f"🎉 训练完成！所有结果已保存到: {results_dir}")

        # 关闭资源
        writer.close()
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    main()
