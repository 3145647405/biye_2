#!/usr/bin/env python3
"""
训练进度监控模块
提供固定位置的训练进度条和实时状态显示
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
import os

@dataclass
class StageInfo:
    """阶段信息"""
    name: str
    num_agvs: int
    num_tasks: int
    target_episodes: int
    target_reward: float
    target_completion_rate: float
    current_episodes: int = 0
    current_reward: float = -999.0
    current_completion_rate: float = 0.0
    completed: bool = False

@dataclass
class TrainingMetrics:
    """训练指标"""
    total_episodes: int = 0
    current_stage: int = 1
    total_stages: int = 9
    current_reward: float = -999.0
    current_completion_rate: float = 0.0
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    training_time: float = 0.0
    
class TrainingProgressMonitor:
    """训练进度监控器 - 简化版"""

    def __init__(self, stages_config: List[Dict], enable_progress_bar: bool = True):
        self.stages_config = stages_config
        self.enable_progress_bar = enable_progress_bar

        # 初始化阶段信息
        self.stages = []
        for i, stage in enumerate(stages_config):
            stage_info = StageInfo(
                name=stage['name'],
                num_agvs=stage['num_agvs'],
                num_tasks=stage['num_tasks'],
                target_episodes=stage['until']['min_episodes'],
                target_reward=stage['until']['min_return'],
                target_completion_rate=stage['until']['min_completion_rate']
            )
            self.stages.append(stage_info)

        # 训练指标
        self.metrics = TrainingMetrics(total_stages=len(self.stages))

        # 控制变量
        self.is_running = False
        self.start_time = None
        self.last_update_time = 0

        # 线程锁
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """开始监控"""
        if not self.enable_progress_bar:
            return

        self.is_running = True
        self.start_time = time.time()

        print("🎯 多AGV调度MAPPO-Attention训练进度监控")
        print("=" * 80)
        print()

        total_episodes = sum(stage.target_episodes for stage in self.stages)
        print(f"📊 总训练目标: {total_episodes} 个回合")
        print(f"🎓 课程学习阶段: {len(self.stages)} 个阶段")
        print()

        print("📋 课程学习阶段概览:")
        for i, stage in enumerate(self.stages, 1):
            status = "🔄" if i == 1 else "⏸️"
            print(f"  {status} 阶段{i}: {stage.name} ({stage.num_agvs}AGV, {stage.num_tasks}任务, 目标{stage.target_episodes}回合)")
        print()
        print("=" * 80)
        
    def update_progress(self, episode: int, stage_idx: int, metrics: Dict[str, Any]):
        """更新进度"""
        if not self.enable_progress_bar or not self.is_running:
            return

        current_time = time.time()
        # 限制更新频率，避免输出过多
        if current_time - self.last_update_time < 2.0:  # 每2秒最多更新一次
            return

        with self.lock:
            # 更新指标
            self.metrics.total_episodes = episode
            self.metrics.current_stage = stage_idx + 1
            self.metrics.current_reward = metrics.get('reward', -999.0)
            self.metrics.current_completion_rate = metrics.get('completion_rate', 0.0)
            self.metrics.actor_loss = metrics.get('actor_loss', 0.0)
            self.metrics.critic_loss = metrics.get('critic_loss', 0.0)
            self.metrics.entropy = metrics.get('entropy', 0.0)

            if self.start_time:
                self.metrics.training_time = time.time() - self.start_time

            # 更新当前阶段信息
            current_stage = self.stages[stage_idx]
            current_stage.current_episodes = episode - sum(s.target_episodes for s in self.stages[:stage_idx])
            current_stage.current_reward = self.metrics.current_reward
            current_stage.current_completion_rate = self.metrics.current_completion_rate

            # 计算总体进度
            total_episodes = sum(stage.target_episodes for stage in self.stages)
            overall_progress = (episode / total_episodes) * 100
            stage_progress = (current_stage.current_episodes / current_stage.target_episodes) * 100

            # 显示进度信息
            print(f"\r🚀 总体进度: {overall_progress:.1f}% ({episode}/{total_episodes}) | "
                  f"阶段{stage_idx+1}: {stage_progress:.1f}% ({current_stage.current_episodes}/{current_stage.target_episodes}) | "
                  f"奖励: {self.metrics.current_reward:.2f} | 完成率: {self.metrics.current_completion_rate:.2f} | "
                  f"时长: {self.metrics.training_time:.0f}s", end="", flush=True)

            self.last_update_time = current_time
    
    def advance_stage(self, new_stage_idx: int):
        """进入下一阶段"""
        if not self.enable_progress_bar or not self.is_running:
            return

        with self.lock:
            # 标记当前阶段完成
            if new_stage_idx > 0:
                self.stages[new_stage_idx - 1].completed = True

            if new_stage_idx < len(self.stages):
                new_stage = self.stages[new_stage_idx]
                print(f"\n\n✅ 阶段{new_stage_idx}完成! 进入阶段{new_stage_idx + 1}: {new_stage.name}")
                print(f"🎯 新阶段目标: {new_stage.num_agvs}个AGV, {new_stage.num_tasks}个任务, {new_stage.target_episodes}回合")
                print("-" * 80)

    def stop_monitoring(self):
        """停止监控"""
        if not self.enable_progress_bar or not self.is_running:
            return

        self.is_running = False

        # 显示最终统计
        print("\n\n" + "=" * 80)
        print("🎉 训练完成统计:")
        print(f"📊 总回合数: {self.metrics.total_episodes}")
        print(f"⏱️  训练时长: {self.metrics.training_time:.1f}秒")
        print(f"🏆 最终奖励: {self.metrics.current_reward:.2f}")
        print(f"✅ 最终完成率: {self.metrics.current_completion_rate:.2f}")

        completed_stages = sum(1 for stage in self.stages if stage.completed)
        print(f"🎯 完成阶段: {completed_stages}/{len(self.stages)}")
        print("=" * 80)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        completed_stages = sum(1 for stage in self.stages if stage.completed)
        
        return {
            "total_episodes": self.metrics.total_episodes,
            "current_stage": self.metrics.current_stage,
            "total_stages": self.metrics.total_stages,
            "completed_stages": completed_stages,
            "current_reward": self.metrics.current_reward,
            "current_completion_rate": self.metrics.current_completion_rate,
            "training_time": self.metrics.training_time,
            "progress_percentage": (self.metrics.total_episodes / sum(s.target_episodes for s in self.stages)) * 100
        }
    
    def display_stage_overview(self):
        """显示阶段概览"""
        if not self.enable_progress_bar:
            return
            
        print("\n📋 训练阶段详细信息:")
        print("-" * 80)
        for i, stage in enumerate(self.stages, 1):
            status = "✅" if stage.completed else ("🔄" if i == self.metrics.current_stage else "⏸️")
            progress = f"{stage.current_episodes}/{stage.target_episodes}" if not stage.completed else "完成"
            
            print(f"{status} 阶段{i}: {stage.name}")
            print(f"    配置: {stage.num_agvs}个AGV, {stage.num_tasks}个任务")
            print(f"    目标: 奖励≥{stage.target_reward}, 完成率≥{stage.target_completion_rate}")
            print(f"    进度: {progress}")
            if not stage.completed and i == self.metrics.current_stage:
                print(f"    当前: 奖励={stage.current_reward:.2f}, 完成率={stage.current_completion_rate:.2f}")
            print()
