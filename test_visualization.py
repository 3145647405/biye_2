#!/usr/bin/env python3
"""
测试可视化功能
验证动态图表生成和数据收集功能
"""

import os
import sys
import numpy as np
import time
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config
from src.visualization import TrainingVisualizer, EnhancedMetricsCollector


def test_visualization():
    """测试可视化功能"""
    print("🧪 开始测试可视化功能...")
    
    # 加载配置
    config = load_config('config.yaml')
    
    # 创建测试目录
    test_dir = Path("test_plots")
    test_dir.mkdir(exist_ok=True)
    
    # 初始化可视化器
    visualizer = TrainingVisualizer(config, save_dir=str(test_dir))
    
    # 生成模拟训练数据
    print("📊 生成模拟训练数据...")
    
    num_episodes = 200
    for episode in range(1, num_episodes + 1):
        # 模拟训练过程中的指标变化
        
        # 奖励逐渐提升（带噪声）
        base_reward = -50 + (episode / num_episodes) * 100
        reward = base_reward + np.random.normal(0, 10)
        
        # 损失逐渐下降
        actor_loss = 2.0 * np.exp(-episode / 50) + np.random.normal(0, 0.1)
        critic_loss = 1.5 * np.exp(-episode / 40) + np.random.normal(0, 0.08)
        
        # 熵逐渐下降
        entropy = 1.0 * np.exp(-episode / 80) + 0.1 + np.random.normal(0, 0.05)
        
        # 完成率逐渐提升
        completion_rate = min(1.0, (episode / num_episodes) * 1.2 + np.random.normal(0, 0.1))
        completion_rate = max(0.0, completion_rate)
        
        # 载重利用率逐渐提升
        load_utilization = min(1.0, (episode / num_episodes) * 0.8 + np.random.normal(0, 0.05))
        load_utilization = max(0.0, load_utilization)
        
        # 路径长度逐渐优化
        path_length = 100 - (episode / num_episodes) * 30 + np.random.normal(0, 5)
        path_length = max(20, path_length)
        
        # 碰撞次数逐渐减少
        collision_count = max(0, int(10 * np.exp(-episode / 30) + np.random.poisson(1)))
        
        # 回合长度
        episode_length = int(200 + np.random.normal(0, 20))
        
        # 添加数据到可视化器
        visualizer.add_episode_data(
            episode=episode,
            reward=reward,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            entropy=entropy,
            completion_rate=completion_rate,
            load_utilization=load_utilization,
            path_length=path_length,
            collision_count=collision_count,
            episode_length=episode_length
        )
        
        # 每20个episode更新一次图表
        if episode % 20 == 0:
            print(f"  Episode {episode}: 奖励={reward:.2f}, 完成率={completion_rate:.2f}")
    
    print("✅ 模拟数据生成完成")
    
    # 创建训练图表
    print("🎨 生成训练图表...")
    fig = visualizer.create_training_plots()
    
    # 保存图表
    plot_path = test_dir / "test_training_plots.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 训练图表已保存到: {plot_path}")
    
    # 保存数据
    data_path = test_dir / "test_training_data.json"
    visualizer.save_data_to_json(str(data_path))
    print(f"💾 训练数据已保存到: {data_path}")
    
    # 生成总结报告
    report_path = test_dir / "test_training_report.json"
    report = visualizer.create_summary_report(str(report_path))
    print(f"📄 训练报告已保存到: {report_path}")
    
    # 显示报告摘要
    if report:
        summary = report['training_summary']
        print("\n📋 训练总结:")
        print(f"  总回合数: {summary['total_episodes']}")
        print(f"  最终平均奖励: {summary['final_performance']['avg_reward_last_100']:.2f}")
        print(f"  最终完成率: {summary['final_performance']['avg_completion_rate_last_100']:.2f}")
        print(f"  最佳奖励: {summary['final_performance']['best_reward']:.2f}")
        print(f"  奖励提升: {summary['learning_progress']['reward_improvement']:.2f}")
    
    print("✅ 可视化功能测试完成")


def test_metrics_collector():
    """测试指标收集器"""
    print("\n🧪 测试指标收集器...")
    
    config = load_config('config.yaml')
    collector = EnhancedMetricsCollector(config)
    
    # 模拟环境信息
    class MockEnv:
        def __init__(self):
            self.max_load = 25
    
    mock_env = MockEnv()
    
    # 收集一些模拟指标
    for episode in range(1, 21):
        episode_reward = -20 + episode * 2 + np.random.normal(0, 5)
        episode_length = 150 + np.random.randint(-20, 20)
        
        # 模拟环境信息
        info = {
            'completion_rate': min(1.0, episode / 20 + np.random.normal(0, 0.1)),
            'agv_states': [(0, 0, np.random.uniform(0, 25)) for _ in range(3)],
            'episode_stats': {
                'collisions': np.random.randint(0, 5),
                'deadlocks': np.random.randint(0, 2)
            }
        }
        
        # 收集回合指标
        metrics = collector.collect_episode_metrics(
            mock_env, episode, episode_reward, episode_length, info
        )
        
        print(f"  Episode {episode}: 奖励={metrics['episode_reward']:.2f}, "
              f"完成率={metrics['completion_rate']:.2f}, "
              f"载重利用率={metrics['load_utilization']:.2f}")
    
    # 获取最近指标
    recent_metrics = collector.get_recent_metrics(10)
    print(f"\n📊 最近10回合平均指标:")
    for key, value in recent_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    print("✅ 指标收集器测试完成")


def test_realtime_visualization():
    """测试实时可视化"""
    print("\n🧪 测试实时可视化...")
    
    config = load_config('config.yaml')
    test_dir = Path("test_realtime")
    test_dir.mkdir(exist_ok=True)
    
    visualizer = TrainingVisualizer(config, save_dir=str(test_dir))
    
    # 启动实时可视化
    visualizer.start_realtime_visualization(update_frequency=5)
    
    print("🔄 开始实时数据更新...")
    
    # 模拟实时训练数据
    for episode in range(1, 31):
        reward = -30 + episode + np.random.normal(0, 5)
        actor_loss = 1.0 * np.exp(-episode / 10) + np.random.normal(0, 0.1)
        critic_loss = 0.8 * np.exp(-episode / 8) + np.random.normal(0, 0.05)
        entropy = 0.5 * np.exp(-episode / 15) + 0.1 + np.random.normal(0, 0.02)
        completion_rate = min(1.0, episode / 30 + np.random.normal(0, 0.05))
        
        visualizer.add_episode_data(
            episode=episode,
            reward=reward,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            entropy=entropy,
            completion_rate=completion_rate
        )
        
        print(f"  添加Episode {episode}数据")
        time.sleep(0.1)  # 模拟训练间隔
    
    # 等待最后一次更新
    time.sleep(2)
    
    # 停止实时可视化
    visualizer.stop_realtime_visualization()
    
    print("✅ 实时可视化测试完成")


def main():
    """主测试函数"""
    try:
        # 测试基础可视化功能
        test_visualization()
        
        # 测试指标收集器
        test_metrics_collector()
        
        # 测试实时可视化
        test_realtime_visualization()
        
        print("\n🎉 所有可视化测试通过！")
        print("📁 测试结果保存在 test_plots/ 和 test_realtime/ 目录中")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
