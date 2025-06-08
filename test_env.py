#!/usr/bin/env python3
"""
环境测试脚本
验证AGV环境的基本功能
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment import AGVEnv
from src.utils import load_config


def test_environment():
    """测试环境基本功能"""
    print("开始测试AGV环境...")
    
    # 加载配置
    config = load_config('config.yaml')
    
    # 创建环境
    env = AGVEnv(config)
    print(f"环境创建成功")
    print(f"地图大小: {env.map_width} x {env.map_height}")
    print(f"AGV数量: {env.num_agvs}")
    print(f"任务数量: {env.num_tasks}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"环境重置成功")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 打印初始状态
    print("\n初始状态:")
    for agv_id, agv_obs in obs.items():
        print(f"AGV {agv_id}:")
        print(f"  自身状态: {agv_obs['agv_own_state']}")
        print(f"  附近任务数: {np.sum(np.any(agv_obs['nearby_tasks_state'] != 0, axis=1))}")
        print(f"  附近AGV数: {np.sum(np.any(agv_obs['nearby_agvs_state'] != 0, axis=1))}")
    
    # 运行几步
    print("\n运行测试步骤...")
    total_reward = 0
    
    for step in range(20):
        # 随机动作
        actions = {}
        for agv_id in range(env.num_agvs):
            actions[agv_id] = np.random.randint(0, env.action_space.n)
        
        # 执行动作
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_reward += step_reward
        
        print(f"步骤 {step + 1}: 奖励 = {step_reward:.2f}, 累计奖励 = {total_reward:.2f}")
        print(f"  完成率: {info['completion_rate']:.2f}")
        print(f"  碰撞次数: {info['episode_stats']['collisions']}")
        
        # 渲染环境
        if step % 5 == 0:
            env.render()
            plt.pause(0.5)
        
        if terminated or truncated:
            print(f"回合结束于步骤 {step + 1}")
            break
    
    print(f"\n测试完成，总奖励: {total_reward:.2f}")
    
    # 关闭环境
    env.close()


def test_path_planning():
    """测试路径规划功能"""
    print("\n测试路径规划...")
    
    config = load_config('config.yaml')
    env = AGVEnv(config)
    
    # 测试A*路径规划
    start = (1, 8)  # 起点
    goal = (24, 1)  # 终点
    
    path = env.planner.plan_path(start, goal)
    
    if path:
        print(f"路径规划成功，路径长度: {len(path)}")
        print(f"起点: {start}, 终点: {goal}")
        print(f"路径前5步: {path[:5]}")
        print(f"路径后5步: {path[-5:]}")
    else:
        print("路径规划失败")
    
    env.close()


def test_attention_models():
    """测试注意力模型"""
    print("\n测试注意力模型...")
    
    import torch
    from src.models import AttentionActor, AttentionCritic
    
    config = load_config('config.yaml')
    device = torch.device('cpu')
    
    # 创建模型
    actor = AttentionActor(config).to(device)
    critic = AttentionCritic(config).to(device)
    
    print(f"Actor参数数量: {sum(p.numel() for p in actor.parameters())}")
    print(f"Critic参数数量: {sum(p.numel() for p in critic.parameters())}")
    
    # 创建测试数据
    batch_size = 4
    obs_dict = {
        'agv_own_state': torch.randn(batch_size, 5),
        'nearby_tasks_state': torch.randn(batch_size, 10, 4),
        'nearby_agvs_state': torch.randn(batch_size, 5, 2)
    }
    
    # 测试Actor前向传播
    with torch.no_grad():
        actor_output = actor(obs_dict)
        print(f"Actor输出形状:")
        for key, value in actor_output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
    
    # 测试Critic前向传播
    # 计算正确的全局状态维度
    num_agvs = config['environment']['num_agvs']
    num_tasks = config['environment']['num_tasks']
    global_state_dim = num_agvs * 5 + num_tasks * 6

    global_state = torch.randn(batch_size, global_state_dim)
    with torch.no_grad():
        critic_output = critic(global_state)
        print(f"Critic输出形状: {critic_output.shape}")
        print(f"全局状态维度: {global_state_dim}")
    
    print("模型测试完成")


def main():
    """主测试函数"""
    try:
        # 测试环境
        test_environment()
        
        # 测试路径规划
        test_path_planning()
        
        # 测试模型
        test_attention_models()
        
        print("\n所有测试通过！")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
