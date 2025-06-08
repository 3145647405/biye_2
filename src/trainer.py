"""
MAPPO训练器实现
包含经验回放、优势估计、策略更新等核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging

from models import AttentionActor, AttentionCritic
from utils import compute_gae, MetricsTracker


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, buffer_size: int, num_agents: int, obs_space: Dict, device: torch.device):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # 初始化缓冲区
        self.observations = {}
        for key, space in obs_space.items():
            self.observations[key] = torch.zeros(
                (buffer_size, num_agents, *space.shape), 
                dtype=torch.float32, device=device
            )
        
        self.actions = torch.zeros((buffer_size, num_agents), dtype=torch.long, device=device)
        self.rewards = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_agents), dtype=torch.bool, device=device)
        self.global_states = torch.zeros((buffer_size, 0), dtype=torch.float32, device=device)  # 动态大小
        
        # GAE相关
        self.advantages = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
    
    def add(self, obs: Dict[int, Dict[str, np.ndarray]], actions: Dict[int, int], 
            rewards: Dict[int, float], values: Dict[int, float], 
            log_probs: Dict[int, float], dones: Dict[int, bool], 
            global_state: np.ndarray):
        """添加一步经验"""
        # 转换观测
        for key in self.observations.keys():
            obs_tensor = torch.stack([
                torch.from_numpy(obs[i][key]) for i in range(self.num_agents)
            ])
            self.observations[key][self.ptr] = obs_tensor.to(self.device)
        
        # 转换其他数据
        self.actions[self.ptr] = torch.tensor([actions[i] for i in range(self.num_agents)], device=self.device)
        self.rewards[self.ptr] = torch.tensor([rewards[i] for i in range(self.num_agents)], device=self.device)
        self.values[self.ptr] = torch.tensor([values[i] for i in range(self.num_agents)], device=self.device)
        self.log_probs[self.ptr] = torch.tensor([log_probs[i] for i in range(self.num_agents)], device=self.device)
        self.dones[self.ptr] = torch.tensor([dones[i] for i in range(self.num_agents)], device=self.device)
        
        # 全局状态
        if self.ptr == 0:
            # 第一次添加时确定全局状态大小
            global_state_size = len(global_state)
            self.global_states = torch.zeros((self.buffer_size, global_state_size), 
                                           dtype=torch.float32, device=self.device)
        
        self.global_states[self.ptr] = torch.from_numpy(global_state).to(self.device)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(self, last_values: Dict[int, float], gamma: float = 0.99, gae_lambda: float = 0.95):
        """计算广义优势估计"""
        last_values_tensor = torch.tensor([last_values[i] for i in range(self.num_agents)], device=self.device)
        
        for agent_id in range(self.num_agents):
            rewards = self.rewards[:self.size, agent_id].cpu().numpy()
            values = self.values[:self.size, agent_id].cpu().numpy()
            dones = self.dones[:self.size, agent_id].cpu().numpy()
            
            # 添加最后一个值
            values_with_last = np.append(values, last_values_tensor[agent_id].cpu().numpy())
            
            advantages = []
            returns = []
            gae = 0
            
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    next_value = last_values_tensor[agent_id].item()
                else:
                    next_value = values_with_last[i + 1]
                
                delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
                gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[i])
            
            self.advantages[:self.size, agent_id] = torch.tensor(advantages, device=self.device)
            self.returns[:self.size, agent_id] = torch.tensor(returns, device=self.device)
    
    def get_batch(self, batch_size: int):
        """获取批次数据"""
        indices = torch.randperm(self.size * self.num_agents)[:batch_size]
        
        # 展平数据
        flat_obs = {}
        for key, obs_tensor in self.observations.items():
            flat_obs[key] = obs_tensor[:self.size].reshape(-1, *obs_tensor.shape[2:])
        
        flat_actions = self.actions[:self.size].reshape(-1)
        flat_values = self.values[:self.size].reshape(-1)
        flat_log_probs = self.log_probs[:self.size].reshape(-1)
        flat_advantages = self.advantages[:self.size].reshape(-1)
        flat_returns = self.returns[:self.size].reshape(-1)
        
        # 选择批次
        batch_obs = {key: obs[indices] for key, obs in flat_obs.items()}
        batch_actions = flat_actions[indices]
        batch_old_values = flat_values[indices]
        batch_old_log_probs = flat_log_probs[indices]
        batch_advantages = flat_advantages[indices]
        batch_returns = flat_returns[indices]
        
        # 标准化优势
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        
        return (batch_obs, batch_actions, batch_old_values, 
                batch_old_log_probs, batch_advantages, batch_returns)
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0


class MAPPOTrainer:
    """MAPPO训练器"""
    
    def __init__(self, config: Dict[str, Any], env, device: torch.device):
        self.config = config
        self.env = env
        self.device = device
        self.logger = logging.getLogger('MAPPO-AGV')
        
        # 训练参数
        train_config = config['training']
        self.batch_size = train_config['batch_size']
        self.mini_batch_size = train_config['mini_batch_size']
        self.num_epochs = train_config['num_epochs']
        self.clip_epsilon = train_config['clip_epsilon']
        self.value_loss_coef = train_config['value_loss_coef']
        self.entropy_coef = train_config['entropy_coef']
        self.max_grad_norm = train_config['max_grad_norm']
        self.gamma = train_config['gamma']
        self.gae_lambda = train_config['gae_lambda']
        
        # 网络初始化
        self.actor = AttentionActor(config).to(device)
        self.critic = AttentionCritic(config).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=train_config['learning_rate'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=train_config['learning_rate'])
        
        # 经验缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=train_config['buffer_size'],
            num_agents=config['environment']['num_agvs'],
            obs_space=env.observation_space,
            device=device
        )
        
        # 指标跟踪
        self.metrics = MetricsTracker()
        
        # 训练状态
        self.total_steps = 0
        self.episode_count = 0
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """收集经验"""
        self.actor.eval()
        self.critic.eval()
        
        obs, info = self.env.reset()
        episode_rewards = defaultdict(float)
        episode_lengths = 0
        episodes_completed = 0
        
        for step in range(num_steps):
            # 获取动作和价值
            with torch.no_grad():
                obs_tensors = self._convert_obs_to_tensors(obs)
                
                # Actor前向传播
                actions = {}
                values = {}
                log_probs = {}
                
                for agent_id in range(self.env.num_agvs):
                    agent_obs = {key: tensor[agent_id:agent_id+1] for key, tensor in obs_tensors.items()}
                    action, log_prob, entropy, value = self.actor.get_action_and_value(agent_obs)
                    
                    actions[agent_id] = action.item()
                    values[agent_id] = value.item()
                    log_probs[agent_id] = log_prob.item()
                
                # Critic前向传播（全局状态）
                global_state = torch.from_numpy(self.env.get_global_state()).unsqueeze(0).to(self.device)
                # global_value = self.critic(global_state).item()
            
            # 环境步进
            next_obs, rewards, terminated, truncated, next_info = self.env.step(actions)
            
            # 记录经验
            dones = {i: terminated or truncated for i in range(self.env.num_agvs)}
            self.buffer.add(obs, actions, rewards, values, log_probs, dones, self.env.get_global_state())
            
            # 更新统计
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            episode_lengths += 1
            
            # 检查回合结束
            if terminated or truncated:
                episodes_completed += 1
                self.episode_count += 1
                
                # 记录回合指标
                avg_reward = np.mean(list(episode_rewards.values()))
                self.metrics.update_episode(
                    episode_reward=avg_reward,
                    episode_length=episode_lengths,
                    completion_rate=next_info.get('completion_rate', 0)
                )
                
                # 重置环境
                obs, info = self.env.reset()
                episode_rewards = defaultdict(float)
                episode_lengths = 0
            else:
                obs = next_obs
                info = next_info
            
            self.total_steps += 1
        
        # 计算最后的价值（用于GAE）
        with torch.no_grad():
            obs_tensors = self._convert_obs_to_tensors(obs)
            last_values = {}
            for agent_id in range(self.env.num_agvs):
                agent_obs = {key: tensor[agent_id:agent_id+1] for key, tensor in obs_tensors.items()}
                _, _, _, value = self.actor.get_action_and_value(agent_obs)
                last_values[agent_id] = value.item()
        
        # 计算GAE
        self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)
        
        return {
            'episodes_completed': episodes_completed,
            'avg_episode_reward': self.metrics.get_episode_mean('episode_reward'),
            'avg_episode_length': self.metrics.get_episode_mean('episode_length'),
            'avg_completion_rate': self.metrics.get_episode_mean('completion_rate')
        }
    
    def _convert_obs_to_tensors(self, obs: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """将观测转换为张量"""
        obs_tensors = {}
        for key in obs[0].keys():
            obs_list = [obs[i][key] for i in range(self.env.num_agvs)]
            obs_tensors[key] = torch.from_numpy(np.stack(obs_list)).to(self.device)
        return obs_tensors

    def update(self) -> Dict[str, float]:
        """更新网络参数"""
        self.actor.train()
        self.critic.train()

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        update_count = 0

        # 多轮更新
        for epoch in range(self.num_epochs):
            # 获取批次数据
            batch_data = self.buffer.get_batch(self.batch_size)
            (batch_obs, batch_actions, batch_old_values,
             batch_old_log_probs, batch_advantages, batch_returns) = batch_data

            # 分小批次更新
            num_mini_batches = max(1, self.batch_size // self.mini_batch_size)
            mini_batch_size = self.batch_size // num_mini_batches

            for i in range(num_mini_batches):
                start_idx = i * mini_batch_size
                end_idx = start_idx + mini_batch_size

                mini_batch_obs = {key: obs[start_idx:end_idx] for key, obs in batch_obs.items()}
                mini_batch_actions = batch_actions[start_idx:end_idx]
                mini_batch_old_values = batch_old_values[start_idx:end_idx]
                mini_batch_old_log_probs = batch_old_log_probs[start_idx:end_idx]
                mini_batch_advantages = batch_advantages[start_idx:end_idx]
                mini_batch_returns = batch_returns[start_idx:end_idx]

                # Actor前向传播
                _, new_log_probs, entropy, new_values = self.actor.get_action_and_value(
                    mini_batch_obs, mini_batch_actions
                )

                # 计算比率
                ratio = torch.exp(new_log_probs - mini_batch_old_log_probs)

                # PPO裁剪损失
                surr1 = ratio * mini_batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mini_batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 熵损失
                entropy_loss = -entropy.mean()

                # 价值损失（Actor网络中的价值头）
                value_loss = F.mse_loss(new_values, mini_batch_returns)

                # 总损失
                total_loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # 反向传播
                self.actor_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 累计损失
                total_actor_loss += actor_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1

        # 清空缓冲区
        self.buffer.clear()

        # 记录训练指标
        metrics = {
            'actor_loss': total_actor_loss / update_count,
            'critic_loss': total_critic_loss / update_count,
            'entropy': total_entropy / update_count,
            'total_steps': self.total_steps,
            'episodes': self.episode_count
        }

        self.metrics.update(**metrics)

        return metrics

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"检查点已保存到 {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']

        self.logger.info(f"检查点已从 {filepath} 加载")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估模型性能"""
        self.actor.eval()
        self.critic.eval()

        episode_rewards = []
        episode_lengths = []
        completion_rates = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                with torch.no_grad():
                    obs_tensors = self._convert_obs_to_tensors(obs)
                    actions = {}

                    for agent_id in range(self.env.num_agvs):
                        agent_obs = {key: tensor[agent_id:agent_id+1] for key, tensor in obs_tensors.items()}
                        action, _ = self.actor.get_action(agent_obs, deterministic=True)
                        actions[agent_id] = action.item()

                obs, rewards, terminated, truncated, info = self.env.step(actions)

                episode_reward += sum(rewards.values())
                episode_length += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            completion_rates.append(info.get('completion_rate', 0))

        return {
            'eval_mean_reward': np.mean(episode_rewards),
            'eval_std_reward': np.std(episode_rewards),
            'eval_mean_length': np.mean(episode_lengths),
            'eval_mean_completion_rate': np.mean(completion_rates)
        }
