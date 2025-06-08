"""
工具函数模块
包含配置加载、日志设置、数据处理等通用功能
"""

import yaml
import logging
import os
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_path} 未找到")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")


def save_config(config: Dict[str, Any], filepath: str):
    """保存配置文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def setup_logging(name: str = "MAPPO-AGV") -> logging.Logger:
    """简单的日志设置函数"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除已有的处理器
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def setup_logging_with_config(config: Dict[str, Any]) -> logging.Logger:
    """设置日志系统"""
    log_config = config.get('logging', {})
    
    # 创建日志目录
    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建logger
    logger = logging.getLogger('MAPPO-AGV')
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 控制台处理器
    if log_config.get('console_log', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_config.get('log_file'):
        file_handler = logging.FileHandler(
            log_dir / log_config['log_file'], 
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42):
    """设置随机种子以确保可重现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(config: Dict[str, Any]):
    """创建必要的目录"""
    dirs_to_create = [
        config['training']['checkpoint_dir'],
        config['training']['log_dir'],
    ]
    
    if config['visualization'].get('save_video', False):
        dirs_to_create.append(config['visualization']['video_dir'])
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """计算两点间的欧几里得距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """计算两点间的曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def normalize_observation(obs: np.ndarray, obs_min: np.ndarray, obs_max: np.ndarray) -> np.ndarray:
    """标准化观测值到[0,1]范围"""
    return (obs - obs_min) / (obs_max - obs_min + 1e-8)


def moving_average(data: List[float], window_size: int = 100) -> List[float]:
    """计算移动平均"""
    if len(data) < window_size:
        return data
    
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        result.append(np.mean(data[start_idx:i+1]))
    
    return result


def save_checkpoint(model_state: Dict[str, Any], filepath: str):
    """保存模型检查点"""
    torch.save(model_state, filepath)


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """加载模型检查点"""
    return torch.load(filepath, map_location='cpu')


class EpisodeBuffer:
    """回合数据缓冲区"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置缓冲区"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.attention_weights = []
    
    def add(self, obs, action, reward, done, value, log_prob, attention_weights=None):
        """添加一步数据"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        if attention_weights is not None:
            self.attention_weights.append(attention_weights)
    
    def get_data(self):
        """获取所有数据"""
        return {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'values': self.values,
            'log_probs': self.log_probs,
            'attention_weights': self.attention_weights
        }
    
    def __len__(self):
        return len(self.observations)


class MetricsTracker:
    """训练指标跟踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.episode_metrics = {}
    
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def update_episode(self, **kwargs):
        """更新回合指标"""
        for key, value in kwargs.items():
            if key not in self.episode_metrics:
                self.episode_metrics[key] = []
            self.episode_metrics[key].append(value)
    
    def get_recent_mean(self, key: str, window: int = 100) -> float:
        """获取最近窗口的平均值"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        
        recent_data = self.metrics[key][-window:]
        return np.mean(recent_data)
    
    def get_episode_mean(self, key: str, window: int = 100) -> float:
        """获取最近回合的平均值"""
        if key not in self.episode_metrics or len(self.episode_metrics[key]) == 0:
            return 0.0
        
        recent_data = self.episode_metrics[key][-window:]
        return np.mean(recent_data)
    
    def reset(self):
        """重置所有指标"""
        self.metrics.clear()
        self.episode_metrics.clear()


def compute_gae(rewards: List[float], values: List[float], dones: List[bool], 
                gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[List[float], List[float]]:
    """计算广义优势估计(GAE)"""
    advantages = []
    returns = []
    
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]
        
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
    
    return advantages, returns
