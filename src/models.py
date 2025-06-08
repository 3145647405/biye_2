"""
注意力增强的MAPPO神经网络模型
包含多层次注意力机制的Actor和Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiLevelAttentionEncoder(nn.Module):
    """多层次注意力编码器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.num_heads = config['model']['num_attention_heads']
        self.dropout = config['model']['attention_dropout']
        
        # 特征投影层
        self.task_projection = nn.Linear(4, self.d_model)
        self.agv_projection = nn.Linear(5, self.d_model)
        self.other_agv_projection = nn.Linear(2, self.d_model)
        
        # 位置编码
        self.position_encoding = PositionalEncoding(self.d_model)
        
        # 任务级自注意力
        self.task_self_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 智能体级交叉注意力
        self.agent_cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 层归一化
        self.task_norm = nn.LayerNorm(self.d_model)
        self.agent_norm = nn.LayerNorm(self.d_model)
        
        # 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
        
        # 最终投影层
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = obs_dict['agv_own_state'].shape[0]
        
        # 1. 特征嵌入
        own_state_emb = self.agv_projection(obs_dict['agv_own_state'])  # (B, d_model)
        task_emb = self.task_projection(obs_dict['nearby_tasks_state'])  # (B, K, d_model)
        agv_emb = self.other_agv_projection(obs_dict['nearby_agvs_state'])  # (B, M, d_model)
        
        # 2. 位置编码（仅对任务）
        task_emb = self.position_encoding(task_emb)
        
        # 3. 任务级自注意力
        task_att_out, task_att_weights = self.task_self_attention(
            task_emb, task_emb, task_emb
        )
        task_att_out = self.task_norm(task_att_out + task_emb)  # 残差连接
        
        # 聚合任务特征
        task_mask = torch.sum(obs_dict['nearby_tasks_state'], dim=-1) != 0  # (B, K)
        task_context = self._masked_mean(task_att_out, task_mask)  # (B, d_model)
        
        # 4. 智能体级交叉注意力
        query = own_state_emb.unsqueeze(1)  # (B, 1, d_model)
        
        if agv_emb.shape[1] > 0:  # 如果有其他AGV
            agent_att_out, agent_att_weights = self.agent_cross_attention(
                query, agv_emb, agv_emb
            )
            agent_context = agent_att_out.squeeze(1)  # (B, d_model)
        else:
            agent_context = torch.zeros_like(own_state_emb)
            agent_att_weights = None
        
        # 5. 特征融合
        combined_features = torch.cat([task_context, agent_context], dim=-1)
        fusion_weights = self.fusion_gate(combined_features)
        
        final_context = (
            fusion_weights * task_context +
            (1 - fusion_weights) * agent_context
        )
        
        # 6. 最终投影
        output_context = self.output_projection(final_context)
        
        return {
            'context': output_context,
            'task_attention': task_att_weights,
            'agent_attention': agent_att_weights,
            'fusion_weights': fusion_weights
        }
    
    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """计算掩码平均值"""
        mask = mask.unsqueeze(-1).float()  # (B, K, 1)
        masked_tensor = tensor * mask
        sum_tensor = torch.sum(masked_tensor, dim=1)  # (B, d_model)
        count = torch.sum(mask, dim=1).clamp(min=1)  # (B, 1)
        return sum_tensor / count


class AttentionActor(nn.Module):
    """注意力增强的Actor网络"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['model']['d_model']
        
        # 多层次注意力编码器
        self.attention_encoder = MultiLevelAttentionEncoder(config)
        
        # Actor网络
        hidden_dims = config['model']['actor_hidden_dims']
        layers = []
        input_dim = self.d_model
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.actor_net = nn.Sequential(*layers)
        
        # 动作输出层
        action_dim = config['environment']['max_nearby_tasks'] + 1
        self.action_head = nn.Linear(input_dim, action_dim)
        
        # 动态权重输出层（多目标优化）
        self.weight_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4个目标权重
            nn.Softmax(dim=-1)
        )
        
        # 价值函数输出层（用于Actor-Critic）
        self.value_head = nn.Linear(input_dim, 1)
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 获取注意力编码结果
        attention_output = self.attention_encoder(obs_dict)
        context = attention_output['context']
        
        # 通过Actor网络
        features = self.actor_net(context)
        
        # 生成动作概率分布
        action_logits = self.action_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 生成动态目标权重
        objective_weights = self.weight_head(features)
        
        # 生成价值估计
        value = self.value_head(features)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'objective_weights': objective_weights,
            'value': value,
            'attention_info': attention_output
        }
    
    def get_action_and_value(self, obs_dict: Dict[str, torch.Tensor], 
                           action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作和价值（用于训练）"""
        output = self.forward(obs_dict)
        
        action_logits = output['action_logits']
        value = output['value']
        
        # 创建动作分布
        probs = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_action(self, obs_dict: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """获取动作（用于执行）"""
        with torch.no_grad():
            output = self.forward(obs_dict)
            action_probs = output['action_probs']
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                probs = torch.distributions.Categorical(action_probs)
                action = probs.sample()
            
            return action, output


class AttentionCritic(nn.Module):
    """注意力增强的Critic网络（中心化）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['model']['d_model']

        # 计算全局状态维度
        num_agvs = config['environment']['num_agvs']
        num_tasks = config['environment']['num_tasks']
        global_state_dim = num_agvs * 5 + num_tasks * 6  # AGV状态5维，任务状态6维

        # 动态计算全局状态维度（用于处理不同配置）
        self.global_state_dim = global_state_dim

        # 全局状态编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_state_dim, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU()
        )
        
        # Critic网络
        hidden_dims = config['model']['critic_hidden_dims']
        layers = []
        input_dim = self.d_model
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.critic_net = nn.Sequential(*layers)
        
        # 价值函数输出
        self.value_head = nn.Linear(input_dim, 1)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        # 编码全局状态
        encoded_state = self.global_encoder(global_state)
        
        # 通过Critic网络
        features = self.critic_net(encoded_state)
        
        # 输出价值
        value = self.value_head(features)
        
        return value.squeeze(-1)
