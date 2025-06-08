"""
训练过程可视化模块
提供实时训练指标监控、散点图绘制和曲线拟合功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import threading
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体 - 改进的字体设置
def setup_chinese_fonts():
    """设置中文字体，提供多种回退选项"""
    import matplotlib.font_manager as fm

    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei',           # Windows 黑体
        'Microsoft YaHei',  # Windows 微软雅黑
        'PingFang SC',      # macOS 苹方
        'Hiragino Sans GB', # macOS 冬青黑体
        'WenQuanYi Micro Hei', # Linux 文泉驿微米黑
        'Noto Sans CJK SC', # Google Noto字体
        'DejaVu Sans'       # 最后回退到英文字体
    ]

    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 找到第一个可用的中文字体
    selected_font = 'DejaVu Sans'  # 默认回退
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    print(f"选择字体: {selected_font}")

    # 设置matplotlib字体
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    return selected_font

# 初始化字体设置
setup_chinese_fonts()

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, config: Dict[str, Any], save_dir: str = "plots"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.episode_data = {
            'episodes': [],
            'rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'completion_rates': [],
            'load_utilizations': [],
            'path_lengths': [],
            'collision_counts': [],
            'episode_lengths': [],
            'timestamps': []
        }
        
        # 步级数据存储
        self.step_data = {
            'steps': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'learning_rates': [],
            'timestamps': []
        }
        
        # 可视化设置 - 按用户要求每100个episode更新一次
        viz_config = config.get('visualization', {})
        self.update_interval = viz_config.get('plot_update_frequency', 100)
        self.enable_realtime = viz_config.get('enable_realtime_plots', True)
        self.save_plots = viz_config.get('save_plots', True)

        # 单一PNG文件设置
        self.single_plot_file = self.save_dir / "training_visualization.png"

        # Loss平滑处理
        self.loss_smoothing_factor = 0.9  # 指数移动平均系数
        self.smoothed_actor_loss = None
        self.smoothed_critic_loss = None
        
        # 图形对象
        self.fig = None
        self.axes = None
        self.is_running = False
        self.update_thread = None
        
        # 曲线拟合函数
        self.fit_functions = {
            'linear': lambda x, a, b: a * x + b,
            'exponential': lambda x, a, b, c: a * np.exp(b * x) + c,
            'polynomial': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
            'moving_average': self._moving_average
        }
    
    def add_episode_data(self, episode: int, reward: float, actor_loss: float,
                        critic_loss: float, entropy: float, completion_rate: float,
                        load_utilization: float = 0.0, path_length: float = 0.0,
                        collision_count: int = 0, episode_length: int = 0):
        """添加回合数据，包含loss平滑处理"""

        # Loss平滑处理 - 使用指数移动平均
        if self.smoothed_actor_loss is None:
            self.smoothed_actor_loss = actor_loss
            self.smoothed_critic_loss = critic_loss
        else:
            self.smoothed_actor_loss = (self.loss_smoothing_factor * self.smoothed_actor_loss +
                                      (1 - self.loss_smoothing_factor) * actor_loss)
            self.smoothed_critic_loss = (self.loss_smoothing_factor * self.smoothed_critic_loss +
                                       (1 - self.loss_smoothing_factor) * critic_loss)

        self.episode_data['episodes'].append(episode)
        self.episode_data['rewards'].append(reward)
        self.episode_data['actor_losses'].append(self.smoothed_actor_loss)  # 使用平滑后的loss
        self.episode_data['critic_losses'].append(self.smoothed_critic_loss)  # 使用平滑后的loss
        self.episode_data['entropies'].append(entropy)
        self.episode_data['completion_rates'].append(completion_rate)
        self.episode_data['load_utilizations'].append(load_utilization)
        self.episode_data['path_lengths'].append(path_length)
        self.episode_data['collision_counts'].append(collision_count)
        self.episode_data['episode_lengths'].append(episode_length)
        self.episode_data['timestamps'].append(time.time())
    
    def add_step_data(self, step: int, actor_loss: float, critic_loss: float, 
                     entropy: float, learning_rate: float = 0.0):
        """添加步级数据"""
        self.step_data['steps'].append(step)
        self.step_data['actor_losses'].append(actor_loss)
        self.step_data['critic_losses'].append(critic_loss)
        self.step_data['entropies'].append(entropy)
        self.step_data['learning_rates'].append(learning_rate)
        self.step_data['timestamps'].append(time.time())
    
    def _moving_average(self, data: np.ndarray, window: int = 50) -> np.ndarray:
        """计算移动平均"""
        if len(data) < window:
            return data
        
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            result[i] = np.mean(data[start_idx:i+1])
        
        return result
    
    def fit_curve(self, x_data: np.ndarray, y_data: np.ndarray, 
                  method: str = 'moving_average') -> Tuple[np.ndarray, Dict[str, Any]]:
        """拟合曲线"""
        if len(x_data) < 10:  # 数据点太少，返回原数据
            return y_data, {'method': method, 'params': None}
        
        try:
            if method == 'moving_average':
                fitted_y = self._moving_average(y_data, window=min(50, len(y_data)//4))
                return fitted_y, {'method': method, 'window': min(50, len(y_data)//4)}
            
            elif method == 'linear':
                popt, _ = curve_fit(self.fit_functions['linear'], x_data, y_data)
                fitted_y = self.fit_functions['linear'](x_data, *popt)
                return fitted_y, {'method': method, 'params': popt.tolist()}
            
            elif method == 'polynomial':
                # 使用numpy的多项式拟合
                coeffs = np.polyfit(x_data, y_data, deg=min(3, len(x_data)//10))
                fitted_y = np.polyval(coeffs, x_data)
                return fitted_y, {'method': method, 'params': coeffs.tolist()}
            
            elif method == 'exponential':
                # 指数拟合需要特殊处理
                try:
                    popt, _ = curve_fit(self.fit_functions['exponential'], x_data, y_data, 
                                      maxfev=1000)
                    fitted_y = self.fit_functions['exponential'](x_data, *popt)
                    return fitted_y, {'method': method, 'params': popt.tolist()}
                except:
                    # 如果指数拟合失败，回退到移动平均
                    fitted_y = self._moving_average(y_data)
                    return fitted_y, {'method': 'moving_average_fallback', 'params': None}
            
        except Exception as e:
            print(f"曲线拟合失败 ({method}): {e}")
            # 回退到移动平均
            fitted_y = self._moving_average(y_data)
            return fitted_y, {'method': 'moving_average_fallback', 'params': None}
        
        return y_data, {'method': 'none', 'params': None}
    
    def create_training_plots(self, save_path: Optional[str] = None) -> plt.Figure:
        """创建训练过程图表 - 优化大数据量显示"""
        # 创建子图 - 调整布局和大小以适应大数据量
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('Multi-AGV MAPPO-Attention Training Visualization (12K Episodes)',
                    fontsize=18, fontweight='bold', y=0.98)

        # 检查数据是否足够
        if len(self.episode_data['episodes']) < 2:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Insufficient Data\nNeed More Training Episodes',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig

        episodes = np.array(self.episode_data['episodes'])
        print(f"绘制图表: {len(episodes)} 个回合的数据")

        # 1. 回合奖励散点图和拟合曲线 - 优化散点样式
        ax = axes[0, 0]
        rewards = np.array(self.episode_data['rewards'])

        # 优化散点图样式 - 扩大散点半径一倍，线宽加粗0.5倍
        point_size = max(4, min(8, 200 / len(rewards) * 15)) * 2  # 散点半径扩大一倍
        inner_alpha = max(0.6, min(0.9, 800 / len(rewards)))  # 内部透明度
        edge_alpha = max(0.2, min(0.5, 400 / len(rewards)))   # 边缘透明度

        # 使用不同颜色 - 深蓝色散点
        ax.scatter(episodes, rewards, alpha=inner_alpha, s=point_size, c='#2E4A7A',
                  label='Episode Rewards', edgecolors='#1A2B42', linewidths=0.45,
                  rasterized=True)
        print(f"奖励散点图: {len(rewards)} 个数据点 (点大小: {point_size:.1f}, 内部透明度: {inner_alpha:.2f})")

        # 拟合曲线 - 使用与散点不同的颜色
        if len(rewards) > 5:  # 降低阈值确保能显示拟合曲线
            try:
                fitted_rewards, _ = self.fit_curve(episodes, rewards, 'moving_average')
                ax.plot(episodes, fitted_rewards, '#FF6B35', linewidth=2.5, label='Moving Average', alpha=0.9)
                print(f"移动平均拟合完成")

                # 添加线性趋势线
                if len(rewards) > 10:
                    linear_fit, _ = self.fit_curve(episodes, rewards, 'linear')
                    ax.plot(episodes, linear_fit, '#4CAF50', linestyle='--', linewidth=2, alpha=0.8, label='Linear Trend')
                    print(f"线性趋势拟合完成")
            except Exception as e:
                print(f"拟合曲线生成失败: {e}")

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Episode Reward', fontsize=11)
        ax.set_title('Episode Reward Progress', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 2. Actor损失散点图 - 优化样式
        ax = axes[0, 1]
        actor_losses = np.array(self.episode_data['actor_losses'])
        ax.scatter(episodes, actor_losses, alpha=inner_alpha, s=point_size, color='#D84315',
                  label='Actor Loss', edgecolors='#BF360C', linewidths=0.45, rasterized=True)

        if len(actor_losses) > 5:
            try:
                fitted_losses, _ = self.fit_curve(episodes, actor_losses, 'moving_average')
                ax.plot(episodes, fitted_losses, '#FF9800', linewidth=2.5, label='Moving Average', alpha=0.9)
            except Exception as e:
                print(f"Actor损失拟合失败: {e}")

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Actor Loss', fontsize=11)
        ax.set_title('Actor Network Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 3. Critic损失散点图 - 优化样式
        ax = axes[0, 2]
        critic_losses = np.array(self.episode_data['critic_losses'])
        ax.scatter(episodes, critic_losses, alpha=inner_alpha, s=point_size, color='#2E7D32',
                  label='Critic Loss', edgecolors='#1B5E20', linewidths=0.45, rasterized=True)

        if len(critic_losses) > 5:
            try:
                fitted_losses, _ = self.fit_curve(episodes, critic_losses, 'moving_average')
                ax.plot(episodes, fitted_losses, '#8BC34A', linewidth=2.5, label='Moving Average', alpha=0.9)
            except Exception as e:
                print(f"Critic损失拟合失败: {e}")

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Critic Loss', fontsize=11)
        ax.set_title('Critic Network Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 4. 熵值散点图 - 优化样式
        ax = axes[1, 0]
        entropies = np.array(self.episode_data['entropies'])
        ax.scatter(episodes, entropies, alpha=inner_alpha, s=point_size, color='#7B1FA2',
                  label='Policy Entropy', edgecolors='#4A148C', linewidths=0.45, rasterized=True)

        if len(entropies) > 5:
            try:
                fitted_entropies, _ = self.fit_curve(episodes, entropies, 'moving_average')
                ax.plot(episodes, fitted_entropies, '#E91E63', linewidth=2.5, label='Moving Average', alpha=0.9)
            except Exception as e:
                print(f"熵值拟合失败: {e}")

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Policy Entropy', fontsize=11)
        ax.set_title('Policy Entropy Change', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 5. 任务完成率 - 优化样式
        ax = axes[1, 1]
        completion_rates = np.array(self.episode_data['completion_rates'])
        ax.scatter(episodes, completion_rates, alpha=inner_alpha, s=point_size, color='#1565C0',
                  label='Completion Rate', edgecolors='#0D47A1', linewidths=0.45, rasterized=True)

        if len(completion_rates) > 5:
            try:
                fitted_rates, _ = self.fit_curve(episodes, completion_rates, 'moving_average')
                ax.plot(episodes, fitted_rates, '#03A9F4', linewidth=2.5, label='Moving Average', alpha=0.9)
            except Exception as e:
                print(f"完成率拟合失败: {e}")

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Task Completion Rate', fontsize=11)
        ax.set_title('Task Completion Rate Progress', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 6. AGV载重利用率 - 优化样式
        ax = axes[1, 2]
        load_utils = np.array(self.episode_data['load_utilizations'])
        if len(load_utils) > 0 and np.any(load_utils > 0):
            ax.scatter(episodes, load_utils, alpha=inner_alpha, s=point_size, color='#00838F',
                      label='Load Utilization', edgecolors='#004D40', linewidths=0.45, rasterized=True)

            if len(load_utils) > 5:
                try:
                    fitted_utils, _ = self.fit_curve(episodes, load_utils, 'moving_average')
                    ax.plot(episodes, fitted_utils, '#26C6DA', linewidth=2.5, label='Moving Average', alpha=0.9)
                except Exception as e:
                    print(f"载重利用率拟合失败: {e}")
        else:
            ax.text(0.5, 0.5, 'Load Utilization Data\nNot Yet Collected',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Load Utilization Rate', fontsize=11)
        ax.set_title('AGV Load Utilization', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 7. 路径长度 - 优化样式
        ax = axes[2, 0]
        path_lengths = np.array(self.episode_data['path_lengths'])
        if len(path_lengths) > 0 and np.any(path_lengths > 0):
            ax.scatter(episodes, path_lengths, alpha=inner_alpha, s=point_size, color='#5D4037',
                      label='Avg Path Length', edgecolors='#3E2723', linewidths=0.45, rasterized=True)

            if len(path_lengths) > 5:
                try:
                    fitted_paths, _ = self.fit_curve(episodes, path_lengths, 'moving_average')
                    ax.plot(episodes, fitted_paths, '#8D6E63', linewidth=2.5, label='Moving Average', alpha=0.9)
                except Exception as e:
                    print(f"路径长度拟合失败: {e}")
        else:
            ax.text(0.5, 0.5, 'Path Length Data\nNot Yet Collected',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Average Path Length', fontsize=11)
        ax.set_title('AGV Average Path Length', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 8. 碰撞次数 - 优化样式
        ax = axes[2, 1]
        collisions = np.array(self.episode_data['collision_counts'])
        ax.scatter(episodes, collisions, alpha=inner_alpha, s=point_size, color='#C62828',
                  label='Collision Count', edgecolors='#B71C1C', linewidths=0.45, rasterized=True)

        if len(collisions) > 5:
            try:
                fitted_collisions, _ = self.fit_curve(episodes, collisions, 'moving_average')
                ax.plot(episodes, fitted_collisions, '#F44336', linewidth=2.5, label='Moving Average', alpha=0.9)
            except Exception as e:
                print(f"碰撞次数拟合失败: {e}")

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Collision Count', fontsize=11)
        ax.set_title('Collisions per Episode', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 9. 综合性能指标 - 优化样式
        ax = axes[2, 2]
        if len(completion_rates) > 0 and len(collisions) > 0:
            # 计算综合性能分数 (完成率高，碰撞少)
            max_collisions = max(collisions) if max(collisions) > 0 else 1
            normalized_collisions = collisions / max_collisions
            performance_score = completion_rates - 0.3 * normalized_collisions

            ax.scatter(episodes, performance_score, alpha=inner_alpha, s=point_size, color='#F57C00',
                      label='Performance Score', edgecolors='#E65100', linewidths=0.45, rasterized=True)

            if len(performance_score) > 5:
                try:
                    fitted_perf, _ = self.fit_curve(episodes, performance_score, 'moving_average')
                    ax.plot(episodes, fitted_perf, '#FFC107', linewidth=2.5, label='Moving Average', alpha=0.9)
                except Exception as e:
                    print(f"综合性能拟合失败: {e}")
        else:
            ax.text(0.5, 0.5, 'Performance Data\nCalculating...',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)

        ax.set_xlabel('Episode Number', fontsize=11)
        ax.set_ylabel('Performance Score', fontsize=11)
        ax.set_title('Overall Performance Assessment', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 优化布局 - 适应大数据量显示
        plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
        
        # 保存图表 - 只保存一张PNG图片
        if save_path:
            save_file = save_path
        else:
            save_file = self.single_plot_file

        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"训练图表已保存到: {save_file}")
        
        return fig

    def start_realtime_visualization(self, update_frequency: int = 10):
        """启动实时可视化 - 每10个episode更新一次"""
        if self.enable_realtime and not self.is_running:
            self.is_running = True
            self.update_frequency = update_frequency
            self.update_thread = threading.Thread(target=self._realtime_update_loop, daemon=True)
            self.update_thread.start()
            print(f"实时可视化已启动 - 每{update_frequency}个episode更新一次")

    def stop_realtime_visualization(self):
        """停止实时可视化"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)
        print("实时可视化已停止")

    def _realtime_update_loop(self):
        """实时更新循环 - 修复滞后问题"""
        last_update_episode = 0

        while self.is_running:
            try:
                current_episodes = len(self.episode_data['episodes'])

                # 检查是否需要更新 - 每100个episode更新一次，与训练同步
                episodes_since_last_update = current_episodes - last_update_episode

                if episodes_since_last_update >= self.update_frequency and current_episodes >= self.update_frequency:
                    # 创建并保存图表到固定文件
                    fig = self.create_training_plots()

                    # 关闭图形以释放内存
                    plt.close(fig)

                    last_update_episode = current_episodes
                    print(f"📊 训练图表已更新 (回合: {current_episodes}, 距上次更新: {episodes_since_last_update}) - 保存到: {self.single_plot_file}")

                # 减少等待时间，提高响应速度
                time.sleep(1)

            except Exception as e:
                print(f"实时可视化更新错误: {e}")
                time.sleep(5)

    def save_data_to_json(self, filepath: str):
        """保存数据到JSON文件"""
        data = {
            'episode_data': self.episode_data,
            'step_data': self.step_data,
            'config': self.config,
            'save_time': time.time()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_data_from_json(self, filepath: str):
        """从JSON文件加载数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.episode_data = data.get('episode_data', self.episode_data)
        self.step_data = data.get('step_data', self.step_data)

        print(f"已加载 {len(self.episode_data['episodes'])} 个回合的数据")

    def create_summary_report(self, save_path: str):
        """创建训练总结报告"""
        if len(self.episode_data['episodes']) < 5:
            print("数据不足，无法生成总结报告")
            return

        # 计算统计信息
        episodes = np.array(self.episode_data['episodes'])
        rewards = np.array(self.episode_data['rewards'])
        completion_rates = np.array(self.episode_data['completion_rates'])

        # 创建报告
        report = {
            'training_summary': {
                'total_episodes': len(episodes),
                'training_duration': self.episode_data['timestamps'][-1] - self.episode_data['timestamps'][0],
                'final_performance': {
                    'avg_reward_last_100': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                    'avg_completion_rate_last_100': np.mean(completion_rates[-100:]) if len(completion_rates) >= 100 else np.mean(completion_rates),
                    'best_reward': np.max(rewards),
                    'best_completion_rate': np.max(completion_rates)
                },
                'learning_progress': {
                    'reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
                    'completion_rate_improvement': completion_rates[-1] - completion_rates[0] if len(completion_rates) > 1 else 0
                }
            },
            'detailed_metrics': {
                'rewards': {
                    'mean': float(np.mean(rewards)),
                    'std': float(np.std(rewards)),
                    'min': float(np.min(rewards)),
                    'max': float(np.max(rewards))
                },
                'completion_rates': {
                    'mean': float(np.mean(completion_rates)),
                    'std': float(np.std(completion_rates)),
                    'min': float(np.min(completion_rates)),
                    'max': float(np.max(completion_rates))
                }
            }
        }

        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"训练总结报告已保存到: {save_path}")
        return report


class EnhancedMetricsCollector:
    """增强的指标收集器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_metrics = []
        self.step_metrics = []

    def collect_episode_metrics(self, env, episode: int, episode_reward: float,
                              episode_length: int, info: Dict[str, Any]) -> Dict[str, float]:
        """收集回合级指标"""
        # 基础指标
        metrics = {
            'episode': episode,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'completion_rate': info.get('completion_rate', 0.0)
        }

        # AGV性能指标
        agv_states = info.get('agv_states', [])
        if agv_states:
            # 计算载重利用率
            total_load = sum(state[2] for state in agv_states)  # state[2] 是当前载重
            max_total_load = len(agv_states) * env.max_load
            load_utilization = total_load / max_total_load if max_total_load > 0 else 0
            metrics['load_utilization'] = load_utilization

            # 计算平均路径长度（简化计算）
            avg_path_length = episode_length / len(agv_states) if len(agv_states) > 0 else 0
            metrics['path_length'] = avg_path_length
        else:
            metrics['load_utilization'] = 0.0
            metrics['path_length'] = 0.0

        # 环境统计指标
        episode_stats = info.get('episode_stats', {})
        metrics['collision_count'] = episode_stats.get('collisions', 0)
        metrics['deadlock_count'] = episode_stats.get('deadlocks', 0)

        # 效率指标
        if metrics['completion_rate'] > 0:
            metrics['efficiency_score'] = metrics['completion_rate'] / (metrics['episode_length'] / 100)
        else:
            metrics['efficiency_score'] = 0.0

        self.episode_metrics.append(metrics)
        return metrics

    def collect_step_metrics(self, step: int, actor_loss: float, critic_loss: float,
                           entropy: float, learning_rate: float = 0.0) -> Dict[str, float]:
        """收集步级指标"""
        metrics = {
            'step': step,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'learning_rate': learning_rate,
            'timestamp': time.time()
        }

        self.step_metrics.append(metrics)
        return metrics

    def get_recent_metrics(self, window: int = 100) -> Dict[str, float]:
        """获取最近的平均指标"""
        if len(self.episode_metrics) < window:
            recent_episodes = self.episode_metrics
        else:
            recent_episodes = self.episode_metrics[-window:]

        if not recent_episodes:
            return {}

        # 计算平均值
        avg_metrics = {}
        for key in recent_episodes[0].keys():
            if key != 'episode':
                values = [ep[key] for ep in recent_episodes if key in ep]
                avg_metrics[f'avg_{key}'] = np.mean(values) if values else 0.0

        return avg_metrics
