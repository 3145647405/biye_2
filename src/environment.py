"""
多AGV调度环境实现
基于Gymnasium API的多智能体仓储环境
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import heapq

# matplotlib导入移到render方法中，避免初始化时的问题


@dataclass
class Task:
    """任务数据类"""
    id: int
    position: Tuple[int, int]  # (x, y) 坐标
    weight: float
    urgency: float
    assigned: bool = False
    completed: bool = False


@dataclass
class AGV:
    """AGV数据类"""
    id: int
    position: Tuple[float, float]  # (x, y) 坐标
    target_position: Optional[Tuple[int, int]] = None
    path: List[Tuple[int, int]] = None
    current_load: float = 0.0
    max_load: float = 25.0
    missions: List[Task] = None
    status: str = "idle"  # idle, moving, picking, delivering
    last_action_time: int = 0
    
    def __post_init__(self):
        if self.missions is None:
            self.missions = []
        if self.path is None:
            self.path = []


class SpaceTimeReservationTable:
    """时空预留表，用于多AGV路径冲突避免"""

    def __init__(self, map_width: int, map_height: int, time_horizon: int = 100):
        self.map_width = map_width
        self.map_height = map_height
        self.time_horizon = time_horizon
        # 预留表：{(x, y, t): agv_id}
        self.reservations = {}
        # 安全缓冲区大小
        self.safety_buffer = 1

    def clear_reservations(self, agv_id: int):
        """清除指定AGV的所有预留"""
        keys_to_remove = [key for key, value in self.reservations.items() if value == agv_id]
        for key in keys_to_remove:
            del self.reservations[key]

    def reserve_path(self, agv_id: int, path_with_time: List[Tuple[int, int, int]], safe_positions: set = None):
        """为AGV预留路径上的时空点"""
        self.clear_reservations(agv_id)

        if safe_positions is None:
            safe_positions = set()

        for x, y, t in path_with_time:
            if 0 <= x < self.map_width and 0 <= y < self.map_height and t < self.time_horizon:
                # 如果是安全位置，跳过预留
                if (x, y) in safe_positions:
                    continue

                # 预留主要位置
                self.reservations[(x, y, t)] = agv_id

                # 预留安全缓冲区（相邻时间步）
                for dt in [-1, 1]:
                    if t + dt >= 0 and t + dt < self.time_horizon:
                        self.reservations[(x, y, t + dt)] = agv_id

    def check_conflicts(self, path_with_time: List[Tuple[int, int, int]], agv_id: int) -> List[Tuple[int, int, int]]:
        """检查路径是否与已预留的时空点冲突"""
        conflicts = []

        for x, y, t in path_with_time:
            if (x, y, t) in self.reservations and self.reservations[(x, y, t)] != agv_id:
                conflicts.append((x, y, t))

        return conflicts

    def is_position_free(self, x: int, y: int, t: int, agv_id: int, safe_positions: set = None) -> bool:
        """检查指定时空点是否可用"""
        # 如果提供了安全位置集合，检查当前位置是否为安全位置
        if safe_positions and (x, y) in safe_positions:
            return True  # 安全位置允许多个AGV共存

        return (x, y, t) not in self.reservations or self.reservations[(x, y, t)] == agv_id

    def cleanup_old_reservations(self, current_time: int):
        """清理过期的预留信息"""
        keys_to_remove = [key for key in self.reservations.keys() if key[2] < current_time - 10]
        for key in keys_to_remove:
            del self.reservations[key]


class AStarPlanner:
    """A*路径规划器"""
    
    def __init__(self, grid_map: np.ndarray):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """启发式函数（曼哈顿距离）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取邻居节点"""
        x, y = pos
        neighbors = []
        
        # 8方向移动
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                self.grid_map[ny, nx] == 0):  # 0表示可通行
                neighbors.append((nx, ny))
        
        return neighbors
    
    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A*路径规划"""
        if start == goal:
            return [start]
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # 无路径


class MultiAGVPathPlanner:
    """支持时空预留的多AGV协调路径规划器"""

    def __init__(self, grid_map: np.ndarray, reservation_table: SpaceTimeReservationTable):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        self.reservation_table = reservation_table
        # 安全位置集合，将在环境中设置
        self.safe_positions = set()

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """启发式函数（曼哈顿距离）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取邻居节点"""
        x, y = pos
        neighbors = []

        # 8方向移动 + 等待动作
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and
                self.grid_map[ny, nx] == 0):  # 0表示可通行
                neighbors.append((nx, ny))

        return neighbors

    def plan_path_with_time(self, start: Tuple[int, int], goal: Tuple[int, int],
                           start_time: int, agv_id: int, max_time: int = 200) -> List[Tuple[int, int, int]]:
        """时空A*路径规划"""
        if start == goal:
            return [(start[0], start[1], start_time)]

        # 使用时空A*：状态为(x, y, t)
        open_set = [(0, start[0], start[1], start_time)]
        came_from = {}
        g_score = {(start[0], start[1], start_time): 0}
        f_score = {(start[0], start[1], start_time): self.heuristic(start, goal)}

        while open_set:
            _, current_x, current_y, current_t = heapq.heappop(open_set)
            current_state = (current_x, current_y, current_t)

            if (current_x, current_y) == goal:
                # 重构路径
                path = []
                while current_state in came_from:
                    path.append(current_state)
                    current_state = came_from[current_state]
                path.append((start[0], start[1], start_time))
                return path[::-1]

            if current_t >= max_time:
                continue

            for next_x, next_y in self.get_neighbors((current_x, current_y)):
                next_t = current_t + 1
                next_state = (next_x, next_y, next_t)

                # 检查时空冲突，考虑安全位置
                if not self.reservation_table.is_position_free(next_x, next_y, next_t, agv_id, self.safe_positions):
                    continue

                tentative_g_score = g_score[current_state] + 1

                if next_state not in g_score or tentative_g_score < g_score[next_state]:
                    came_from[next_state] = current_state
                    g_score[next_state] = tentative_g_score
                    f_score[next_state] = tentative_g_score + self.heuristic((next_x, next_y), goal)
                    heapq.heappush(open_set, (f_score[next_state], next_x, next_y, next_t))

        return []  # 无路径

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  start_time: int, agv_id: int) -> List[Tuple[int, int]]:
        """规划路径并返回空间坐标序列"""
        path_with_time = self.plan_path_with_time(start, goal, start_time, agv_id)
        if path_with_time:
            # 预留路径，传递安全位置信息
            self.reservation_table.reserve_path(agv_id, path_with_time, self.safe_positions)
            # 返回空间路径
            return [(x, y) for x, y, t in path_with_time]
        return []


class AGVEnv(gym.Env):
    """多AGV调度环境"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        env_config = config['environment']
        
        # 环境参数
        self.map_width = env_config['map_width']
        self.map_height = env_config['map_height']
        self.num_agvs = env_config['num_agvs']
        self.num_tasks = env_config['num_tasks']
        self.max_load = env_config['max_load']
        self.max_steps = env_config['max_steps']
        
        # 货架配置
        self.shelf_width = env_config['shelf_width']
        self.shelf_height = env_config['shelf_height']
        self.shelf_spacing = env_config['shelf_spacing']
        
        # 观测配置
        self.max_nearby_tasks = env_config['max_nearby_tasks']
        self.max_nearby_agvs = env_config['max_nearby_agvs']
        self.observation_range = env_config['observation_range']
        
        # 奖励配置
        self.rewards = config['rewards']
        
        # 设置起点和终点
        self.start_position = (1, self.map_height - 2)  # 左下角
        self.end_position = (self.map_width - 2, 1)     # 右上角

        # 初始化地图
        self._create_map()

        # 初始化时空预留表和路径规划器
        self.reservation_table = SpaceTimeReservationTable(self.map_width, self.map_height)
        self.planner = AStarPlanner(self.grid_map)  # 保留原有规划器作为备用
        self.multi_agv_planner = MultiAGVPathPlanner(self.grid_map, self.reservation_table)

        # 设置安全位置（起点和终点）
        self.safe_positions = {
            (int(self.start_position[0]), int(self.start_position[1])),  # 起点
            (int(self.end_position[0]), int(self.end_position[1]))       # 终点
        }
        self.multi_agv_planner.safe_positions = self.safe_positions
        
        # 定义动作和观测空间
        self._setup_spaces()
        
        # 环境状态
        self.agvs: List[AGV] = []
        self.tasks: List[Task] = []
        self.current_step = 0
        self.episode_stats = {}
    
    def _create_map(self):
        """创建仓库地图"""
        # 初始化为全部可通行
        self.grid_map = np.zeros((self.map_height, self.map_width), dtype=int)
        
        # 添加边界墙
        self.grid_map[0, :] = 1  # 上边界
        self.grid_map[-1, :] = 1  # 下边界
        self.grid_map[:, 0] = 1  # 左边界
        self.grid_map[:, -1] = 1  # 右边界
        
        # 添加货架（简化版本）
        shelf_positions = []
        for y in range(2, self.map_height - 2, self.shelf_height + self.shelf_spacing):
            for x in range(3, self.map_width - 3, self.shelf_width + self.shelf_spacing):
                # 放置货架
                for dy in range(self.shelf_height):
                    for dx in range(self.shelf_width):
                        if (y + dy < self.map_height - 1 and 
                            x + dx < self.map_width - 1):
                            self.grid_map[y + dy, x + dx] = 1
                            shelf_positions.append((x + dx, y + dy))
        
        self.shelf_positions = shelf_positions
        
        # 确保起点和终点可通行
        self.grid_map[self.map_height - 2, 1] = 0  # 起点
        self.grid_map[1, self.map_width - 2] = 0   # 终点
    
    def _setup_spaces(self):
        """设置动作和观测空间"""
        # 动作空间：选择任务ID（0到max_nearby_tasks-1）或无操作(max_nearby_tasks)
        self.action_space = spaces.Discrete(self.max_nearby_tasks + 1)
        
        # 观测空间：字典格式
        self.observation_space = spaces.Dict({
            'agv_own_state': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(5,), dtype=np.float32
            ),  # [x, y, vx, vy, load_ratio]
            
            'nearby_tasks_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_nearby_tasks, 4), dtype=np.float32
            ),  # [x, y, weight, urgency] for each task
            
            'nearby_agvs_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_nearby_agvs, 2), dtype=np.float32
            )   # [x, y] for each nearby AGV
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_stats = {
            'total_reward': 0,
            'tasks_completed': 0,
            'collisions': 0,
            'deadlocks': 0
        }

        # 清理时空预留表
        self.reservation_table.reservations.clear()
        
        # 初始化AGV
        self._initialize_agvs()
        
        # 初始化任务
        self._initialize_tasks()
        
        # 获取初始观测
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def _initialize_agvs(self):
        """初始化AGV"""
        self.agvs = []
        for i in range(self.num_agvs):
            agv = AGV(
                id=i,
                position=(float(self.start_position[0]), float(self.start_position[1])),
                max_load=self.max_load
            )
            self.agvs.append(agv)
    
    def _initialize_tasks(self):
        """初始化任务"""
        self.tasks = []
        task_weights = self.config['environment']['task_weights']
        urgency_levels = self.config['environment']['urgency_levels']
        
        # 获取可放置任务的位置（货架附近）
        valid_positions = []
        for shelf_pos in self.shelf_positions:
            # 在货架周围寻找可通行位置
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                x, y = shelf_pos[0] + dx, shelf_pos[1] + dy
                if (0 < x < self.map_width - 1 and 
                    0 < y < self.map_height - 1 and
                    self.grid_map[y, x] == 0):
                    valid_positions.append((x, y))
        
        # 随机选择任务位置
        if len(valid_positions) < self.num_tasks:
            # 如果可用位置不足，使用所有可通行位置
            valid_positions = []
            for y in range(1, self.map_height - 1):
                for x in range(1, self.map_width - 1):
                    if self.grid_map[y, x] == 0:
                        valid_positions.append((x, y))
        
        selected_positions = np.random.choice(
            len(valid_positions), 
            size=min(self.num_tasks, len(valid_positions)), 
            replace=False
        )
        
        for i, pos_idx in enumerate(selected_positions):
            position = valid_positions[pos_idx]
            weight = np.random.choice(task_weights)
            urgency = np.random.choice(urgency_levels)
            
            task = Task(
                id=i,
                position=position,
                weight=weight,
                urgency=urgency
            )
            self.tasks.append(task)
    
    def step(self, actions: Dict[int, int]):
        """环境步进"""
        self.current_step += 1

        # 清理过期的时空预留
        self.reservation_table.cleanup_old_reservations(self.current_step)

        # 处理每个AGV的动作
        rewards = {}
        for agv_id, action in actions.items():
            reward = self._process_agv_action(agv_id, action)
            rewards[agv_id] = reward
            # 更新AGV的最后动作时间
            self.agvs[agv_id].last_action_time = self.current_step

        # 更新AGV位置并获取移动效率奖励
        movement_rewards = self._update_agv_positions()
        for agv_id, movement_reward in movement_rewards.items():
            rewards[agv_id] = rewards.get(agv_id, 0) + movement_reward

        # 检测碰撞并分配奖励
        collision_rewards = self._check_collisions()
        for agv_id, collision_reward in collision_rewards.items():
            rewards[agv_id] = rewards.get(agv_id, 0) + collision_reward

        # 检测任务完成并分配奖励
        completion_rewards = self._check_task_completion()
        for agv_id, completion_reward in completion_rewards.items():
            rewards[agv_id] = rewards.get(agv_id, 0) + completion_reward

        # 检测死锁并分配奖励
        deadlock_rewards = self._check_deadlocks()
        for agv_id, deadlock_reward in deadlock_rewards.items():
            rewards[agv_id] = rewards.get(agv_id, 0) + deadlock_reward

        # 计算总奖励
        for agv_id, reward in rewards.items():
            self.episode_stats['total_reward'] += reward

        # 获取新观测
        observations = self._get_observations()

        # 检查是否结束
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        info = self._get_info()

        return observations, rewards, terminated, truncated, info

    def _process_agv_action(self, agv_id: int, action: int) -> float:
        """处理单个AGV的动作"""
        agv = self.agvs[agv_id]
        reward = 0.0

        # 获取附近可用任务
        nearby_tasks = self._get_nearby_tasks(agv)

        if action < len(nearby_tasks):
            # 选择任务
            selected_task = nearby_tasks[action]
            if not selected_task.assigned and agv.current_load + selected_task.weight <= agv.max_load:
                # 分配任务
                selected_task.assigned = True
                agv.missions.append(selected_task)
                agv.current_load += selected_task.weight
                agv.target_position = selected_task.position
                agv.status = "moving"

                # 使用多AGV协调路径规划
                current_pos = (int(agv.position[0]), int(agv.position[1]))
                agv.path = self.multi_agv_planner.plan_path(
                    current_pos, selected_task.position, self.current_step, agv.id
                )

                # 任务选择奖励（暂时不给，等到实际拾取时给予）
                print(f"任务分配: AGV{agv.id} 选择任务{selected_task.id}")
            else:
                # 无效动作惩罚：选择了不可用的任务
                reward += self.rewards.get('inefficient_action', -0.5)

        elif action == self.max_nearby_tasks:
            # 无操作或前往终点
            if agv.missions and agv.status != "delivering":
                agv.target_position = self.end_position
                agv.status = "delivering"
                current_pos = (int(agv.position[0]), int(agv.position[1]))
                agv.path = self.multi_agv_planner.plan_path(
                    current_pos, self.end_position, self.current_step, agv.id
                )
                print(f"前往终点: AGV{agv.id} 开始前往终点卸货")

        # 移动效率奖励将在_update_agv_positions中计算

        # 基础步进惩罚（已大幅降低）
        reward += self.rewards['step_penalty']

        return reward

    def _update_agv_positions(self):
        """更新AGV位置并计算移动效率奖励"""
        movement_rewards = {}

        for agv in self.agvs:
            if agv.path and len(agv.path) > 1:
                # 记录移动前位置
                old_position = agv.position

                # 沿路径移动
                next_pos = agv.path[1]
                agv.position = (float(next_pos[0]), float(next_pos[1]))
                agv.path.pop(0)

                # 计算移动效率奖励
                if agv.target_position:
                    old_distance = np.sqrt((old_position[0] - agv.target_position[0])**2 +
                                         (old_position[1] - agv.target_position[1])**2)
                    new_distance = np.sqrt((agv.position[0] - agv.target_position[0])**2 +
                                         (agv.position[1] - agv.target_position[1])**2)

                    # 如果距离目标更近，给予小额奖励
                    if new_distance < old_distance:
                        efficient_reward = self.rewards.get('efficient_move', 0.02)
                        movement_rewards[agv.id] = movement_rewards.get(agv.id, 0) + efficient_reward

                # 检查是否到达目标
                if len(agv.path) == 1:  # 只剩目标位置
                    agv.path = []
                    if agv.status == "moving":
                        # 到达任务位置，开始拾取
                        agv.status = "picking"
                    elif agv.status == "delivering":
                        # 到达终点，完成任务
                        agv.status = "idle"

        return movement_rewards

    def _check_collisions(self):
        """检测AGV碰撞并返回奖励"""
        collision_rewards = {}
        positions = {}

        # 定义允许多AGV共存的位置（起点和终点）
        safe_positions = {
            (int(self.start_position[0]), int(self.start_position[1])),  # 起点
            (int(self.end_position[0]), int(self.end_position[1]))       # 终点
        }

        for agv in self.agvs:
            pos_key = (int(agv.position[0]), int(agv.position[1]))

            # 检查是否在安全位置（起点或终点）
            if pos_key in safe_positions:
                # 在起点或终点，允许多个AGV共存，不算碰撞
                continue

            if pos_key in positions:
                # 在非安全位置发生碰撞
                self.episode_stats['collisions'] += 1
                collision_agv = positions[pos_key]

                # 给碰撞的AGV负奖励
                collision_rewards[agv.id] = collision_rewards.get(agv.id, 0) + self.rewards['collision']
                collision_rewards[collision_agv.id] = collision_rewards.get(collision_agv.id, 0) + self.rewards['collision']

                print(f"碰撞检测: AGV{agv.id} 和 AGV{collision_agv.id} 在位置 {pos_key} 发生碰撞")
            else:
                positions[pos_key] = agv

        return collision_rewards

    def _check_task_completion(self):
        """检测任务完成并返回奖励"""
        completion_rewards = {}

        for agv in self.agvs:
            if agv.status == "picking":
                # 检查是否在任务位置
                for task in agv.missions:
                    if (not task.completed and
                        abs(agv.position[0] - task.position[0]) < 0.5 and
                        abs(agv.position[1] - task.position[1]) < 0.5):
                        task.completed = True
                        agv.status = "moving"  # 继续移动到下一个任务或终点

                        # 给予任务拾取奖励
                        pickup_reward = self.rewards['task_pickup'] * (task.urgency / 9.0)
                        completion_rewards[agv.id] = completion_rewards.get(agv.id, 0) + pickup_reward
                        print(f"任务拾取: AGV{agv.id} 拾取任务{task.id}，获得奖励 {pickup_reward:.2f}")

            elif agv.status == "idle":
                # 检查是否在终点附近
                if (abs(agv.position[0] - self.end_position[0]) < 0.5 and
                    abs(agv.position[1] - self.end_position[1]) < 0.5):
                    # 在终点完成所有任务
                    completed_tasks = [t for t in agv.missions if t.completed]
                    if completed_tasks:
                        self.episode_stats['tasks_completed'] += len(completed_tasks)

                        # 给予任务完成奖励
                        for task in completed_tasks:
                            mission_reward = self.rewards['mission_complete']
                            completion_rewards[agv.id] = completion_rewards.get(agv.id, 0) + mission_reward
                            print(f"任务完成: AGV{agv.id} 完成任务{task.id}，获得奖励 {mission_reward}")

                        # 清空已完成的任务
                        agv.missions = [t for t in agv.missions if not t.completed]
                        agv.current_load = sum(t.weight for t in agv.missions)

        return completion_rewards

    def _check_deadlocks(self):
        """检测死锁并返回奖励"""
        deadlock_rewards = {}

        for agv in self.agvs:
            # 检测位置死锁：AGV在同一位置停留过久
            if (self.current_step - agv.last_action_time > 30 and
                agv.status != "idle"):
                self.episode_stats['deadlocks'] += 1

                # 给予死锁惩罚
                deadlock_rewards[agv.id] = deadlock_rewards.get(agv.id, 0) + self.rewards['deadlock']
                print(f"死锁检测: AGV{agv.id} 在位置 {agv.position} 停留过久，获得惩罚 {self.rewards['deadlock']}")

                # 重置最后动作时间，避免重复惩罚
                agv.last_action_time = self.current_step

        return deadlock_rewards

    def _get_nearby_tasks(self, agv: AGV) -> List[Task]:
        """获取AGV附近的可用任务"""
        available_tasks = [t for t in self.tasks if not t.assigned and not t.completed]

        # 计算距离并排序
        task_distances = []
        for task in available_tasks:
            distance = np.sqrt(
                (agv.position[0] - task.position[0])**2 +
                (agv.position[1] - task.position[1])**2
            )
            if distance <= self.observation_range:
                task_distances.append((distance, task))

        # 按距离排序，返回最近的任务
        task_distances.sort(key=lambda x: x[0])
        nearby_tasks = [task for _, task in task_distances[:self.max_nearby_tasks]]

        return nearby_tasks

    def _get_nearby_agvs(self, agv: AGV) -> List[AGV]:
        """获取AGV附近的其他AGV"""
        nearby_agvs = []
        for other_agv in self.agvs:
            if other_agv.id != agv.id:
                distance = np.sqrt(
                    (agv.position[0] - other_agv.position[0])**2 +
                    (agv.position[1] - other_agv.position[1])**2
                )
                if distance <= self.observation_range:
                    nearby_agvs.append(other_agv)

        # 按距离排序
        nearby_agvs.sort(key=lambda x: np.sqrt(
            (agv.position[0] - x.position[0])**2 +
            (agv.position[1] - x.position[1])**2
        ))

        return nearby_agvs[:self.max_nearby_agvs]

    def _get_observations(self) -> Dict[int, Dict[str, np.ndarray]]:
        """获取所有AGV的观测"""
        observations = {}

        for agv in self.agvs:
            # AGV自身状态
            load_ratio = agv.current_load / agv.max_load
            velocity = (0.0, 0.0)  # 简化处理，实际可以计算速度
            own_state = np.array([
                agv.position[0], agv.position[1],
                velocity[0], velocity[1], load_ratio
            ], dtype=np.float32)

            # 附近任务状态
            nearby_tasks = self._get_nearby_tasks(agv)
            tasks_state = np.zeros((self.max_nearby_tasks, 4), dtype=np.float32)
            for i, task in enumerate(nearby_tasks):
                if i < self.max_nearby_tasks:
                    tasks_state[i] = [
                        task.position[0], task.position[1],
                        task.weight, task.urgency / 9.0  # 标准化紧急程度
                    ]

            # 附近AGV状态
            nearby_agvs = self._get_nearby_agvs(agv)
            agvs_state = np.zeros((self.max_nearby_agvs, 2), dtype=np.float32)
            for i, other_agv in enumerate(nearby_agvs):
                if i < self.max_nearby_agvs:
                    agvs_state[i] = [other_agv.position[0], other_agv.position[1]]

            observations[agv.id] = {
                'agv_own_state': own_state,
                'nearby_tasks_state': tasks_state,
                'nearby_agvs_state': agvs_state
            }

        return observations

    def _is_terminated(self) -> bool:
        """检查是否终止"""
        # 所有任务完成
        return all(task.completed for task in self.tasks)

    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for task in self.tasks if task.completed)
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0

        return {
            'episode_stats': self.episode_stats.copy(),
            'completion_rate': completion_rate,
            'current_step': self.current_step,
            'agv_states': [(agv.position, agv.status, agv.current_load) for agv in self.agvs]
        }

    def render(self, mode='human'):
        """渲染环境"""
        # 延迟导入matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("警告: matplotlib未安装，跳过渲染")
            return None

        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            plt.ion()

        self.ax.clear()

        # 绘制地图
        self.ax.imshow(self.grid_map, cmap='gray_r', alpha=0.3)

        # 绘制货架
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self.grid_map[y, x] == 1:
                    rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8,
                                           linewidth=1, edgecolor='black',
                                           facecolor='brown', alpha=0.7)
                    self.ax.add_patch(rect)

        # 绘制起点和终点
        start_circle = patches.Circle(self.start_position, 0.3,
                                    color='green', alpha=0.8, label='起点')
        end_circle = patches.Circle(self.end_position, 0.3,
                                  color='red', alpha=0.8, label='终点')
        self.ax.add_patch(start_circle)
        self.ax.add_patch(end_circle)

        # 绘制任务
        for task in self.tasks:
            if not task.completed:
                color = 'orange' if task.urgency > 5 else 'yellow'
                alpha = 0.9 if task.assigned else 0.6
                task_circle = patches.Circle(task.position, 0.2,
                                           color=color, alpha=alpha)
                self.ax.add_patch(task_circle)

                # 显示任务信息
                self.ax.text(task.position[0], task.position[1] + 0.4,
                           f'T{task.id}\nW:{task.weight}\nU:{task.urgency}',
                           ha='center', va='bottom', fontsize=8)

        # 绘制AGV
        colors = ['blue', 'purple', 'cyan', 'magenta', 'lime', 'pink']
        for i, agv in enumerate(self.agvs):
            color = colors[i % len(colors)]

            # AGV主体
            agv_circle = patches.Circle(agv.position, 0.25,
                                      color=color, alpha=0.8)
            self.ax.add_patch(agv_circle)

            # AGV信息
            info_text = f'AGV{agv.id}\n{agv.status}\nLoad:{agv.current_load:.1f}'
            self.ax.text(agv.position[0], agv.position[1] - 0.6, info_text,
                        ha='center', va='top', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

            # 绘制路径
            if agv.path:
                path_x = [pos[0] for pos in agv.path]
                path_y = [pos[1] for pos in agv.path]
                self.ax.plot(path_x, path_y, color=color, alpha=0.5, linewidth=2)

        # 设置图形属性
        self.ax.set_xlim(-0.5, self.map_width - 0.5)
        self.ax.set_ylim(-0.5, self.map_height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # 翻转Y轴使(0,0)在左上角
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f'多AGV调度环境 - 步数: {self.current_step}')

        # 添加图例和统计信息
        legend_text = (f'已完成任务: {self.episode_stats["tasks_completed"]}\n'
                      f'碰撞次数: {self.episode_stats["collisions"]}\n'
                      f'死锁次数: {self.episode_stats["deadlocks"]}\n'
                      f'总奖励: {self.episode_stats["total_reward"]:.2f}')

        self.ax.text(0.02, 0.98, legend_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                    facecolor='white', alpha=0.8))

        if mode == 'human':
            plt.pause(0.1)
            plt.draw()
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return buf

    def close(self):
        """关闭环境"""
        if hasattr(self, 'fig'):
            try:
                import matplotlib.pyplot as plt
                plt.close(self.fig)
            except ImportError:
                pass

    def get_global_state(self) -> np.ndarray:
        """获取全局状态（用于中心化Critic）"""
        global_state = []

        # AGV状态
        for agv in self.agvs:
            agv_state = [
                agv.position[0], agv.position[1],
                agv.current_load / agv.max_load,
                len(agv.missions),
                1.0 if agv.status == "idle" else 0.0
            ]
            global_state.extend(agv_state)

        # 任务状态
        for task in self.tasks:
            task_state = [
                task.position[0], task.position[1],
                task.weight / 25.0,  # 标准化重量
                task.urgency / 9.0,  # 标准化紧急程度
                1.0 if task.assigned else 0.0,
                1.0 if task.completed else 0.0
            ]
            global_state.extend(task_state)

        return np.array(global_state, dtype=np.float32)
