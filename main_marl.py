# -*- encoding: utf-8 -*-
'''
@Time    :   2025/05/11 10:19:46
@Author  :   Li Zeng 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from copy import deepcopy

# Compatible versions (Python 3.10)
"""
torch==2.0.1
numpy==1.24.3
matplotlib==3.7.1
gym==0.26.2
"""

# Simple 2v2 adversarial environment
class AdversarialEnv:
    def __init__(self):
        self.num_agents = 4  # 2v2
        self.agent_positions = np.array([
            [0.3, 0.3],  # Team 1 - Agent 1 (初始位置向中心靠拢)
            [0.3, 0.7],  # Team 1 - Agent 2
            [0.7, 0.3],  # Team 2 - Agent 1
            [0.7, 0.7]   # Team 2 - Agent 2
        ])
        self.agent_velocities = np.zeros((4, 2))
        self.team1_goals = 0
        self.team2_goals = 0
        self.ball_pos = np.array([0.5, 0.5])
        self.ball_vel = np.zeros(2)
        self.max_vel = 0.05
        self.action_dim = 2
        self.obs_dim = 8
        self.max_steps = 200
        self.current_step = 0
        
        # 边界控制参数
        self.boundary_penalty = 0.2  # 边界惩罚系数
        self.safe_zone = 0.9  # 安全区域(0.05到0.95)
        
        # 球门参数
        self.goal_width = 0.4
        self.goal_depth = 0.05
        self.left_goal_pos = 0.0
        self.right_goal_pos = 1.0
        self.goal_y_range = (0.3, 0.7)

    def reset(self):
        # 随机初始化位置（在中心区域）
        self.agent_positions = np.array([
            [0.3 + 0.1*np.random.rand(), 0.3 + 0.4*np.random.rand()],  # Team1
            [0.3 + 0.1*np.random.rand(), 0.3 + 0.4*np.random.rand()],  # Team1
            [0.6 + 0.1*np.random.rand(), 0.3 + 0.4*np.random.rand()],  # Team2
            [0.6 + 0.1*np.random.rand(), 0.3 + 0.4*np.random.rand()]   # Team2
        ])
        self.agent_velocities = np.zeros((4, 2))
        self.ball_pos = np.array([0.5, 0.5])
        self.ball_vel = np.zeros(2)
        self.current_step = 0
        return self._get_obs()
    
    def _get_obs(self):
        obs = []
        for i in range(4):
            agent_obs = np.concatenate([
                self.agent_positions[i],
                self.agent_velocities[i],
                self.ball_pos,
                self.ball_vel
            ])
            obs.append(agent_obs)
        return obs
    
    def step(self, actions):
        actions = np.clip(actions, -1, 1) * self.max_vel
        
        # 更新智能体位置
        for i in range(4):
            self.agent_velocities[i] = actions[i]
            self.agent_positions[i] += self.agent_velocities[i]
            self.agent_positions[i] = np.clip(self.agent_positions[i], 0, 1)
        
        # 更新球物理
        for i in range(4):
            dist = np.linalg.norm(self.agent_positions[i] - self.ball_pos)
            if dist < 0.1:
                self.ball_vel += self.agent_velocities[i] * 2.0
        
        self.ball_pos += self.ball_vel
        self.ball_vel *= 0.95  # 摩擦减速

        # 边界检测
        if self.ball_pos[1] < 0 or self.ball_pos[1] > 1:
            self.ball_vel[1] *= -0.8
            
        if ((self.ball_pos[0] < 0 and not 
             (self.goal_y_range[0] < self.ball_pos[1] < self.goal_y_range[1])) or
            (self.ball_pos[0] > 1 and not 
             (self.goal_y_range[0] < self.ball_pos[1] < self.goal_y_range[1]))):
            self.ball_vel[0] *= -0.8
        
        self.ball_pos = np.clip(self.ball_pos, 0.0, 1.0)
        
        # 计算边界惩罚
        boundary_penalties = np.zeros(4)
        for i in range(4):
            if (self.agent_positions[i][0] < 1-self.safe_zone or 
                self.agent_positions[i][0] > self.safe_zone or
                self.agent_positions[i][1] < 1-self.safe_zone or 
                self.agent_positions[i][1] > self.safe_zone):
                boundary_penalties[i] = self.boundary_penalty
        
        # 得分判定
        team1_reward = 0
        team2_reward = 0
        done = False
        
        # 右球门得分
        if (self.ball_pos[0] >= self.right_goal_pos - self.goal_depth and 
            self.goal_y_range[0] < self.ball_pos[1] < self.goal_y_range[1]):
            team1_reward += 1
            self.team1_goals += 1
            done = True
        
        # 左球门得分
        if (self.ball_pos[0] <= self.left_goal_pos + self.goal_depth and 
            self.goal_y_range[0] < self.ball_pos[1] < self.goal_y_range[1]):
            team2_reward += 1
            self.team2_goals += 1
            done = True
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        # 最终奖励 = 得分奖励 - 边界惩罚
        rewards = [
            team1_reward - boundary_penalties[0],
            team1_reward - boundary_penalties[1], 
            team2_reward - boundary_penalties[2],
            team2_reward - boundary_penalties[3]
        ]
        next_obs = self._get_obs()
        
        return next_obs, rewards, done, {}

    def render(self):
        plt.clf()
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.1, 1.1)
        
        # 绘制场地边界和安全区域
        plt.plot([0,0], [0,1], 'k-')
        plt.plot([1,1], [0,1], 'k-')
        plt.plot([0,1], [0,0], 'k-')
        plt.plot([0,1], [1,1], 'k-')
        plt.plot([1-self.safe_zone]*2, [0,1], 'b--', alpha=0.3)
        plt.plot([self.safe_zone]*2, [0,1], 'b--', alpha=0.3)
        plt.plot([0,1], [1-self.safe_zone]*2, 'b--', alpha=0.3)
        plt.plot([0,1], [self.safe_zone]*2, 'b--', alpha=0.3)
        
        # 绘制球门
        plt.plot([self.left_goal_pos]*2, list(self.goal_y_range), 'k-', linewidth=4)
        plt.plot([self.right_goal_pos]*2, list(self.goal_y_range), 'k-', linewidth=4)
        
        # 绘制智能体和球
        for i, pos in enumerate(self.agent_positions):
            color = 'blue' if i < 2 else 'red'
            plt.plot(pos[0], pos[1], 'o', markersize=10, color=color)
            plt.text(pos[0], pos[1], f'A{i+1}', ha='center', va='center', color='white')
        plt.plot(self.ball_pos[0], self.ball_pos[1], 'ko', markersize=6)
        
        # 显示比分
        plt.text(0.25, 1.05, f'Team1: {self.team1_goals}', color='blue')
        plt.text(0.75, 1.05, f'Team2: {self.team2_goals}', color='red')
        
        plt.title('2v2 Adversarial Environment (With Boundary Control)')
        plt.pause(0.01)

# Replay Buffer with corrected sampling
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", 
            field_names=["obs", "actions", "rewards", "next_obs", "dones"])
    
    def add(self, obs, actions, rewards, next_obs, dones):
        e = self.experience(obs, actions, rewards, next_obs, dones)
        self.buffer.append(e)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        obs = torch.FloatTensor(np.stack([e.obs for e in batch]))
        actions = torch.FloatTensor(np.stack([e.actions for e in batch]))
        rewards = torch.FloatTensor(np.stack([e.rewards for e in batch]))
        next_obs = torch.FloatTensor(np.stack([e.next_obs for e in batch]))
        dones = torch.FloatTensor(np.stack([e.dones for e in batch]))
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Critic Network
class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# MADDPG with corrected update method
class MADDPG:
    def __init__(self, env, gamma=0.95, tau=0.01, lr_actor=1e-3, lr_critic=1e-3):
        self.env = env
        self.num_agents = env.num_agents
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        
        # 初始化网络
        self.actors = [Actor(self.obs_dim, self.action_dim) for _ in range(self.num_agents)]
        self.critics = [Critic(self.obs_dim*self.num_agents, self.action_dim*self.num_agents) 
                       for _ in range(self.num_agents)]
        self.target_actors = [deepcopy(actor) for actor in self.actors]
        self.target_critics = [deepcopy(critic) for critic in self.critics]
        
        # 优化器
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr_actor) 
                               for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=lr_critic) 
                                for critic in self.critics]
        
        # 训练参数
        self.gamma = gamma
        self.tau = tau
        self.noise = OUNoise(self.action_dim)
        
        # 新增奖励参数
        self.ball_follow_reward = 0.01  # 追球奖励
        self.team_coop_reward = 0.02   # 团队协作奖励
    
    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        
        # 预处理输入
        obs = torch.cat([obs[:,i,:] for i in range(self.num_agents)], dim=1)
        actions = torch.cat([actions[:,i,:] for i in range(self.num_agents)], dim=1)
        next_obs = torch.cat([next_obs[:,i,:] for i in range(self.num_agents)], dim=1)
        
        for i in range(self.num_agents):
            agent_rewards = rewards[:,i]
            agent_dones = dones[:,i]
            
            # 添加追球奖励
            ball_dist = torch.norm(obs[:,i*self.obs_dim:i*self.obs_dim+2] - 
                                  obs[:,-4:-2], dim=1)
            agent_rewards += self.ball_follow_reward / (1 + ball_dist)
            
            # 添加团队协作奖励
            if i < 2:  # Team 1
                teammate_dist = torch.norm(
                    obs[:,i*self.obs_dim:i*self.obs_dim+2] - 
                    obs[:,(1-i)*self.obs_dim:(1-i)*self.obs_dim+2], dim=1)
            else:  # Team 2
                teammate_dist = torch.norm(
                    obs[:,i*self.obs_dim:i*self.obs_dim+2] - 
                    obs[:,(5-i)*self.obs_dim:(5-i)*self.obs_dim+2], dim=1)
            
            agent_rewards += self.team_coop_reward * (
                1 - torch.abs(teammate_dist - 0.4)/0.1)
            
            # Critic更新
            with torch.no_grad():
                target_actions = torch.cat([
                    self.target_actors[j](next_obs[:,j*self.obs_dim:(j+1)*self.obs_dim]) 
                    for j in range(self.num_agents)], dim=1)
                target_Q = agent_rewards + (1 - agent_dones) * self.gamma * \
                          self.target_critics[i](next_obs, target_actions).squeeze()
            
            current_Q = self.critics[i](obs, actions).squeeze()
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # Actor更新
            actor_actions = [self.actors[j](obs[:,j*self.obs_dim:(j+1)*self.obs_dim]) 
                           if j == i else self.actors[j](obs[:,j*self.obs_dim:(j+1)*self.obs_dim]).detach()
                           for j in range(self.num_agents)]
            actor_actions = torch.cat(actor_actions, dim=1)
            actor_loss = -self.critics[i](obs, actor_actions).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            # 更新目标网络
            for target_param, param in zip(self.target_actors[i].parameters(), 
                                         self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critics[i].parameters(), 
                                         self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def act(self, obs, noise=True, exploration_rate=0.0):
        actions = []
        for i in range(self.num_agents):
            if random.random() < exploration_rate:
                # 探索阶段：限制动作范围
                action = np.clip(np.random.rand(2) * 0.6 - 0.3, -0.5, 0.5)
            else:
                # 利用阶段：使用网络输出
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                action = self.actors[i](obs_tensor).squeeze(0).detach().numpy()
                if noise:
                    action += self.noise.sample()
                action = np.clip(action, -0.8, 0.8)  # 限制最大速度
            actions.append(action)
        return actions
    
    def save_models(self, path_prefix):
        """保存所有模型参数"""
        for i in range(self.num_agents):
            torch.save({
                'actor': self.actors[i].state_dict(),
                'critic': self.critics[i].state_dict(),
                'target_actor': self.target_actors[i].state_dict(),
                'target_critic': self.target_critics[i].state_dict(),
                'actor_optimizer': self.actor_optimizers[i].state_dict(),
                'critic_optimizer': self.critic_optimizers[i].state_dict()
            }, f"{path_prefix}_agent{i}.pt")
    
    def load_models(self, path_prefix):
        """载入所有模型参数"""
        for i in range(self.num_agents):
            checkpoint = torch.load(f"{path_prefix}_agent{i}.pt")
            self.actors[i].load_state_dict(checkpoint['actor'])
            self.critics[i].load_state_dict(checkpoint['critic'])
            self.target_actors[i].load_state_dict(checkpoint['target_actor'])
            self.target_critics[i].load_state_dict(checkpoint['target_critic'])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizers[i].load_state_dict(checkpoint['critic_optimizer'])

# Ornstein-Uhlenbeck noise
class OUNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.copy(self.mu)
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

# Training function
# 修改训练函数以支持周期性可视化
# 修改训练函数
def train(env, maddpg, episodes=1000, batch_size=64, update_every=10, render_interval=100):
    buffer = ReplayBuffer(100000)
    scores = []
    plt.figure(figsize=(8, 8))
    plt.ion()
    
    for ep in range(episodes):
        obs = env.reset()
        ep_rewards = np.zeros(env.num_agents)
        done = False
        
        # 动态调整探索率
        exploration_rate = max(0.1, 1.0 - ep/500)  # 线性衰减
        
        while not done:
            actions = maddpg.act(obs, exploration_rate=exploration_rate)
            next_obs, rewards, done, _ = env.step(actions)
            
            buffer.add(obs, actions, rewards, next_obs, [done]*env.num_agents)
            obs = next_obs
            ep_rewards += np.array(rewards)
            
            if len(buffer) >= batch_size and ep % update_every == 0:
                batch = buffer.sample(batch_size)
                maddpg.update(batch)
            
            # 周期性渲染
            if ep % render_interval == 0:
                env.render()
                plt.pause(0.001)
        
        # 记录和保存
        scores.append(ep_rewards[0] + ep_rewards[1] - ep_rewards[2] - ep_rewards[3])
        if ep % 50 == 0:
            avg_score = np.mean(scores[-50:] if len(scores) > 50 else scores)
            print(f"Episode {ep}, Avg Score Diff: {avg_score:.2f}")
            if ep % 500 == 0:
                maddpg.save_models(f"maddpg_ep{ep}")
    
    plt.ioff()
    plt.close()
    
    # 绘制训练曲线
    plt.figure()
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Team1 - Team2 Score Difference")
    plt.title("Training Progress")
    plt.show()


# Testing function
# 修改测试函数以支持可视化
def test(env, maddpg, episodes=10, render=True, frame_delay=0.1):
    """
    参数说明：
    env: 环境实例
    maddpg: 训练好的模型
    episodes: 测试回合数
    render: 是否渲染可视化
    frame_delay: 每帧之间的延迟时间（秒），越大速度越慢
    """
    plt.figure(figsize=(8, 8))
    plt.ion()  # 开启交互模式
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        
        if render:
            env.render()
            plt.pause(2)  # 回合开始前延长暂停时间
            
        while not done:
            actions = maddpg.act(obs, noise=False)
            next_obs, rewards, done, _ = env.step(actions)
            obs = next_obs
            step += 1
            
            if render:
                env.render()
                plt.pause(frame_delay)  # 控制每帧显示时长
                
                # 在关键事件时额外暂停
                if any(rewards):  # 如果有得分
                    plt.pause(1)  # 得分后暂停1秒
                elif step % 10 == 0:  # 每10步暂停稍长
                    plt.pause(0.5)
                
        print(f"Test {ep}: Team1 {env.team1_goals} - {env.team2_goals} Team2 (Steps: {step})")
        
        if render:
            plt.pause(2)  # 回合结束后延长暂停时间
    
    if render:
        plt.ioff()  # 关闭交互模式
        plt.close()


# Main execution
# 修改主函数以支持交互式选择是否可视化
if __name__ == "__main__":
    env = AdversarialEnv()
    maddpg = MADDPG(env)
    
    # 训练或加载模型
    train_new = input("Train new model? (y/n): ").lower() == 'y'
    
    if train_new:
        print("Training...")
        train(env, maddpg, episodes=10000, render_interval=100)
        maddpg.save_models("maddpg_model")
        print("Model saved to maddpg_model_agent[0-3].pt")
    else:
        try:
            maddpg.load_models("maddpg_model")
            print("Model loaded from maddpg_model_agent[0-3].pt")
        except FileNotFoundError:
            print("No saved model found, training new model...")
            train(env, maddpg, episodes=1000, render_interval=100)
            maddpg.save_models("maddpg_model")
    
    # 测试
    print("\nTesting...")
    visualize = input("Show visualization during testing? (y/n): ").lower() == 'y'
    test(env, maddpg, episodes=10, render=visualize)