# -*- coding: utf-8 -*-
# RoboMaster AI Challenge Simulator (RMAICS)

from kernal import kernal
import numpy as np

class rmaics(object):

    def __init__(self, agent_num, render=True):
        self.game = kernal(car_num=agent_num, render=render)
        self.g_map = self.game.get_map()
        self.memory = []
        self.agent_num = agent_num # 机器人数量
        self.pre_car_hit=0

    def reset(self):
        self.state = self.game.reset()
        # state, object
        self.obs = self.get_observation(self.state)
        return self.obs

    def step(self, actions):
        state = self.game.step(actions)
        obs = self.get_observation(state)
        rewards = self.get_reward(state)

        self.memory.append([self.obs, actions, rewards])
        self.state = state

        # return obs, rewards, state.done, None
        return obs, rewards, state.done
    
    def get_observation(self, state):
        # personalize your observation here
        # obs = state
        if self.agent_num > 1:
            obs = np.concatenate(([item for agent in state.agents for item in agent],state.compet.ravel()))
        else:
            obs = np.concatenate((state.agents[0].ravel(),state.compet.ravel()))
        return obs
    
    def get_reward(self, state):
        # 基础奖励，鼓励存活
        reward = 0.1
        # print(state.agents)
        # 收到伤害惩罚
        damage_dealt = max(0, state.agents[0][6] - state.agents[0][17])  # 当前血量减去上一时刻的血量
        reward -= damage_dealt * 0.5

        # 胜利和失败奖励
        # if state.agents[0][6] <= 0:
        #     reward -= 100  # 失败的惩罚
        # elif state.agents[0][6] > 0 :
        #     reward += 100  # 胜利的奖励

        # 射击奖励
        if state.agents[0][4] == 1:  # 如果正在射击
            reward += 0.2

        # 补给收集奖励
        if state.agents[0][8] == 1:  # 如果不在补给状态
            reward += 1

        # 位置奖励，假设地图中心是 (400, 250)
        distance_to_center = np.sqrt((state.agents[0][1] - 600)**2 + (state.agents[0][2] - 400)**2)
        reward += distance_to_center * (-0.01)

        # 防御动作奖励
        # 假设 state[11] 表示在防御加成区待的时间
        if state.agents[0][11]:  # 如果待的时间超过500
            reward += 0.02*state.agents[0][11]
        if self.pre_car_hit-state.agents[0][14] < 0:  # 如果轮子或装甲板撞车
            reward -= 10

        # if state.agents[0][10] <= 0 and state.agents[0][4] == 1:  # 如果剩余子弹量不足，但仍然尝试发射子弹
        #     reward -= 1

        # if abs(state.agents[0][4]) > 90:  # 如果云台相对角度长时间大于+-90
        #     reward -= 0.1

        if state.agents[0][8] == 1 and state.agents[0][7] > 0:  # 如果不在补弹区但尝试使用补弹
            reward -= 1
        # 时间惩罚
        # reward -= 0.01

        self.pre_car_hit = state.agents[0][14]

        return reward


    def play(self):
        self.game.play()

    def save_record(self, file):
        self.game.save_record(file)