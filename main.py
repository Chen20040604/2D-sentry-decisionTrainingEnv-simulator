# import gym
import gymnasium as gym
from qlearn import Agent
from utils import plotLearning
import numpy as np
import torch
import pickle
from rmaics import rmaics
from kernal import record_player
import time

# Function to save the Agent instance
def save_agent(agent, filename):
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)

# Function to load the Agent instance
def load_agent(filename):
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    return agent

if __name__ == '__main__':
    game = rmaics(agent_num=1, render=True)
    game.reset()
    agent = Agent(gamma=0.9, epsilon=0.9, batch_size=64, n_actions=6, eps_end=0.01,
                  input_dims=[26], lr=0.001)
    scores, eps_history = [], []
    n_games = 5
    
    for i in range(n_games):
        print('test: start one episode')
        score = 0
        done = False
        observation = game.reset()
        while not done:
            
            action = agent.choose_action(observation)  
            # observation_, reward, done, info = env.step(action)
            observation_, reward, done = game.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    # Save the Agent instance to a pickle file
    save_agent(agent, 'RM-Sentry1209.pkl')
    # filename = 'RM-SentryDecision.png'
    # plotLearning(x, scores, eps_history, filename)