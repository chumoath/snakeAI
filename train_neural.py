import torch
from game import SnakeGame
from neural.agent import DQNAgent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

def train(episodes=1000, save_interval=100, render=False):
    env = SnakeGame(grid_size=10, render=render)
    agent = DQNAgent(input_shape=(10,10), n_actions=3)  # 3种动作
    writer = SummaryWriter('logs')

    scores = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            if render:
                env.render_game()

        scores.append(info['score'])
        writer.add_scalar('score', info['score'], episode)
        writer.add_scalar('epsilon', agent.epsilon, episode)

        if episode % save_interval == 0 and episode > 0:
            agent.save(f'models/snake_dqn_{episode}.pth')
            print(f'Episode {episode}, Score: {info["score"]}, Avg Score: {np.mean(scores[-50:])}')

    writer.close()
    agent.save('models/snake_dqn_final.pth')
    print('Training finished.')

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    train(episodes=2000, render=False)
