import torch
from game import SnakeGame
from neural.agent import DQNAgent
import time

def test(model_path, episodes=10, render=True):
    env = SnakeGame(grid_size=10, render=render)
    agent = DQNAgent(input_shape=(10,10), n_actions=3)
    agent.load(model_path)
    agent.epsilon = 0  # 完全利用

    scores = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, eval_mode=True)
            state, _, done, info = env.step(action)
            if render:
                env.render_game()
                time.sleep(0.1)
        scores.append(info['score'])
        print(f'Test Episode {episode+1}, Score: {info["score"]}')

    print(f'Average Score over {episodes} episodes: {sum(scores)/len(scores)}')

if __name__ == '__main__':
    test('models/snake_dqn_final.pth', episodes=10)
