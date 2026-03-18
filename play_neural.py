import pygame
from game import SnakeGame
from neural.agent import DQNAgent

def auto_play(model_path):
    env = SnakeGame(grid_size=10, render=True)
    agent = DQNAgent(input_shape=(10,10), n_actions=3)
    agent.load(model_path)
    agent.epsilon = 0  # 纯利用

    state = env.reset()
    done = False
    total_score = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        action = agent.act(state, eval_mode=True)
        state, _, done, info = env.step(action)
        total_score = info['score']
        env.render_game()
    print(f'Game Over! Final Score: {total_score}')
    pygame.quit()

if __name__ == '__main__':
    auto_play('models/snake_dqn_final.pth')
