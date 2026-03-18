import pygame
import time
from game import SnakeGame
from llm.llm_agent import LocalLLMAgent
from llm.switch_llm_model import get_model

def play_with_llm():
    # 初始化游戏（开启渲染）
    env = SnakeGame(grid_size=10, render=True)
    
    # 初始化本地大模型智能体（请确保Ollama已运行且有模型）
    agent = LocalLLMAgent(model_name=get_model())  # 可根据需要换其他模型
    
    # 重置游戏
    state = env.reset()
    done = False
    total_score = 0
    step_count = 0
    
    print("大模型开始思考...")
    
    clock = pygame.time.Clock()
    
    while not done:
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # 调用大模型获取动作（传入蛇、食物、方向信息）
        action = agent.get_action(env.snake, env.food, env.direction)
        
        # 执行动作
        state, reward, done, info = env.step(action)
        total_score = info['score']
        step_count += 1
        
        # 渲染游戏
        env.render_game()
        
        # 控制帧率，留出推理时间（Ollama推理通常需要几百毫秒到几秒）
        # 这里设置较慢的帧率，避免过快移动
        clock.tick(2)  # 每秒2帧，给大模型足够时间
        
        # 可选：显示当前决策信息
        if step_count % 5 == 0:
            print(f"步骤 {step_count}, 得分 {total_score}, 上次输出: {agent.last_response}")
    
    print(f"游戏结束！最终得分: {total_score}, 总步数: {step_count}")
    pygame.quit()

if __name__ == "__main__":
    play_with_llm()
