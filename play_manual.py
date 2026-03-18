import pygame
from game import SnakeGame

def play_manual():
    # 初始化游戏，开启渲染
    env = SnakeGame(grid_size=10, cell_size=40, render=True)
    env.reset()
    clock = pygame.time.Clock()
    started = False  # 是否已开始移动
    font = pygame.font.Font(None, 36)  # 用于显示提示文字

    while True:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                # 计算新方向
                new_dir = None
                if event.key in (pygame.K_UP, pygame.K_w):
                    new_dir = (0, -1)  # 上
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    new_dir = (0, 1)   # 下
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    new_dir = (-1, 0)  # 左
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    new_dir = (1, 0)   # 右

                if new_dir:
                    if not started:
                        # 第一次按键：启动游戏，设置方向
                        env.direction = new_dir
                        started = True
                        # 立即执行一步（让蛇开始移动）
                        _, _, done, info = env.step(0)
                        if done:
                            print(f"Game Over! Final Score: {info['score']}")
                            pygame.quit()
                            return
                    else:
                        # 已开始后，允许改变方向（禁止反向）
                        if not (new_dir[0] == -env.direction[0] and new_dir[1] == -env.direction[1]):
                            # 手动更新方向，更新完方向只需要forward;agent需要被step转换防向
                            env.direction = new_dir

        # 如果已开始，执行一步移动
        if started:
            _, _, done, info = env.step(0)
            if done:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.quit()
                return

        # 渲染游戏
        env.render_game()

        # 如果未开始，显示提示文字
        if not started:
            # 在屏幕中央绘制提示
            text = font.render("Press any arrow key to start", True, (255, 255, 255))
            text_rect = text.get_rect(center=(env.width // 2, env.height // 2))
            env.screen.blit(text, text_rect)
            pygame.display.flip()

        clock.tick(1)  # 控制帧率

if __name__ == "__main__":
    play_manual()
