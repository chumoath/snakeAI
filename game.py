import pygame
import numpy as np
import random

class SnakeGame:
    def __init__(self, grid_size=10, cell_size=40, render=False):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render = render
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # 蛇：列表，每个元素为(x, y)，头部在索引0
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (1, 0)  # 初始向右
        # 添加一段身体（尾部）
        self.snake.append((self.snake[0][0] - 1, self.snake[0][1]))
        self.food = self._place_food()
        self.score = 0
        self.done = False
        self.steps = 0
        return self._get_state()

    def _place_food(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in self.snake:
                return pos

    def _get_state(self):
        """返回网格状态：0空，1蛇身，2食物，3蛇头（可选）"""
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i, segment in enumerate(self.snake):
            if i == 0:
                state[segment[1], segment[0]] = 3  # 蛇头
            else:
                state[segment[1], segment[0]] = 1  # 蛇身
        state[self.food[1], self.food[0]] = 2
        return state

    def step(self, action):
        """
        action: 0-直行，1-左转，2-右转（相对于当前方向）
        返回：next_state, reward, done, info
        """
        # 计算新方向和新蛇头
        dirs = [(1,0), (0,1), (-1,0), (0,-1)]  # 右、下、左、上
        current_idx = dirs.index(self.direction)
        #          1       2
        # right   up     down
        # left    down    up
        # up      left    right
        # down    left    right
        if current_idx == 0 and action == 1:
            new_idx = 3
        elif current_idx == 0 and action == 2:
            new_idx = 1
        elif current_idx == 2 and action == 1:
            new_idx = 1
        elif current_idx == 2 and action == 1:
            new_idx = 3
        elif action == 1:
            new_idx = 2
        elif action == 2:
            new_idx = 0
        else:
            new_idx = current_idx  # 直行

        new_dir = dirs[new_idx]
        self.direction = new_dir

        new_head = (self.snake[0][0] + new_dir[0], self.snake[0][1] + new_dir[1])

        # 检查是否吃到食物
        ate_food = (new_head == self.food)

        # ---------- 碰撞检测（先检测，不更新蛇）----------
        # 撞墙
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return self._get_state(), -10, True, {"score": self.score}

        # 自身碰撞
        if ate_food:
            # 吃到食物时，尾部保留，新蛇头不能与任何现有身体重叠
            if new_head in self.snake:
                return self._get_state(), -10, True, {"score": self.score}
        else:
            # 没吃到食物时，尾部会移除，新蛇头不能与 snake[1:] 重叠（不包括尾部）
            # 但允许新蛇头等于当前尾部（因为尾部即将移除）
            if new_head in self.snake[1:]:  # 检查除头部外的所有身体
                if new_head != self.snake[-1]:  # 如果不是尾部，则碰撞
                    return self._get_state(), -10, True, {"score": self.score}

        # ---------- 无碰撞，更新蛇 ----------
        if ate_food:
            self.snake.insert(0, new_head)
            self.score += 1
            self.food = self._place_food()
            reward = 10
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = -0.1

        self.steps += 1
        if self.steps > 1000:  # 防止无限循环
            self.done = True

        return self._get_state(), reward, self.done, {"score": self.score}

    def render_game(self):
        if not self.render:
            return
        self.screen.fill((0, 0, 0))
        # 画网格线
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (self.width, y))

        # 画食物
        fx, fy = self.food
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size))

        # 画蛇
        for i, segment in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # 蛇头亮绿色
            pygame.draw.rect(self.screen, color,
                             (segment[0] * self.cell_size, segment[1] * self.cell_size,
                              self.cell_size, self.cell_size))
        pygame.display.flip()
        self.clock.tick(10)  # 控制帧率
