import requests
import json
import time

class LocalLLMAgent:
    def __init__(self, model_name="qwen2.5:7b", base_url="http://localhost:11434"):
        """
        初始化本地大模型智能体
        :param model_name: Ollama中的模型名称（如 qwen2.5:7b, llama3:8b 等）
        :param base_url: Ollama API地址
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # 动作映射：将文本动作转换为游戏动作索引
        self.action_map = {
            "left": 1,    # 左转
            "forward": 0, # 直行
            "right": 2    # 右转
        }
        
        # 缓存上一次的响应，可用于调试
        self.last_response = None

    def _build_prompt(self, snake, food, direction):
        """
        将游戏状态构建为提示词
        """
        head = snake[0]
        dir_names = {(1,0):"右", (0,1):"下", (-1,0):"左", (0,-1):"上"}
        dir_str = dir_names.get(direction, "未知")
        
        # 计算前方、左转、右转的位置
        forward_pos = (head[0] + direction[0], head[1] + direction[1])
        left_dir = (-direction[1], direction[0])  # 左转90度
        right_dir = (direction[1], -direction[0]) # 右转90度
        left_pos = (head[0] + left_dir[0], head[1] + left_dir[1])
        right_pos = (head[0] + right_dir[0], head[1] + right_dir[1])
        
        # 曼哈顿距离
        dist_to_food = abs(head[0] - food[0]) + abs(head[1] - food[1])
        
        prompt = f"""你是一个贪吃蛇游戏AI，需要根据当前游戏状态选择下一步移动方向。网格大小为10x10，坐标从(0,0)到(9,9)。

当前状态：
- 蛇头位置：({head[0]}, {head[1]})
- 当前方向：{dir_str}
- 食物位置：({food[0]}, {food[1]})
- 蛇身长度：{len(snake)}，蛇身坐标：{snake[1:] if len(snake)>1 else []}
- 到食物的曼哈顿距离：{dist_to_food}

分析各个方向的危险情况：
- 前方({forward_pos[0]}, {forward_pos[1]})：{'墙壁' if forward_pos[0]<0 or forward_pos[0]>=10 or forward_pos[1]<0 or forward_pos[1]>=10 else '空地'}，{'自身' if forward_pos in snake[1:] else '安全'}
- 左转({left_pos[0]}, {left_pos[1]})：{'墙壁' if left_pos[0]<0 or left_pos[0]>=10 or left_pos[1]<0 or left_pos[1]>=10 else '空地'}，{'自身' if left_pos in snake[1:] else '安全'}
- 右转({right_pos[0]}, {right_pos[1]})：{'墙壁' if right_pos[0]<0 or right_pos[0]>=10 or right_pos[1]<0 or right_pos[1]>=10 else '空地'}，{'自身' if right_pos in snake[1:] else '安全'}

指令：基于以上信息，选择最安全且能尽快吃到食物的移动方向。只输出一个词：left（左转）、forward（直行）或right（右转）。不要输出任何其他内容。"""

        return prompt

    def get_action(self, snake, food, direction):
        """
        调用本地大模型获取动作
        :return: 动作索引 0:直行, 1:左转, 2:右转
        """
        prompt = self._build_prompt(snake, food, direction)
        
        # 调用Ollama API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,   # 低温度保证确定性
            "max_tokens": 10
        }
        
        try:
            print (payload)
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            raw_output = result.get("response", "").strip().lower()
            print(raw_output)
            self.last_response = raw_output
            
            # 解析输出（只取第一个词）
            words = raw_output.split()
            if words:
                action_word = words[0]
                # 处理可能的同义词或拼写错误
                if action_word in ["left", "左"]:
                    return 1
                elif action_word in ["right", "右"]:
                    return 2
                elif action_word in ["forward", "straight", "前", "直"]:
                    return 0
            # 默认直行
            return 0
            
        except Exception as e:
            print(f"调用Ollama失败: {e}")
            return 0  # 出错时默认直行
