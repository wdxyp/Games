import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
from datetime import datetime
from collections import deque

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 常量定义
MODEL_PATH = "breakout_model.pth"
MODEL_HISTORY_DIR = "model_history"
AUTOSAVE_INTERVAL = 100000  # 每10局自动保存一次


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


# 初始化 pygame
pygame.init()

# 游戏窗口初始大小
width, height = 800, 600
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)
pygame.display.set_caption("打砖块游戏-V2.0")

# 颜色定义
COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (100, 100, 100),
    'light_gray': (200, 200, 200),
    'brick': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0)]
}

# 字体设置
font = pygame.font.SysFont('simhei', 36)
small_font = pygame.font.SysFont('simhei', 24)


class Brick:
    def __init__(self, x, y, w, h, color):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)


class Paddle:
    def __init__(self, x, y, w, h, agent):
        self.rect = pygame.Rect(x, y, w, h)
        self.agent = agent  # 接收智能体实例
        self.speed = 8 + (1 - self.agent.epsilon) * 8  # 动态调整速度：epsilon=1时8，epsilon=0.2时9.6（原固定10）

    def move(self, dx):
        self.rect.x += dx * self.speed
        self.rect.clamp_ip(screen.get_rect())

    def draw(self, surface):
        pygame.draw.rect(surface, COLORS['white'], self.rect)


class Ball:
    def __init__(self, x, y, r):
        self.rect = pygame.Rect(x - r, y - r, 2 * r, 2 * r)
        self.dx = 3
        self.dy = -3
        self.speed = 3

    def move(self):
        self.rect.x += self.dx * self.speed
        self.rect.y += self.dy * self.speed

        # 边界碰撞
        if self.rect.left < 0 or self.rect.right > width:
            self.dx *= -1
        if self.rect.top < 0:
            self.dy *= -1

    def draw(self, surface):
        pygame.draw.circle(surface, COLORS['white'], self.rect.center, self.rect.width // 2)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class BreakoutAgent:
    def __init__(self):
        # 超参数
        self.input_size = 6  # 球x,y, 球拍x, 球速x,y, 砖块剩余
        self.hidden_size = 512  # 增大隐藏层提升模型容量（原128）
        self.output_size = 3  # 左, 不动, 右

        self.policy_net = DQN(self.input_size, self.hidden_size, self.output_size).to(device)
        self.target_net = DQN(self.input_size, self.hidden_size, self.output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)  # 恢复标准学习率设置（原0.01→0.001）
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.9)  # 延长学习率衰减步长

        self.gamma = 0.99  # 增大折扣因子（原0.95），更重视未来奖励
        self.epsilon = 1.0
        self.epsilon_decay = 0.995  # 减缓探索率衰减（原0.995）
        self.min_epsilon = 0.1  # 提高最小探索率（原0.05），保持长期探索
        self.batch_size = 256  # 增大批量大小（原64），提升梯度稳定性
        self.target_update = 300  # 降低目标网络更新频率（原200），减少训练波动

        self.memory = ReplayBuffer(50000)  # 增大经验池容量（原10000）
        self.steps = 0
        self.recent_scores = deque(maxlen=10)  # 存储最近10局得分

        # 模型自动加载
        if os.path.exists(MODEL_PATH):
            self.load_model()
        else:
            print("未找到已有模型，开始新训练")
    def act(self, state):
        if np.random.rand() < self.epsilon:
            # 玻尔兹曼探索：根据Q值计算概率
            q_values = self.policy_net(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
            probs = np.exp(q_values / 0.1)  # 温度参数0.1控制随机性（越小越确定）
            probs /= np.sum(probs)
            return np.random.choice(self.output_size, p=probs)
        else:
            return self.policy_net(torch.FloatTensor(state).to(device)).argmax().item()
    def save_model(self, path=MODEL_PATH):
        """保存完整训练状态（支持指定路径）"""
        # 保存当前模型
        save_data = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        torch.save(save_data, MODEL_PATH)
        print(f"模型已保存到 {MODEL_PATH}")

        # 保存历史版本
        if not os.path.exists(MODEL_HISTORY_DIR):
            os.makedirs(MODEL_HISTORY_DIR)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = f"{MODEL_HISTORY_DIR}/model_{timestamp}.pth"
        torch.save(save_data, history_path)
        print(f"历史版本保存到 {history_path}")

    def load_model(self, path=MODEL_PATH):
        """加载训练状态（支持指定路径）"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"从 {MODEL_PATH} 加载已保存模型")

    def get_state(self, paddle, ball, bricks):
        state = [
            ball.rect.centerx / width,  # 归一化坐标
            ball.rect.centery / height,
            paddle.rect.centerx / width,
            ball.dx / 10.0,  # 速度归一化
            ball.dy / 10.0,
            len(bricks) / 30.0  # 剩余砖块比例
        ]
        return np.array(state, dtype=np.float32)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        if self.steps % 50 == 0:  # 每1000步检查一次
            recent_scores = self.recent_scores  # 假设记录了最近10局得分（需在Game类中维护）
        if np.std(self.recent_scores) < 500:  # 得分波动小（策略固定）
            self.epsilon = min(0.5, self.epsilon + 0.1)  # 临时提升探索率
        
        # 采样批次数据
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        # 转换为张量
        state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
        action_batch = torch.LongTensor(batch[1]).to(device)
        reward_batch = torch.FloatTensor(batch[2]).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(device)
        done_batch = torch.BoolTensor(batch[4]).to(device)

        # 计算当前Q值
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # 计算目标Q值
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        target_q = reward_batch + (self.gamma * next_q * ~done_batch)

        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # 优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # 更新目标网络
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return loss.item()


class Game:
    def __init__(self):
        import os
        os.makedirs("model_history", exist_ok=True)  # 创建model_history文件夹（如果不存在）
        self.level = 1
        self.total_score = 0
        self.high_score = 0
        self.episode = 0
        self.start_time = time.time()
        self.total_play_time = 0
        self.running = True  # 始终自动运行
        self.agent = BreakoutAgent()
        self.level_complete = False  # 关卡完成标记
        self.level_complete_time = 0  # 关卡完成时间
        self.game_complete = False  # 全部关卡完成标记
        self.reset()

    def reset(self):
        """重置游戏状态"""
        self.episode += 1

        # 自动保存逻辑
        if self.episode % AUTOSAVE_INTERVAL == 0:
            self.agent.save_model()

        # 更新最高分
        if self.total_score > self.high_score:
            self.high_score = self.total_score
            self.agent.save_model()

        # 记录当前局得分到agent的最近得分列表
        self.agent.recent_scores.append(self.total_score)

        # 重置游戏状态（仅游戏失败时重置关卡）
        if not getattr(self, 'level_up', False):
            self.level = 1
        # 关卡n（n≥2）开始时加载上一关卡的模型
        if self.level >= 2:
            load_path = f"model_history/breakout_level{self.level - 1}_model.pth"
            if os.path.exists(load_path):
                self.agent.load_model(load_path)
                print(f"[关卡提示] 关卡{self.level}开始，已加载关卡{self.level - 1}保存的模型参数")
            else:
                print(f"[警告] 未找到关卡{self.level - 1}保存的模型，使用当前模型继续游戏")
        self.paddle = Paddle(width // 2 - 75, height - 40, 300, 10, self.agent)  # 传递智能体实例到球拍类
        self.ball = Ball(width // 2, height // 2, 10)
        self.bricks = self.create_bricks()
        self.consecutive_hits = 0
        self.failed_attempts = 0
        self.current_action = 1  # 初始化当前动作为不动
        self.episode_start = time.time()
        self.total_score = 0

    def create_bricks(self):
        bricks = []
        rows = 4 + self.level
        cols = 7 + self.level
        brick_w = width // cols - 5
        brick_h = 30
        for row in range(rows):
            for col in range(cols):
                x = col * (brick_w + 5) + 20
                y = row * (brick_h + 5) + 80
                color = COLORS['brick'][(row + self.level) % 5]
                bricks.append(Brick(x, y, brick_w, brick_h, color))
        return bricks

    def calculate_reward(self, paddle_hit, brick_hit, done):
        reward = 0

        if brick_hit:
            self.consecutive_hits += 1
            reward = 10 * (1 + self.level / 5) + self.consecutive_hits * 3  # 降低砖块击中基础奖励（原20→10）并调整连击系数（原5→3）
            self.failed_attempts = 0
        elif paddle_hit:
            reward = 10  # 增大球拍击中奖励（原5→10）
            # 根据球的水平方向给予额外奖励（右移时选右，左移时选左）
            if (self.ball.dx > 0 and self.current_action == 2) or (self.ball.dx < 0 and self.current_action == 0):
                reward += 8  # 方向匹配额外奖励
            self.consecutive_hits = 0
            self.failed_attempts = 0
        else:
            self.consecutive_hits = 0
            self.failed_attempts += 1
            reward = -3 ** (self.failed_attempts // 8)  # 加重失败惩罚（原2→3）但减缓增长速度
        # 计算球拍位置相对于屏幕中心的偏离度（0-1，偏离越大值越大）
        paddle_center = self.paddle.rect.x + self.paddle.rect.width // 2
        screen_center = width // 2
        position_deviation = abs(paddle_center - screen_center) / (width // 2)
        # 偏离度适中时给予奖励（避免总在边缘）
        if 0.2 < position_deviation < 0.8:
            reward += 3  # 适中位置奖励
        if done:
            reward -= 40  # 加重游戏结束惩罚（原20→40）

        # 生存奖励
        reward += 0.5  # 增大生存奖励（原0.1→0.5）

        self.total_score += reward
        return reward

    def run(self):
        clock = pygame.time.Clock()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                screen.fill(COLORS['black'])

                # 游戏逻辑
                state = self.agent.get_state(self.paddle, self.ball, self.bricks)
                action = self.agent.select_action(state)

                # 执行动作
                if action == 0:  # 左
                    self.paddle.move(-1)
                elif action == 2:  # 右
                    self.paddle.move(1)

                # 更新球的位置
                self.ball.move()

                # 碰撞检测
                paddle_hit = self.ball.rect.colliderect(self.paddle.rect)
                if paddle_hit:
                    self.ball.dy *= -1  # 球拍碰撞时反转垂直速度
                brick_hit = False
                for brick in self.bricks[:]:
                    if self.ball.rect.colliderect(brick.rect):
                        self.bricks.remove(brick)
                        self.ball.dy *= -1
                        brick_hit = True
                        break

                # 检查游戏状态（关卡设置 10）
                done = self.ball.rect.top > height
                if not self.bricks:
                    # 保存当前关卡模型
                    import os
                    os.makedirs("model_history", exist_ok=True)
                    current_level = self.level
                    save_path = f"model_history/breakout_level{current_level}_model.pth"
                    self.agent.save_model(save_path)
                    print(f"[关卡提示] 关卡{current_level}完成，已保存模型参数到 {save_path}")
                    
                    self.level = min(self.level + 1, 10)
                    self.level_complete = True
                    self.level_complete_time = time.time()
                    if self.level == 10:
                        self.game_complete = True  # 标记所有关卡完成
                    self.level_up = True
                    self.reset()
                    del self.level_up


                # 计算奖励
                reward = self.calculate_reward(paddle_hit, brick_hit, done)
                next_state = self.agent.get_state(self.paddle, self.ball, self.bricks)

                # 存储经验
                self.agent.memory.add(state, action, reward, next_state, done)

                # 训练模型
                loss = self.agent.update_model()

                # 绘制游戏元素
                self.paddle.draw(screen)
                self.ball.draw(screen)
                for brick in self.bricks:
                    brick.draw(screen)

                # 绘制关卡完成提示（持续2秒）
                if self.level_complete and (time.time() - self.level_complete_time) < 2:
                    text = font.render('恭喜过关！/n 按任意键继续', True, COLORS['white'])
                    text_rect = text.get_rect(center=(width//2, height//2))
                    screen.blit(text, text_rect)
                    pygame.display.flip()
                    
                    # 等待按键
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                waiting = False
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return

                # 所有关卡完成后停止更新
                if self.game_complete:
                    pygame.display.flip()
                    while True:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return

                # 计算时间统计
                current_time = time.time()
                total_time = current_time - self.start_time
                episode_time = current_time - self.episode_start

                # 显示统计信息（横向排列）
                debug_info = [
                    f"总局: {self.episode}",
                    f"关卡: {self.level}",
                    f"得分: {int(self.total_score)}",
                    f"最高: {int(self.high_score)}",
                    f"总时: {int(total_time)}s",
                    f"本局: {int(episode_time)}s",
                    f"ε: {self.agent.epsilon:.2f}",
                    f"Loss: {loss:.4f}" if loss else ""
                ]

                debug_font = pygame.font.SysFont('simhei', 14)  # 使用更小的字体
                x_pos = 20  # 起始X坐标
                y_pos = 10  # Y坐标固定在顶部
                spacing = 15  # 项间距

                for text in debug_info:
                    # 带背景的文字渲染
                    surf = debug_font.render(text, True, COLORS['white'], COLORS['black'])
                    rect = surf.get_rect(topleft=(x_pos, y_pos))
                    screen.blit(surf, rect)
                    x_pos += rect.width + spacing  # 自动计算下一个项的位置

                if done:
                    self.level_up = True
                    self.reset()
                    del self.level_up

                pygame.display.flip()
                clock.tick(60)
        finally:
            # 确保程序退出前保存
            self.agent.save_model()


if __name__ == "__main__":
    game = Game()
    game.run()