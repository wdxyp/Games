import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from datetime import datetime


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # 增加一个隐藏层
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # 增加的隐藏层前向传播
        x = self.fc4(x)
        return x


# 初始化 pygame
pygame.init()

# 游戏窗口初始大小
width, height = 800, 600
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("打砖块游戏-V1.1")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)
BRICK_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# 字体设置，使用系统字体，调整统计信息字体大小
font = pygame.font.SysFont('simhei', 36)
small_font = pygame.font.SysFont('simhei', 24)


# 砖块类
class Brick:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))


# 球拍类
class Paddle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self):
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))

    def move(self, dx):
        self.x += dx
        if self.x < 0:
            self.x = 0
        elif self.x > width - self.width:
            self.x = width - self.width


# 球类
class Ball:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.dx = 3
        self.dy = -3

    def draw(self):
        pygame.draw.circle(screen, WHITE, (self.x, self.y), self.radius)

    def move(self):
        self.x += self.dx
        self.y += self.dy

        if self.x < self.radius or self.x > width - self.radius:
            self.dx = -self.dx
        if self.y < self.radius:
            self.dy = -self.dy


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(np.array(action).reshape(-1, 1))
        reward = torch.FloatTensor(np.array(reward).reshape(-1, 1))
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done).reshape(-1, 1))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# 创建游戏元素实例
paddle = Paddle(width // 2 - 75, height - 20, 150, 10)  # 增大球拍宽度
ball = Ball(width // 2, height // 2, 10)

# 砖块列表
bricks = []
brick_width, brick_height = 80, 30
for row in range(5):
    for col in range(8):  # 减少砖块列数
        brick = Brick(col * (brick_width + 5) + 20, row * (brick_height + 5) + 80, brick_width, brick_height,
                      BRICK_COLORS[row])
        bricks.append(brick)

# 定义DQN网络和优化器
input_size = 4  # 状态空间维度：球的x,y坐标，球拍的x坐标
hidden_size = 512
output_size = 3  # 动作空间维度：左移、不动、右移
policy_net = DQN(input_size, hidden_size, output_size)
target_net = DQN(input_size, hidden_size, output_size)
target_net.load_state_dict(policy_net.state_dict())
learning_rate = 0.1
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# 经验回放缓冲区
replay_buffer = ReplayBuffer(30000)  # 增大缓冲区容量

# 超参数
batch_size = 256
gamma = 0.995  # 增大折扣因子
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.05
num_episodes = 15000  # 增加训练回合数   未使用

# 指数衰减
def exponential_decay(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)

# 在每次选择动作时更新 epsilon
epsilon = exponential_decay(epsilon, epsilon_decay, min_epsilon)

# 统计数据
#try:
#     with open('game_stats.txt', 'r', encoding='utf-8') as f:
#         stats = f.read().split(',')
#         total_attempts = int(stats[0])
#         total_score = int(stats[1])
#         total_game_duration = float(stats[2])
#         try:
#             episode_scores = eval(stats[3])
#         except SyntaxError:
#             print("episode_scores 格式错误，重置为空列表。")
#             episode_scores = []
# except FileNotFoundError:
total_attempts = 0
total_score = 0
total_game_duration = 0
episode_scores = []
#     with open('game_stats.txt', 'w', encoding='utf-8') as f:
#         f.write(f"{total_attempts},{total_score},{total_game_duration},{episode_scores}")

game_start_time = None
game_duration = 0

# 游戏状态
running = False
paused = False
game_over = False

# 连续击中计数器
consecutive_hits = 0
# 无动作计数器
no_action_counter = 0

# 获取当前游戏状态
def get_state():
    return np.array([ball.x, ball.y, paddle.x, paddle.x + paddle.width])


# 重置游戏状态
def reset_game():
    global paddle, ball, bricks, game_start_time, game_duration, consecutive_hits, no_action_counter
    paddle = Paddle(width // 2 - 75, height - 20, 150, 10)  # 增大球拍宽度
    ball = Ball(width // 2, height // 2, 10)
    bricks = []
    brick_width, brick_height = 80, 30
    for row in range(5):
        for col in range(8):  # 减少砖块列数
            brick = Brick(col * (brick_width + 5) + 20, row * (brick_height + 5) + 80, brick_width, brick_height,
                          BRICK_COLORS[row])
            bricks.append(brick)
    adjust_game_elements()
    game_start_time = time.time()
    game_duration = 0
    consecutive_hits = 0
    no_action_counter = 0


# 绘制美化按钮
def draw_button(text, x, y, width, height, inactive_color, active_color, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    current_color = active_color if x + width > mouse[0] > x and y + height > mouse[1] > y else inactive_color
    pygame.draw.rect(screen, current_color, (x, y, width, height))
    pygame.draw.rect(screen, WHITE, (x, y, width, height), 2)  # 绘制白色边框
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.center = (x + width // 2, y + height // 2)
    screen.blit(text_surface, text_rect)
    if click[0] == 1 and action is not None and x + width > mouse[0] > x and y + height > mouse[1] > y:
        action()


# 球与球拍碰撞检测
def check_ball_paddle_collision(ball, paddle):
    if (
            ball.y + ball.radius >= height - paddle.height
            and ball.y - ball.radius <= height
            and ball.x + ball.radius >= paddle.x
            and ball.x - ball.radius <= paddle.x + paddle.width
    ):
        ball.dy = -ball.dy
        return True
    return False


# 球与砖块碰撞检测
def check_ball_brick_collision(ball, bricks):
    for brick in bricks.copy():
        if (
                ball.x + ball.radius >= brick.x
                and ball.x - ball.radius <= brick.x + brick.width
                and ball.y + ball.radius >= brick.y
                and ball.y - ball.radius <= brick.y + brick.height
        ):
            ball.dy = -ball.dy
            bricks.remove(brick)
            return True
    return False


# 调整游戏元素大小和位置
def adjust_game_elements():
    global paddle, ball, bricks
    # 调整球拍大小和位置
    paddle_width_ratio = 400 / 800
    paddle_height_ratio = 10 / 600
    paddle_width = int(width * paddle_width_ratio)
    paddle_height = int(height * paddle_height_ratio)
    paddle.x = width // 2 - paddle_width // 2
    paddle.y = height - paddle_height
    paddle.width = paddle_width
    paddle.height = paddle_height

    # 调整球的位置和半径
    ball_radius_ratio = 10 / 800
    ball.radius = int(width * ball_radius_ratio)
    ball.x = width // 2
    ball.y = height // 2

    # 调整砖块大小和位置
    bricks.clear()
    brick_width_ratio = 80 / 800
    brick_height_ratio = 30 / 600
    new_brick_width = int(width * brick_width_ratio)
    new_brick_height = int(height * brick_height_ratio)
    for row in range(5):
        for col in range(8):  # 减少砖块列数
            brick_x = col * (new_brick_width + 5 * width / 800) + 20 * width / 800
            brick_y = row * (new_brick_height + 5 * height / 600) + 80 * height / 600
            brick = Brick(brick_x, brick_y, new_brick_width, new_brick_height, BRICK_COLORS[row])
            bricks.append(brick)


# 游戏主循环
step_count = 0  # 用于更新目标网络的计数器
target_update_freq = 200  # 更频繁地更新目标网络

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
        #         with open('game_stats.txt', 'w', encoding='utf-8') as f:
        #             f.write(f"{total_attempts},{total_score},{total_game_duration + game_duration},{episode_scores}")
                pygame.quit()
                quit()
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.size
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                adjust_game_elements()

        screen.fill(BLACK)

        if not running:
            draw_button("开始游戏", width // 2 - 100, height // 2 - 30, 200, 60, GRAY, LIGHT_GRAY,
                        lambda: globals().update({'running': True, 'game_start_time': time.time()}))
            draw_button("退出游戏", width // 2 - 100, height // 2 + 30, 200, 60, GRAY, LIGHT_GRAY, pygame.quit)
        else:
            if game_over:
                draw_button("重新开始", width // 2 - 100, height // 2 - 30, 200, 60, GRAY, LIGHT_GRAY,
                            lambda: globals().update({'game_over': False}))
                draw_button("查看得分", width // 2 - 100, height // 2 + 30, 200, 60, GRAY, LIGHT_GRAY,
                            lambda: print("每局得分情况:", episode_scores))
                draw_button("退出游戏", width // 2 - 100, height // 2 + 90, 200, 60, GRAY, LIGHT_GRAY, pygame.quit)
            else:
                draw_button("暂停", width - 120, 10, 100, 40, GRAY, LIGHT_GRAY, lambda: globals().update({'paused': True}))
                if paused:
                    draw_button("继续", width - 120, 60, 100, 40, GRAY, LIGHT_GRAY, lambda: globals().update({'paused': False}))
                    draw_button("结束游戏", width - 120, 110, 100, 40, GRAY, LIGHT_GRAY, lambda: globals().update({'game_over': True}))
                else:
                    if game_start_time is not None:
                        game_duration = time.time() - game_start_time

                    # 根据epsilon贪婪策略选择动作
                    if random.random() < epsilon:
                        action = random.randint(0, 2)  # 0:左移, 1:不动, 2:右移
                    else:
                        state = torch.FloatTensor(get_state()).unsqueeze(0)
                        q_values = policy_net(state)
                        action = torch.argmax(q_values).item()

                    total_attempts += 1

                    # 执行动作
                    if action == 0:
                        paddle.move(-5 * width / 800)
                    elif action == 2:
                        paddle.move(5 * width / 800)

                    # 移动球
                    ball.move()

                    # 碰撞检测
                    paddle_hit = check_ball_paddle_collision(ball, paddle)
                    brick_hit = check_ball_brick_collision(ball, bricks)

                    reward = 0
                    if paddle_hit or brick_hit:
                        consecutive_hits += 1
                        base_reward = 10
                        bonus_reward = consecutive_hits * 2  # 连续击中奖励
                        reward = base_reward + bonus_reward
                        no_action_counter = 0
                    else:
                        consecutive_hits = 0
                        no_action_counter += 1

                    if no_action_counter > 50:  # 惩罚长时间无动作
                        reward = -10
                        no_action_counter = 0

                    # 在碰撞检测之后，增加球与球拍接近奖励
                    ball_paddle_distance = abs(ball.x - (paddle.x + paddle.width // 2))
                    if ball_paddle_distance < paddle.width // 2:
                        proximity_reward = (paddle.width // 2 - ball_paddle_distance) / (paddle.width // 2) * 2
                        reward += proximity_reward

                    total_score += reward
                    total_score = int(total_score)

                    # 球出界
                    single_done = False  # 用于游戏主逻辑的单一布尔值
                    if ball.y > height:
                        single_done = True
                        reward = -10

                    # 所有砖块被击碎
                    if len(bricks) == 0:
                        single_done = True
                        reward = 100  # 所有砖块被击碎的奖励

                    # 记录经验
                    next_state = get_state()
                    replay_buffer.push(get_state(), action, reward, next_state, single_done)

                    # 从经验回放中采样并训练
                    if len(replay_buffer) > batch_size:
                        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
                        q_values = policy_net(state)
                        next_q_values_policy = policy_net(next_state)
                        next_actions = torch.argmax(next_q_values_policy, 1).unsqueeze(1)
                        next_q_values_target = target_net(next_state)
                        max_next_q_values = next_q_values_target.gather(1, next_actions)
                        expected_q_values = reward + gamma * max_next_q_values * (1 - done)
                        criterion = nn.MSELoss()
                        loss = criterion(q_values.gather(1, action), expected_q_values)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # 更新目标网络
                    step_count += 1
                    if step_count % target_update_freq == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                    # 绘制游戏元素
                    paddle.draw()
                    ball.draw()
                    for brick in bricks:
                        brick.draw()

                    # 显示统计信息，调整布局
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    score_text = small_font.render(f"得分: {total_score}", True, WHITE)
                    screen.blit(score_text, (10, 10))
                    attempts_text = small_font.render(f"尝试次数: {total_attempts}", True, WHITE)
                    screen.blit(attempts_text, (10, 35))
                    time_text = small_font.render(f"当前时间: {current_time}", True, WHITE)
                    screen.blit(time_text, (10, 60))
                    duration_text = small_font.render(f"游戏时长: {int(total_game_duration + game_duration)} 秒", True, WHITE)
                    screen.blit(duration_text, (10, 85))

                    if single_done:
                        episode_scores.append(total_score)
                        total_score = 0
                        reset_game()

        pygame.display.flip()

        # 衰减epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

except Exception as e:
    # 保存游戏统计数据
    # with open('game_stats.txt', 'w', encoding='utf-8') as f:
    #     f.write(f"{total_attempts},{total_score},{total_game_duration + game_duration},{episode_scores}")
    # 打印异常信息
    print(f"发生异常: {e}")
    # 退出pygame
    pygame.quit()