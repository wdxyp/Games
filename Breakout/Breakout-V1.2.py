import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from datetime import datetime
import os

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)  # 使用双缓冲
pygame.display.set_caption("打砖块游戏-V1.2")

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

# 经验回放区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

# 新增关卡相关变量
level = 1  # 新增当前关卡
max_level = 5  # 新增最大关卡数
level_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,165,0)]  # 新增关卡颜色

# 创建游戏元素实例
paddle = Paddle(width // 2 - 75, height - 20, 150, 10)  # 增大球拍宽度
ball = Ball(width // 2, height // 2, 10)
# 砖块列表
bricks = []

# 定义DQN网络和优化器
input_size = 4  # 状态空间维度：球的x,y坐标，球拍的x坐标
hidden_size = 256  # 增加隐藏层神经元数量
output_size = 3  # 动作空间维度：左移、不动、右移
policy_net = DQN(input_size, hidden_size, output_size).to(device)
model_path = r'c:\Users\zhengchunji\PycharmProjects\PythonProject\breakout_model.pth'  # 新增模型路径
if os.path.exists(model_path):  # 新增模型加载
    policy_net.load_state_dict(torch.load(model_path))
    print("成功加载已保存模型")  # 将模型移动到 GPU 上
target_net = DQN(input_size, hidden_size, output_size).to(device)  # 将模型移动到 GPU 上
target_net.load_state_dict(policy_net.state_dict())
learning_rate = 0.05  # 降低学习率
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)  # 学习率调度器

# 经验回放区
replay_buffer = ReplayBuffer(15000)  # 增加缓冲区容量

# 超参数
batch_size = 128  # 增加批量大小
gamma = 0.99  # 调整折扣因子
epsilon = 1.0
epsilon_decay = 0.999  # 调整epsilon衰减率
min_epsilon = 0.05
num_episodes = 20000  # 增加训练回合数

# 指数衰减
def exponential_decay(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)

# 在每次选择动作时更新 epsilon
epsilon = exponential_decay(epsilon, epsilon_decay, min_epsilon)

# 统计数据
total_attempts = 0
total_score = 0
total_game_duration = 0
episode_scores = []
game_start_time = None
game_duration = 0

# 在全局变量区域添加失败计数器
failed_attempts = 0  # 新增失败尝试计数器
max_failed_attempts = 10  # 新增最大允许失败次数

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
# 修改 reset_game 函数，添加关卡重置逻辑
def reset_game():
    global paddle, ball, bricks, game_start_time, game_duration, consecutive_hits, no_action_counter, level
    # 重置关卡为 1
    level = 1
    paddle = Paddle(width // 2 - 75, height - 20, 150, 10)  # 增大球拍宽度
    ball = Ball(width // 2, height // 2, 10)
    bricks = []
    # 修改后的砖块生成逻辑（根据关卡变化）
    rows = 4 + level  # 每关增加一行
    cols = 7 + level  # 每关增加一列
    brick_width, brick_height = 80, 30
    for row in range(rows):
        for col in range(cols):
            color = level_colors[(row + level) % 5]  # 与上面保持一致的修改
            brick = Brick(col * (brick_width + 5) + 20, row * (brick_height + 5) + 80, 
                         brick_width, brick_height, color)
            bricks.append(brick)
    adjust_game_elements()
    game_start_time = time.time()
    game_duration = 0
    consecutive_hits = 0
    no_action_counter = 0
    # 每100个回合保存一次模型
    if total_attempts % 100 == 0:
        torch.save(policy_net.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")
    return  # 保持原有返回

# 定义统一的按钮宽度、高度和字体
button_width = 120
button_height = 30
button_font = pygame.font.SysFont('simhei', 18)  # 与状态栏字体大小一致

# 修改绘制按钮的函数，使用新的字体
def draw_button(text, x, y, width, height, inactive_color, active_color, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    current_color = active_color if x + width > mouse[0] > x and y + height > mouse[1] > y else inactive_color
    pygame.draw.rect(screen, current_color, (x, y, width, height))
    pygame.draw.rect(screen, WHITE, (x, y, width, height), 2)  # 绘制白色边框
    text_surface = button_font.render(text, True, WHITE)  # 使用新的字体
    text_rect = text_surface.get_rect()
    text_rect.center = (x + width // 2, y + height // 2)
    screen.blit(text_surface, text_rect)
    if click[0] == 1 and action is not None and x + width > mouse[0] > x and y + height > mouse[1] > y:
        action()

# 球与球拍碰撞检测
def check_ball_paddle_collision(ball, paddle):
    return (
        ball.y + ball.radius >= height - paddle.height
        and ball.y - ball.radius <= height
        and ball.x + ball.radius >= paddle.x
        and ball.x - ball.radius <= paddle.x + paddle.width
    )

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

# 调整游戏元素大小和位置（保留这个带关卡逻辑的新版本）
# 修改1：调整砖块初始化逻辑
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

    # 调整砖块大小和位置（根据当前关卡）
    bricks.clear()
    brick_width_ratio = 80 / 800
    brick_height_ratio = 30 / 600
    new_brick_width = int(width * brick_width_ratio)
    new_brick_height = int(height * brick_height_ratio)
    game_area_top = int(height * 0.15)  # 动态计算基准位置，可根据实际情况调整比例

    rows = 4 + level  # ← 正确逻辑
    cols = 7 + level  # ← 正确逻辑
    for row in range(rows):
        for col in range(cols):
            brick_x = col * (new_brick_width + 5) + 20
            # 应用动态基准位置，确保砖块从状态栏下方开始绘制
            brick_y = game_area_top + row * (new_brick_height + 5)  
            color = level_colors[(row + level) % 5]
            brick = Brick(brick_x, brick_y, new_brick_width, new_brick_height, color)
            bricks.append(brick)

# 修改奖励计算函数
def calculate_reward(paddle_hit, brick_hit, consecutive_hits, no_action_counter, failed_attempts, max_failed_attempts):
    reward = 0
    if brick_hit:
        consecutive_hits += 1
        base_reward = 10
        bonus_reward = consecutive_hits * 2  # 连续击中奖励
        reward = base_reward + bonus_reward
        no_action_counter = 0
        failed_attempts = 0  # 重置失败计数器
    elif paddle_hit:
        # 球拍击中给予较小的奖励
        reward = 2
        consecutive_hits = 0
        no_action_counter = 0
        failed_attempts = 0
    else:
        consecutive_hits = 0
        no_action_counter += 1
        failed_attempts += 1  # 增加失败计数
        if no_action_counter > 50:  # 惩罚长时间无动作
            reward = -10
            no_action_counter = 0
        if failed_attempts >= max_failed_attempts:
            reward = -20  # 连续失败时加大惩罚
        else:
            reward = -1  # 普通失败惩罚调小

    return reward, consecutive_hits, no_action_counter, failed_attempts

# 游戏主循环
step_count = 0  # 用于更新目标网络的计数器
target_update_freq = 100  # 更频繁地更新目标网络
clock = pygame.time.Clock()  # 创建时钟对象
fps = 60  # 设置帧率

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.size
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)  # 使用双缓冲
                adjust_game_elements()

        screen.fill(BLACK)

        if not running:
            # 调整开始菜单按钮大小和字体
            btn_width, btn_height = 120, 30  # 缩小按钮尺寸
            draw_button("开始游戏", width//2 - btn_width//2, height//2 - 50, 
                      btn_width, btn_height, GRAY, LIGHT_GRAY,
                      lambda: globals().update({'running': True, 'game_start_time': time.time()}))
            draw_button("退出游戏", width//2 - btn_width//2, height//2, 
                      btn_width, btn_height, GRAY, LIGHT_GRAY, pygame.quit)
        else:
            if game_over:
                # 修改游戏结束菜单按钮
                btn_width, btn_height = 120, 30
                draw_button("重新开始", width // 2 - btn_width // 2, height // 2 - 50,
            btn_width, btn_height, GRAY, LIGHT_GRAY,
            lambda: [globals().update({'running': False, 'game_over': False, 'paused': False}), reset_game(),
                     globals().update({'running': True, 'game_start_time': time.time()})])
                draw_button("查看得分", width // 2 - btn_width // 2, height // 2,
            btn_width, btn_height, GRAY, LIGHT_GRAY,
            lambda: print("每局得分情况:", episode_scores))
                draw_button("退出游戏", width // 2 - btn_width // 2, height // 2 + 50,
            btn_width, btn_height, GRAY, LIGHT_GRAY, pygame.quit)
            else:
                # 只保留一个暂停按钮
                btn_width, btn_height = 120, 30
                draw_button("暂停", width - 140, 20, btn_width, btn_height, GRAY, LIGHT_GRAY, lambda: globals().update({'paused': True}))
                if paused:
                    # 修改暂停菜单按钮
                    #btn_width, btn_height = 120, 30
                    draw_button("继续", width - 140, 70, btn_width, btn_height, GRAY, LIGHT_GRAY, 
                              lambda: globals().update({'paused': False}))
                    draw_button("结束游戏", width - 140, 110, btn_width, btn_height, GRAY, LIGHT_GRAY, lambda:[
                        globals().update({'game_over': True}),
                        torch.save(policy_net.state_dict(), model_path),
                        print("手动保存模型成功")
                    ])
                else:
                    # === 全局状态栏参数 ===
                    stats_font = pygame.font.SysFont('simhei', 18)
                    stat_y = 20  # 提升到游戏运行主作用域
                    
                    button_width, button_height = 120, 30
                    
                    # === 顶部状态栏 ===
                    # 移除多余的暂停按钮绘制代码
                    # draw_button("暂停", width - 100, stat_y, button_width, button_height, GRAY, LIGHT_GRAY,
                    #           lambda: globals().update({'paused': True}))
                    
                    line1 = f"关卡:{level}  次数:{len(episode_scores)+1}"
                    line2 = f"得分:{total_score}  时间:{int(game_duration)}s"
                    screen.blit(stats_font.render(line1, True, WHITE), (20, stat_y))
                    screen.blit(stats_font.render(line2, True, WHITE), (20, stat_y + 25))

                    # === 游戏画面区域 ===
                    game_area_top = int(height * 0.15)

                    # 绘制游戏元素
                    paddle.y = height - 40
                    paddle.draw()
                    ball.draw()
                    for brick in bricks:
                        pygame.draw.rect(screen, brick.color, 
                                       (brick.x, brick.y, brick.width, brick.height))

                    if not paused:
                        if game_start_time is not None:
                            game_duration = time.time() - game_start_time
                            # 根据epsilon贪婪策略选择动作
                            if random.random() < epsilon:
                                action = random.randint(0, 2)  # 0:左移, 1:不动, 2:右移
                            else:
                                state = torch.FloatTensor(get_state()).unsqueeze(0).to(device)
                                q_values = policy_net(state)
                                action = torch.argmax(q_values).item()
                            total_attempts += 1
                            # 执行动作
                            move_distance = 5 * width / 800
                            
                            # 如果连续失败超过阈值，增强移动力度
                            if failed_attempts >= max_failed_attempts:
                                move_distance *= 2  # 移动距离加倍
                                failed_attempts = 0  # 重置计数器
                                print("激活增强移动策略")  # 调试信息
                            if action == 0:
                                paddle.move(-move_distance)
                            elif action == 2:
                                paddle.move(move_distance)
                            # 移动球
                            ball.move()
                            # 碰撞检测
                            paddle_hit = check_ball_paddle_collision(ball, paddle)
                            brick_hit = check_ball_brick_collision(ball, bricks)
                            # 计算奖励
                            reward, consecutive_hits, no_action_counter, failed_attempts = calculate_reward(paddle_hit, brick_hit, consecutive_hits, no_action_counter, failed_attempts, max_failed_attempts)
                            total_score += reward
                            total_score = int(total_score)
                            # 球出界
                            single_done = False  # 用于游戏主逻辑的单一布尔值
                            if ball.y > height:
                                single_done = True
                                reward = -10
                            # 所有砖块被击碎
                            if len(bricks) == 0:
                                # 先保存原始速度再重置游戏
                                original_dy = ball.dy
                                original_dx = ball.dx
                                
                                level = min(level + 1, max_level)
                                reward = 100 + (level * 20)
                                reset_game()  # 这里会创建新的 ball 实例
                                
                                # 对新创建的 ball 实例应用速度调整
                                new_dy = abs(original_dy) * 1.1
                                new_dy = max(min(new_dy, 15), 3) * (-1 if original_dy < 0 else 1)
                                
                                new_dx = abs(original_dx) * 1.05
                                new_dx = max(min(new_dx, 8), 3) * (1 if original_dx > 0 else -1)
                                
                                # 直接操作新 ball 实例的属性
                                ball.dy = int(new_dy)  # 确保为整数
                                ball.dx = int(new_dx)  # 确保为整数
                                
                                paddle.width = max(80, paddle.width - 20)  # 每关缩小球拍
                            if single_done:
                                episode_scores.append(total_score)
                                total_score = 0
                                reset_game()

        pygame.display.flip()
        # 限制帧率
        clock.tick(fps)
        # 衰减epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

except Exception as e:
    print(f"发生异常: {e}")
    pygame.quit()