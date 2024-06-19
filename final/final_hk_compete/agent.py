import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

from model import Linear_QNet, QTrainer
from model2 import Linear_QNet2, QTrainer2


from helper import plot



MAX_MEMORY = 100000  # 能儲存的最多項目量
BATCH_SIZE = 1000    # 批量處理的量
LR = 0.001   # 學習率        

class Agent:
    def __init__(self):

        self.n_games = 0   # 遊戲次數
        self.epsilon = 0   
        self.gamma = 0.9   # discount rate 記數率 必須是小於1的數

        self.memory1 = deque(maxlen=MAX_MEMORY)  
        self.memory2 = deque(maxlen=MAX_MEMORY)  

        self.model1 = Linear_QNet(11, 256, 3)     
        self.model2 = Linear_QNet2(11, 256, 256, 3)  

        self.trainer1 = QTrainer(self.model1, lr=LR, gamma=self.gamma)  
        self.trainer2 = QTrainer2(self.model2, lr=LR, gamma=self.gamma)  

        self.game = SnakeGameAI()   

    # 獲取遊戲的各項狀態
    def get_state(self, snake_num):

        game = self.game

        if snake_num == 1:

            head = game.snake1[0]
        else:

            head = game.snake2[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.y, head.y + 20)

        dir_l = (game.direction1 == Direction.LEFT if snake_num == 1 else game.direction2 == Direction.LEFT)
        dir_r = (game.direction1 == Direction.RIGHT if snake_num == 1 else game.direction2 == Direction.RIGHT)
        dir_u = (game.direction1 == Direction.UP if snake_num == 1 else game.direction2 == Direction.UP)
        dir_d = (game.direction1 == Direction.DOWN if snake_num == 1 else game.direction2 == Direction.DOWN)

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y   # food down
        ]

        return np.array(state, dtype=int)
    
    # 儲存各項需要的參數
    def remember(self, snake_num, state, action, reward, next_state, done):

        if snake_num == 1:
            self.memory1.append((state, action, reward, next_state, done))
        else:
            self.memory2.append((state, action, reward, next_state, done))

    # 立即利用新獲得的經驗來更新模型參數
    def train_long_memory(self, snake_num):

        if snake_num == 1:
            memory = self.memory1
            trainer = self.trainer1
        else:
            memory = self.memory2
            trainer = self.trainer2

        if len(memory) > BATCH_SIZE:
            mini_sample = random.sample(memory, BATCH_SIZE)
        else:
            mini_sample = memory

        for state, action, reward, next_state, done in mini_sample:
            trainer.train_step(state, action, reward, next_state, done)

    # 讓 ai 跑遊戲的這個動作
    def train_short_memory(self, snake_num, state, action, reward, next_state, done):
        
        if snake_num == 1:
            self.trainer1.train_step(state, action, reward, next_state, done)
        else:
            self.trainer2.train_step(state, action, reward, next_state, done)

    # 讓 ai 跑遊戲的這個動作
    def get_action(self, snake_num, state):

        if snake_num == 1:
            epsilon = self.epsilon
            model = self.model1
        else:
            epsilon = self.epsilon
            model = self.model2

        epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train(max_games=1000):

    plot_scores1 = []
    plot_mean_scores1 = []

    plot_scores2 = []
    plot_mean_scores2 = []

    total_score1 = 0
    total_score2 = 0

    record1 = 0
    record2 = 0

    agent = Agent()

    while agent.n_games < max_games: 
        # 得到當前狀態 
        state1 = agent.get_state(1)
        state2 = agent.get_state(2)
        # 選擇動作
        final_move1 = agent.get_action(1, state1)
        final_move2 = agent.get_action(2, state2)
        # 執行動作
        reward1, reward2, done1, done2, score1, score2 = agent.game.play_step(final_move1, final_move2)
        # 短期記憶訓練
        next_state1 = agent.get_state(1)
        next_state2 = agent.get_state(2)
        # 記住經驗
        agent.remember(1, state1, final_move1, reward1, next_state1, done1)
        agent.remember(2, state2, final_move2, reward2, next_state2, done2)

        if done1 or done2:

            agent.game.reset() 

            agent.n_games += 1
            agent.train_long_memory(1)
            agent.train_long_memory(2)


            # 如果當前得分高於記錄，保存模型
            if score1 > record1:
                record1 = score1
                agent.model1.save()

            if score2 > record2:
                record2 = score2
                agent.model2.save()

            print(f'Game {agent.n_games}\n'    
                  f'Score1: {score1}    Record1: {record1}\n'  
                  f'Score2: {score2}    Record2: {record2}\n' 
                  f'--------------------------------------'
                  )

            plot_scores1.append(score1)
            total_score1 += score1
            mean_score1 = total_score1 / agent.n_games
            plot_mean_scores1.append(mean_score1)
            

            plot_scores2.append(score2)
            total_score2 += score2
            mean_score2 = total_score2 / agent.n_games
            plot_mean_scores2.append(mean_score2)

            plot(plot_scores1, plot_mean_scores1,plot_scores2, plot_mean_scores2)
            

if __name__ == '__main__':
    train()

    




