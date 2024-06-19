import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet2(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(p=0.5)  

        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):

        x = F.relu(self.dropout1(self.linear1(x)))
        x = F.relu(self.dropout2(self.linear2(x)))
        x = self.linear3(x)

        return x
    
    # 儲存 model
    def save(self, file_name='model.pth'):
        model_name = './model'

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        file_name = os.path.join(model_name, file_name)

        torch.save(self.state_dict(), file_name)


# 提高穩定性 ，否則難以收斂

class QTrainer2:

    def __init__(self, model, lr, gamma):

        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # 初始化 優化器
        self.criterion = nn.MSELoss() # loss 值


    # 這裡就是把資料都變成 tensor 也就是張量
    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # 如果 state 是一維的話 ， 則擴展維度以適應批處理格式 unsqueeze函式就是擴展為批處理維度
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # 轉為原組


        pred = self.model(state) # 當前狀態的 Q 值

        target = pred.clone() # 計算目標 Q 值


        # 更新目標 Q 值 ，Q-learning 的公式
        for idx in range(len(done)):
            Q_new = reward[idx]

            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    

        self.optimizer.zero_grad() # 在計算損失之前，使用 self.optimizer.zero_grad() 清除先前的梯度
        loss = self.criterion(target, pred)

        # 反傳遞
        loss.backward() # 計算損失對模型參數的梯度

        self.optimizer.step() # 使用 self.optimizer.step() 更新模型參數



