import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 增加隱藏層神經元數量
        self.fc2 = nn.Linear(128, 64)     # 增加隱藏層
        self.fc3 = nn.Linear(64, 10)      # 輸出層
        self.dropout = nn.Dropout(0.5)    # dropout 層

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))    # 使用 ReLU 激活函數
        x = self.dropout(x)        # 添加 dropout 層
        x = F.relu(self.fc2(x))    # 第二個隱藏層使用 ReLU
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)