import numpy as np
import torch
import torch.nn as nn
import math
import time
import os
import copy
import random

# ịnput_channels = 8:
#     + 0: cỏ, 1: tường, 2: hộp, 3: vật phẩm tăng bán kính, 4: vật phẩm tăng số lượng
#     + 5: vị trí của agent
#     + 6: vị trí của đối thủ
#     + 7: Danh sách chứa các bom hiện có (timer đếm ngược đến 7)

#DQN nghĩa là dựa vào mạng neuron, dựa vào 8 giá trị đầu vào và trả về được
class DQN(nn.Module):
    def __init__(self, input_channels=8, num_actions=6):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 13 * 13, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)