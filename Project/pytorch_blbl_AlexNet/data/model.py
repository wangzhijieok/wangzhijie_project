import torch
from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        # 调用父类 init 方法
        super(AlexNet, self).__init__()
        # 先定义激活函数
        self.relu = nn.ReLU()
        # 开搭建网络层
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flat = nn.Flatten()

        self.f1 = nn.Linear(in_features=6 * 6 * 256, out_features=4096)
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        self.f3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = self.relu(self.c5(x))
        x = self.relu(self.c6(x))
        x = self.relu(self.c7(x))
        x = self.s8(x)

        x = self.flat(x)
        x = self.relu(self.f1(x))
        # 随机使部分神经元失活
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.relu(self.f2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.relu(self.f3(x))
        return x