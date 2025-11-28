import torch
import torchsummary
from matplotlib.pyplot import summer
from torch import nn
from torchsummary import summary
from torchinfo import summary


class LeNet(nn.Module): #继承module类
    #初始化
    def __init__(self):
        super(LeNet, self).__init__()
        #卷积层
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        #平展
        self.flatten = nn.Flatten()
        #全连接
        self.f5 = nn.Linear(in_features=16*5*5, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)
    #前向传播
    def forward(self, x):
        x = self.sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #实例化
    model = LeNet().to(device)

    print(summary(model, input_size=(1, 1, 28, 28)))  # 注意输入维度格式