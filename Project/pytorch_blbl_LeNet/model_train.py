import time
import pandas as pd
import torch
import copy
import pandas
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet


# 每一轮训练 wb 验证loss


# 数据加载过程
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transforms.ToTensor())
    # FashionMNIST本来就是28x28 不用再resize
    # transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()])

    # 划分训练集   20%验证
    train_data, val_data = Data.random_split(train_data, [round(len(train_data) * 0.8), round(len(train_data) * 0.2)])
    train_dataloader = Data.DataLoader(train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8)
    val_dataloader = Data.DataLoader(val_data,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=8)
    return train_dataloader, val_dataloader


# 模型训练过程
def train_model_process(train_dataloader, val_dataloader, model, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    # 交叉熵损失函数（分类问题）   已经包含了softmax
    # log_softmax = F.log_softmax(output, dim=1)
    # loss = F.nll_loss(log_softmax, target)
    criterion = nn.CrossEntropyLoss()
    # 优化器 本质是梯度下降法，方便后面模型训练的参数更新。 比如SGD Adam。。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    model = model.to(device)

    # 复制当前模型参数，在训练过程保存
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_losses = []
    # 测试集损失列表
    val_losses = []
    # 训练集准确度列表
    train_accuracies = []
    # 测试集准确度列表
    val_accuracies = []
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        train_loss = 0.0
        train_correct = 0

        val_loss = 0.0
        val_correct = 0

        # 样本数量
        train_num = 0
        val_num = 0

        # 从所有数据中每个批次取数据训练计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 放置参数
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置训练模式
            model.train()

            # 前向传播过程，
            output = model(b_x)

            # 查找每一行最大值对应行标  也就是分类问题中最后概率最大的那个序号
            # argmax 得到每个样本预测的类别索引
            pred = output.argmax(dim=1, keepdim=True)
            # 输出和标签计算损失
            loss = criterion(output, b_y)

            # 梯度归零 防止把之前的梯度累加
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据网络模型反向传播的梯度信息来更新网络的参数，降低loss函数计算值
            optimizer.step()

            # 计算样本级别的总loss，一边计算平均loss
            # pytorch默认loss是平均值  bach样本数量
            # 每个批次bach大小可能不同，分批累加得到一轮所有批次的总损失函数，除以总样本数就是平均损失
            train_loss += loss.item() * b_x.size(0)
            train_correct += pred.eq(b_y.view_as(pred)).sum().item()
            train_num += b_x.size()[0]
        # //for step

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pred = output.argmax(dim=1, keepdim=True)
            val_loss += criterion(output, b_y).item() * b_x.size()[0]
            val_correct += pred.eq(b_y.view_as(pred)).sum().item()

            val_num += b_x.size()[0]
        # //for step

        train_accuracies.append(train_correct / train_num)
        train_losses.append(train_loss / train_num)
        val_accuracies.append(val_correct / val_num)
        val_losses.append(val_loss / val_num)
        # 输出最后一个
        print('train loss：{:.4f}， train acc：{:.4f}'.format(train_losses[-1], train_accuracies[-1]))
        print('val loss：{:.4f}， val acc：{:.4f}'.format(val_losses[-1], val_accuracies[-1]))

        # 保存当前最优准确度和参数
        if (val_accuracies[-1] > best_acc):
            best_acc = val_accuracies[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 训练耗时
        time_elapsed = time.time() - since
        print('训练耗时{:.0f}m {:.0f}s'.format(time_elapsed / 60, time_elapsed % 60))
        print('-' * 10)
    #  //for epoch

    # 把最优参数加载到模型   保存最佳权重
    # model.load_state_dict(best_model_wts)
    # torch.save(model.load_state_dict(best_model_wts), './best_model.pth')
    torch.save(best_model_wts, './best_model.pth')

    train_data_process = pd.DataFrame(data={'epoch': range(num_epochs),
                                            'train_loss': train_losses,
                                            'val_loss': val_losses,
                                            'train_accuracy': train_accuracies,
                                            'val_accuracy': val_accuracies, })

    return train_data_process


def matplot_acc_loss(train_data_process):
    plt.figure(figsize=(12, 4))
    # 一行两列的第一张图
    plt.subplot(1, 2, 1)
    plt.plot(train_data_process['epoch'], train_data_process['train_loss'], label='train_loss',
             marker='o', markersize=5)
    plt.plot(train_data_process['epoch'], train_data_process['val_loss'], label='val_loss',
             marker='o', markersize=5)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_data_process['epoch'], train_data_process['train_accuracy'], label='train_accuracy',
             marker='o', markersize=5)
    plt.plot(train_data_process['epoch'], train_data_process['val_accuracy'], label='val_accuracy',
             marker='o', markersize=5)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    lenet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_data_process = train_model_process(train_dataloader, val_dataloader, lenet, num_epochs=2)
    matplot_acc_loss(train_data_process)
