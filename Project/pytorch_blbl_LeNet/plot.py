from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data

# 下载数据集
train_data = FashionMNIST(root='./data',
                          train=True,
                          download=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))
# 数据随机打包
train_loader = data.DataLoader(train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)