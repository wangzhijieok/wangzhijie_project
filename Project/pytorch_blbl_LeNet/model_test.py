import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import LeNet


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True)
    return test_dataloader


def test_model_process(test_dataloader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 初始化参数
    test_corrects = 0
    test_num = 0

    # 模型在推理的过程中 只进行前向传播，不反向传播即不更新参数
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 特征数据和标签都放入gpu中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()
            output = model(test_data_x)
            # dim = 1 指沿着第1维度方向（列）寻找最大值  张量维度tensorflow 如（0）dim=0  [1,2] dim=1
            pre_label = torch.argmax(output, dim=1)
            # 累计预测正确的样本
            test_corrects += (pre_label == test_data_y).float().sum()
            test_num += test_data_x.size(0) # 此处相当于 +1

    test_accuracy = test_corrects / test_num
    print("Test Accuracy: {:.4f}".format(test_accuracy))


def test_predict(test_dataloader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            result = pre_label.item()
            label = b_y.item()
            print("Predict: {}, Label: {}, 预测正确性：{}".format(result, label, result==label))


if __name__ == '__main__':
    # 创建一个与保存参数时结构完全相同的模型
    model = LeNet()
    # 加载训练好的权重
    model.load_state_dict(torch.load('./best_model.pth'))
    # 加载数据
    test_dataloader = test_data_process()
    #test_model_process(test_dataloader, model)
    test_predict(test_dataloader, model)

