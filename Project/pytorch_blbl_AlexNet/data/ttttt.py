import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# 1. 载入预训练模型
vgg = models.vgg16(pretrained=True).features.eval()

# 2. 加载并预处理猫图
img_path = "cat.jpg"  # 这里替换成你的图片路径
img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
x = transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

# 3. 可视化若干层的特征
layer_ids = [0, 5, 10, 17, 24]  # 对应 VGG16 的 Conv1~Conv5
layer_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5']

plt.figure(figsize=(12, 8))

with torch.no_grad():
    for idx, (lid, lname) in enumerate(zip(layer_ids, layer_names)):
        feat = vgg[:lid+1](x)
        fmap = feat[0, 0].cpu()  # 只展示第一个通道
        plt.subplot(2, 3, idx+1)
        plt.imshow(fmap, cmap='gray')
        plt.title(f"{lname} (Layer {lid+1})")
        plt.axis('off')

# 展示原图
plt.subplot(2, 3, 6)
plt.imshow(Image.open(img_path))
plt.title("Input Image")
plt.axis('off')

plt.tight_layout()
plt.show()
