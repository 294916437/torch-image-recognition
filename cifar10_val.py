import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# 1. 加载模型架构
model = models.resnet34(weights=None)  # 创建 ResNet34 模型实例

# 2. 修改 fc 层，与训练时保持一致
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),  # 添加dropout防止过拟合
    nn.Linear(num_ftrs, 256),  # 添加中间层
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)  # CIFAR-10有10个类别
)

# 3. 加载训练好的模型状态字典
model.load_state_dict(torch.load('result_improved.pth'))  # 加载改进后的模型权重

# 4. 将模型设置为评估模式
model.eval()

# 5. 将模型移动到正确的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 6. 数据预处理 - 与训练时保持一致
transform = transforms.Compose([
    transforms.Resize(32),  # CIFAR-10是32x32的图像
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10的均值和标准差
])

# 7. 加载图像
image = Image.open('./val-data/cifar10/cat_1.jpg')  # 替换为您的图像路径
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)  # 创建一个 batch

# 8. 将输入移动到正确的设备
input_batch = input_batch.to(device)

# 9. 进行推理
with torch.no_grad():
    output = model(input_batch)

# 10. 获取预测结果
_, predicted = torch.max(output.data, 1)

# 11. 打印预测结果
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 类别列表
predicted_class = classes[predicted.item()]  # 获取预测的类别名称
print(f'Predicted class: {predicted_class}')  # 打印预测的类别名称

# 12. 打印置信度分数
probabilities = torch.nn.functional.softmax(output, dim=1)[0]
for i, prob in enumerate(probabilities):
    print(f'{classes[i]}: {prob.item()*100:.2f}%')