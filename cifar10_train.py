import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
import copy
from torch.multiprocessing import freeze_support
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter  

# 指定使用 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确保使用 GPU 0

# 1. 数据准备
data_dir = './data'

# 确保数据目录存在
if not os.path.exists(data_dir):
    print(f"数据目录 '{data_dir}' 不存在。请创建该目录并放入图像数据。")
    exit()

# 增强的数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 添加随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 添加颜色抖动
    transforms.ToTensor(),
    # 使用CIFAR-10的真实均值和标准差，而不是近似值(0.5,0.5,0.5)
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 加载数据集 - 保持CIFAR-10
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128,  # 增加batch size以加速训练
                         shuffle=True, num_workers=4)  # 增加worker数量

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128,
                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. 模型构建
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  # 加载预训练的 ResNet34 模型

# 修改模型架构，添加dropout和调整全连接层
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),  # 添加dropout防止过拟合
    nn.Linear(num_ftrs, 256),  # 添加中间层
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, len(classes))
)

# 3. 训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 使用标签平滑交叉熵损失
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑

# 优化器使用AdamW并添加权重衰减
optimizer = optim.AdamW([
    # 微调模型，使用不同的学习率
    {'params': [param for name, param in model.named_parameters() if 'fc' not in name], 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
], weight_decay=0.01)  # 添加权重衰减

# 使用余弦退火学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# TensorBoard
writer = SummaryWriter('runs/cifar10_improved')  # 创建 SummaryWriter

def train_model(model, criterion, optimizer, scheduler, num_epochs=80, patience=10):  # 增加训练轮数和耐心值
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 早停机制
    patience_counter = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            dataloader = trainloader if phase == 'train' else testloader  # 根据 phase 选择 dataloader
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        # 梯度裁剪，防止梯度爆炸
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # TensorBoard 写入
            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'New best validation accuracy: {best_acc:.4f}')

            # 早停机制监测
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            elif phase == 'val':
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best val Acc: {best_acc:4f}')
                    model.load_state_dict(best_model_wts)
                    return model

        # 学习率更新
        scheduler.step()
        # 打印当前学习率
        current_lr = scheduler.get_last_lr()
        print(f'Current learning rate: {current_lr}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 4. 运行训练
if __name__ == '__main__':
    # 多进程支持
    freeze_support()
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=80, patience=10)  # 增加训练轮数和耐心值

    # 5. 模型保存
    torch.save(model.state_dict(), 'result_improved.pth')

    # 6. 模型评估
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 计算每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(f'Overall accuracy on the test images: {100 * correct / total:.2f}%')
    
    # 打印每个类别的准确率
    for i in range(10):
        print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

    writer.close()  # 关闭 TensorBoard