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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# 启用 cuDNN 的自动调整算法以加速卷积操作
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # 使用非确定性算法以提高性能
# 1. 数据准备函数添加异常处理
def prepare_data():
    try:
        data_dir = './data/Medicine'

        # 确保数据目录存在
        if not os.path.exists(data_dir):
            print(f"数据目录 '{data_dir}' 不存在。请创建该目录并放入中医药图像数据。")
            exit()

        # 检查训练和测试目录
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print(f"训练或测试目录不存在。请确保在 '{data_dir}' 下有 'train' 和 'test' 文件夹。")
            exit()

        # 获取类别列表
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        num_classes = len(classes)
        
        # 简化数据预处理以加速训练
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 使用 ImageFolder 加载数据
        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)

        # 降低批量大小以减少内存使用
        trainloader = DataLoader(trainset, batch_size=96, shuffle=True, 
                                num_workers=8, pin_memory=True)
        testloader = DataLoader(testset, batch_size=96, shuffle=False, 
                               num_workers=8, pin_memory=True)
        
        return trainloader, testloader, classes, num_classes
    except Exception as e:
        print(f"数据准备阶段发生错误: {str(e)}")
        exit(1)

# 2. 修改训练函数添加异常处理
def train_model(model, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    start_time = time.time()
    epoch_times = []  # 用于记录每个epoch的时间

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 早停机制
    patience_counter = 0
    best_loss = float('inf')

    # 打印训练开始信息
    start_time_str = time.strftime("%H:%M:%S")
    print(f"\n=== 训练开始于 {start_time_str} ===")
    print(f"使用MobileNetV3 Small模型 + 标准训练 + 批量大小96")
    print(f"共 {num_epochs} 轮训练，每轮有 {len(trainloader)} 个训练批次和 {len(testloader)} 个验证批次")

    for epoch in range(num_epochs):
        try:
            epoch_start = time.time()
            
            print(f"\n[Epoch {epoch+1}/{num_epochs}]" + "=" * 40)
            
            # 每个 epoch 有训练和验证阶段
            for phase in ['train', 'val']:
                try:
                    if phase == 'train':
                        model.train()  # 设置模型为训练模式
                        dataloader = trainloader
                        phase_desc = "训练阶段"
                    else:
                        model.eval()   # 设置模型为评估模式
                        dataloader = testloader
                        phase_desc = "验证阶段"

                    print(f"\n[{phase_desc}]")
                    
                    running_loss = 0.0
                    running_corrects = 0
                    batch_count = 0
                    total_batches = len(dataloader)
                    
                    # 在每个阶段开始时清理缓存释放内存
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"清理 GPU 缓存时出错: {str(e)}")
                        
                    # 迭代数据
                    for i, (inputs, labels) in enumerate(dataloader):
                        try:
                            batch_start = time.time()
                            inputs = inputs.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)
                            
                            # 梯度清零
                            optimizer.zero_grad(set_to_none=True)

                            # 前向传播 - 标准方式（无混合精度）
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs = model(inputs)
                                _, preds = torch.max(outputs, 1)
                                loss = criterion(outputs, labels)

                                # 反向传播和优化 - 标准方式
                                if phase == 'train':
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    optimizer.step()
                            
                            # 统计
                            batch_size = inputs.size(0)
                            batch_count += 1
                            running_loss += loss.item() * batch_size
                            running_corrects += torch.sum(preds == labels.data).item()
                            
                            # 计算当前批次的损失和准确率
                            try:
                                current_loss = running_loss / (batch_count * batch_size)
                                current_acc = 100 * running_corrects / (batch_count * batch_size)
                            except ZeroDivisionError:
                                print("\n警告: 计算损失和准确率时发生除零错误")
                                current_loss = 0.0
                                current_acc = 0.0
                            
                            # 每处理一定数量的批次更新进度条
                            if i % 10 == 0 or i == total_batches - 1:
                                try:
                                    # 计算批处理速度和预估剩余时间
                                    batch_time = time.time() - batch_start
                                    samples_per_sec = batch_size / max(batch_time, 1e-5)  # 避免除零
                                    
                                    # 格式化输出
                                    print_str = f"\r批次: [{i+1}/{total_batches}] "
                                    print_str += f"损失: {current_loss:.4f} 准确率: {current_acc:.2f}% "
                                    if i > 0:  # 跳过第一个批次，因为时间可能不准确
                                        print_str += f"速度: {samples_per_sec:.1f}样本/秒"
                                    
                                    print(print_str, end="")
                                except Exception as e:
                                    print(f"\n警告: 显示进度时出错: {str(e)}")
                            
                            # 在 TensorBoard 中记录每个批次的损失 (降低记录频率以提高性能)
                            try:
                                if phase == 'train' and i % 100 == 0:  # 每100个批次记录一次
                                    batch_idx = epoch * len(dataloader) + i
                                    writer.add_scalar('Batch/loss', loss.item(), batch_idx)
                            except Exception as e:
                                print(f"\n警告: TensorBoard 写入时出错: {str(e)}")
                                
                        except Exception as e:
                            print(f"\n警告: 处理批次 {i+1}/{total_batches} 时出错: {str(e)}")
                            print("继续处理下一批次...")
                            continue
                            
                        finally:
                            # 主动释放不需要的张量
                            try:
                                del inputs, labels, outputs, preds
                                if phase == 'train' and 'loss' in locals():
                                    del loss
                            except Exception:
                                pass  # 忽略清理错误
                    
                    # 完成一个阶段后换行和计算总指标
                    print()  # 确保进度条后换行
                    
                    try:
                        # 计算阶段总体指标
                        dataset_size = len(dataloader.dataset)
                        if dataset_size > 0 and batch_count > 0:
                            epoch_loss = running_loss / dataset_size
                            epoch_acc = running_corrects / dataset_size
                        else:
                            print("警告: 数据集大小为0或批次计数为0")
                            epoch_loss = 0.0
                            epoch_acc = 0.0
                        
                        # 打印阶段结果
                        print(f"{phase_desc} 完成 - 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.4f}")

                        # TensorBoard 写入
                        writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
                        writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)

                        # 如果是最佳模型，保存权重
                        if phase == 'val' and epoch_acc > best_acc:
                            best_acc = epoch_acc
                            best_model_wts = copy.deepcopy(model.state_dict())
                            print(f'✅ 新的最佳验证准确率: {best_acc:.4f}')
                            # 保存最佳模型
                            try:
                                model_save_path = f'medicine_best_model.pth'
                                torch.save(model.state_dict(), model_save_path)
                                print(f'💾 模型已保存到: {model_save_path}')
                            except Exception as e:
                                print(f"保存模型时出错: {str(e)}")

                        # 早停机制监测
                        if phase == 'val':
                            if epoch_loss < best_loss:
                                best_loss = epoch_loss
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                print(f"⚠️ 连续 {patience_counter}/{patience} 轮未改善")
                                if patience_counter >= patience:
                                    print("\n🛑 触发早停机制")
                                    time_elapsed = time.time() - start_time
                                    print(f'最佳验证准确率: {best_acc:.4f}')
                                    model.load_state_dict(best_model_wts)
                                    return model
                    except Exception as e:
                        print(f"计算阶段指标时出错: {str(e)}")
                        # 继续训练，不中断
                
                except Exception as e:
                    print(f"处理阶段 {phase} 时发生错误: {str(e)}")
                    print("尝试继续下一个阶段...")
                    continue

            try:
                # 计算并存储本轮epoch的时间
                epoch_duration = time.time() - epoch_start
                epoch_times.append(epoch_duration)
                
                # 学习率更新
                if scheduler:
                    try:
                        scheduler.step()
                        # 记录学习率
                        current_lr = scheduler.get_last_lr()
                        writer.add_scalar('Learning_rate', current_lr[0], epoch)
                        print(f"📊 当前学习率: {current_lr[0]:.6f}")
                    except Exception as e:
                        print(f"更新学习率时出错: {str(e)}")
                
                # 显示本轮用时
                print(f"⏱️ 本轮用时: {epoch_duration:.1f}秒")
                
                # 进度总结
                overall_progress = (epoch + 1) / num_epochs * 100
                print(f"总进度: {overall_progress:.1f}% [{epoch+1}/{num_epochs}]")
            except Exception as e:
                print(f"更新轮总结时出错: {str(e)}")
                
        except Exception as e:
            print(f"轮 {epoch+1} 训练过程中发生错误: {str(e)}")
            print("尝试继续下一轮训练...")
            continue

    try:
        # 训练结束，显示总结信息
        time_elapsed = time.time() - start_time
        end_time_str = time.strftime("%H:%M:%S")
        print(f"\n=== 训练结束于 {end_time_str} ===")
        print(f"总用时: {time_elapsed:.1f}秒")
        print(f'最佳验证准确率: {best_acc:.4f}')
        print("-" * 60)

        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)
    except Exception as e:
        print(f"训练结束后处理时出错: {str(e)}")
        
    return model

# 3. 在测试和评估阶段添加异常处理
if __name__ == '__main__':
    # 多进程支持
    freeze_support()
    
    try:
        # 调用数据准备函数，只执行一次
        trainloader, testloader, classes, num_classes = prepare_data()
        
        # 使用 MobileNetV3 Small 
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
        # 3. 训练模型
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 使用标签平滑交叉熵损失
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 优化器使用AdamW并添加权重衰减，根据新的batch size调整学习率
        optimizer = optim.AdamW([
            {'params': [param for name, param in model.named_parameters() 
                       if 'classifier' not in name], 'lr': 0.0002},
            {'params': model.classifier.parameters(), 'lr': 0.002}
        ], weight_decay=0.01)
        
        # 使用余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # TensorBoard
        writer = SummaryWriter(f'runs/medicine_classification')
        
        print("准备训练数据和模型...")
        
        # 训练模型，降低轮数加快训练
        print("开始训练模型...")
        model = train_model(model, criterion, optimizer, scheduler, num_epochs=18, patience=3)
        
        # 5. 模型保存
        try:
            final_model_path = f'medicine_final_model.pth'
            torch.save(model.state_dict(), final_model_path)
            print(f"\n💾 最终模型已保存到: {final_model_path}")
        except Exception as e:
            print(f"保存最终模型时出错: {str(e)}")
        
        print("\n开始评估模型...")
        # 6. 模型评估 - 添加异常处理
        try:
            correct = 0
            total = 0
            class_correct = list(0. for i in range(num_classes))
            class_total = list(0. for i in range(num_classes))
            
            model.eval()
            with torch.no_grad():  # 只保留no_grad
                total_batches = len(testloader)
                print(f"评估中... 共 {total_batches} 批次")
                
                for i, data in enumerate(testloader):
                    try:
                        images, labels = data
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        # 计算每个类别的准确率
                        c = (predicted == labels).squeeze()
                        for j in range(labels.size(0)):
                            try:
                                label = labels[j]
                                class_correct[label] += c[j].item()
                                class_total[label] += 1
                            except Exception as e:
                                print(f"计算类别准确率时出错: {str(e)}")
                        
                        # 更新进度条
                        try:
                            progress = (i + 1) / total_batches
                            bar_length = 30
                            filled_length = int(bar_length * progress)
                            bar = '█' * filled_length + '▒' * (bar_length - filled_length)
                            current_acc = 100 * correct / total if total > 0 else 0
                            
                            print(f"\r评估进度: |{bar}| {progress*100:.1f}% - 批次 {i+1}/{total_batches} - 当前准确率: {current_acc:.2f}%", end="")
                        except Exception as e:
                            print(f"\n更新进度条时出错: {str(e)}")
                        
                    except Exception as e:
                        print(f"\n处理评估批次 {i+1}/{total_batches} 时出错: {str(e)}")
                        continue
                    finally:
                        # 释放内存
                        try:
                            del images, labels, outputs, predicted
                        except:
                            pass
                
                # 完成后换行
                print()
            
            try:
                # 计算总体准确率
                overall_accuracy = 100 * correct / total if total > 0 else 0
                print(f'\n🎯 测试集上的总体准确率: {overall_accuracy:.2f}%')
                
                # 打印每个类别的准确率
                print('\n📊 各类别准确率:')
                print('-' * 60)
                print(f"{'类别':<30} {'准确率':>10} {'样本数':>8}")
                print('-' * 60)
                
                # 按准确率排序显示
                try:
                    class_accuracies = [(i, 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0) 
                                        for i in range(num_classes)]
                    class_accuracies.sort(key=lambda x: x[1], reverse=True)
                except Exception as e:
                    print(f"排序类别准确率时出错: {str(e)}")
                    class_accuracies = [(i, 0) for i in range(num_classes)]
                
                # 只显示前10个最佳和后10个最差的类别，避免输出过多
                try:
                    print("【最佳10个类别】")
                    for idx, acc in class_accuracies[:10]:
                        if class_total[idx] > 0:
                            print(f"{classes[idx]:<30} {acc:>8.2f}% {class_total[idx]:>8}")
                            # 将各类别准确率添加到TensorBoard
                            writer.add_scalar(f'Test_Accuracy/{classes[idx]}', acc, 0)
                    
                    print("\n【最差10个类别】")
                    for idx, acc in class_accuracies[-10:]:
                        if class_total[idx] > 0:
                            print(f"{classes[idx]:<30} {acc:>8.2f}% {class_total[idx]:>8}")
                            writer.add_scalar(f'Test_Accuracy/{classes[idx]}', acc, 0)
                except Exception as e:
                    print(f"显示类别准确率时出错: {str(e)}")
                
                print('-' * 60)
                
                # 添加总体准确率到TensorBoard
                writer.add_scalar('Test_Accuracy/Overall', overall_accuracy, 0)
                writer.add_text('Test_Results', f'总体准确率: {overall_accuracy:.2f}%', 0)
            except Exception as e:
                print(f"计算和显示准确率统计时出错: {str(e)}")
        
        except Exception as e:
            print(f"模型评估过程中发生错误: {str(e)}")
        
        finally:
            try:
                writer.close()  # 确保关闭 TensorBoard
            except:
                pass
            
            print(f"\n✅ 训练和评估完成!")
    
    except Exception as e:
        print(f"程序执行过程中发生错误: {str(e)}")