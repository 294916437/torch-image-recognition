import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import os
import sys
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='中药图像分类模型验证')
    parser.add_argument('--image', type=str, help='要分类的图像路径')
    parser.add_argument('--model', type=str, default='medicine_best_model.pth',
                        help='模型权重文件路径，默认为medicine_best_model.pth')
    parser.add_argument('--top_k', type=int, default=5,
                        help='显示前k个预测结果，默认为5')
    parser.add_argument('--interactive', action='store_true',
                        help='进入交互模式，可以连续测试多张图片')
    return parser.parse_args()

def get_class_names():
    """从训练数据文件夹获取类别名称"""
    try:
        data_dir = './data/Medicine'
        train_dir = os.path.join(data_dir, 'train')
        
        if not os.path.exists(train_dir):
            print(f"警告：目录 {train_dir} 不存在")
            return None
            
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        print(f"加载了 {len(classes)} 个类别")
        return classes
    except Exception as e:
        print(f"获取类别名称时出错: {e}")
        return None

def load_model(model_path, num_classes):
    """加载训练好的模型"""
    try:
        # 检查CUDA可用性
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建MobileNetV3 Small模型 (与训练时使用的相同架构)
        model = models.mobilenet_v3_small(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
        # 加载权重
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 '{model_path}' 不存在")
            sys.exit(1)
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()  # 设置为评估模式
        
        print(f"成功加载模型: {model_path}")
        return model, device
    except Exception as e:
        print(f"加载模型时出错: {e}")
        sys.exit(1)

def preprocess_image(image_path):
    """预处理单张图像"""
    try:
        # 确保图像路径存在
        if not os.path.exists(image_path):
            print(f"错误：图像文件 '{image_path}' 不存在")
            return None, None
            
        # 与训练时使用相同的预处理
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()  # 保存原始图像用于显示
        
        # 应用变换
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # 添加批次维度
        
        return input_batch, original_image
    except Exception as e:
        print(f"预处理图像时出错: {e}")
        return None, None

def predict_image(model, image_tensor, device, class_names, top_k=5):
    """使用模型预测图像"""
    try:
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # 获取前k个预测结果
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # 转换为Python列表
        probs = top_probs.cpu().numpy()
        indices = top_indices.cpu().numpy()
        
        # 获取类别名称
        if class_names is not None:
            classes = [class_names[idx] for idx in indices]
        else:
            classes = [f"类别{idx}" for idx in indices]
            
        return classes, probs
    except Exception as e:
        print(f"预测时出错: {e}")
        return [], []


def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 获取类别名称
        class_names = get_class_names()
        num_classes = len(class_names) if class_names else 160  # 默认假设有160个类别
        
        # 加载模型
        model, device = load_model(args.model, num_classes)
        
        # 如果没有提供图像路径或者使用交互式模式
        if not args.image or args.interactive:
            print("\n==== 中药图像识别交互模式 ====")
            print("请输入图像路径进行识别，输入'exit'或'q'退出")
            
            while True:
                image_path = input("\n请输入图像路径: ")
                if image_path.lower() in ['exit', 'quit', 'q']:
                    print("退出程序")
                    break
                
                # 预处理图像
                input_tensor, original_image = preprocess_image(image_path)
                if input_tensor is None:
                    continue
                
                # 进行预测
                classes, probs = predict_image(model, input_tensor, device, class_names, args.top_k)
                if not classes:
                    continue
                
                # 打印预测结果
                print("\n预测结果:")
                print("-" * 60)
                for i, (cls, prob) in enumerate(zip(classes, probs)):
                    print(f"{i+1}. {cls:<30} - 置信度: {prob*100:.2f}%")
                print("-" * 60)
        else:
            # 单图模式
            input_tensor, original_image = preprocess_image(args.image)
            if input_tensor is None:
                sys.exit(1)
                
            # 进行预测
            classes, probs = predict_image(model, input_tensor, device, class_names, args.top_k)
            if not classes:
                sys.exit(1)
                
            # 打印预测结果
            print("\n预测结果:")
            print("-" * 60)
            for i, (cls, prob) in enumerate(zip(classes, probs)):
                print(f"{i+1}. {cls:<30} - 置信度: {prob*100:.2f}%")
            print("-" * 60)
            
                
    except KeyboardInterrupt:
        print("\n程序已被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

# 命令行运行示例如下
# python medicine_val.py --image ./val-data/medicine/gancao_0.jpg --model medicine_final_model.pth
if __name__ == "__main__":
    main()