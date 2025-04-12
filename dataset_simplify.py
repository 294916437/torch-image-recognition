import os
import shutil
import random
import argparse  # 添加argparse库

def simplify_dataset(source_dir, target_dir, samples_per_class=10):
    """
    简化数据集，每个类别只保留指定数量的样本
    
    参数:
        source_dir (str): 源数据集目录
        target_dir (str): 目标数据集目录
        samples_per_class (int): 每个类别保留的样本数
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        # 计算相对路径，用于在目标目录中创建相同的结构
        rel_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, rel_path) if rel_path != '.' else target_dir
        
        # 如果当前目录是类别目录（包含图片），则只保留指定数量的图片
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            print(f"处理类别目录: {rel_path}")
            
            # 创建目标目录
            os.makedirs(target_path, exist_ok=True)
            
            # 按顺序选择前N个
            selected_files = image_files[:samples_per_class]
            
            # 复制文件
            for file in selected_files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_path, file)
                shutil.copy2(src_file, dst_file)
                
            print(f"  已选择 {len(selected_files)}/{len(image_files)} 个文件")
        
        # 为非图片目录创建相应的目录结构
        elif not image_files and not files:  # 空目录
            os.makedirs(target_path, exist_ok=True)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="简化数据集，每个类别只保留指定数量的样本")
    parser.add_argument("source_dir", help="源数据集目录")
    parser.add_argument("target_dir", help="目标数据集目录")
    parser.add_argument("-n", "--num", type=int, default=10, help="每个类别保留的样本数 (默认: 10)")
    
    args = parser.parse_args()
    
    # 使用命令行参数
    source_dataset = args.source_dir
    target_dataset = args.target_dir
    samples_per_class = args.num
    
    print(f"开始简化数据集，每个类别保留{samples_per_class}个样本...")
    print(f"源数据集: {source_dataset}")
    print(f"目标数据集: {target_dataset}")
    
    simplify_dataset(source_dataset, target_dataset, samples_per_class)
    print(f"数据集简化完成! 结果保存在: {target_dataset}")

if __name__ == "__main__":
    main()