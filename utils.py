"""
工具函数模块
"""
import os
import numpy as np
import torch
from config import Config


def set_seed(seed=Config.RANDOM_SEED):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_directory():
    """创建输出目录"""
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    print(f"输出目录: {Config.OUTPUT_PATH}")


def save_training_losses(train_losses, val_losses, output_path):
    """保存训练损失"""
    loss_file = os.path.join(output_path, 'training_losses.txt')
    with open(loss_file, 'w', encoding='utf-8') as f:
        f.write('Epoch\tTrain_Loss\tVal_Loss\n')
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f'{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n')
    print(f'训练损失已保存至 {loss_file}')


def print_data_summary(train_data, val_data, test_data):
    """打印数据摘要"""
    print("\n" + "=" * 60)
    print("数据摘要")
    print("=" * 60)
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"测试集样本数: {len(test_data)}")

    # 统计测站数量
    train_stations = set(d['station'] for d in train_data)
    val_stations = set(d['station'] for d in val_data)
    test_stations = set(d['station'] for d in test_data)

    print(f"训练集测站数: {len(train_stations)}")
    print(f"验证集测站数: {len(val_stations)}")
    print(f"测试集测站数: {len(test_stations)}")
    print("=" * 60 + "\n")


def check_cuda():
    """检查CUDA可用性"""
    print("\n" + "=" * 60)
    print("系统信息")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"使用设备: {Config.DEVICE}")
    print("=" * 60 + "\n")


def format_time(seconds):
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Logger:
    """日志记录器"""

    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, message):
        """记录日志"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        print(message)

    def log_separator(self):
        """记录分隔符"""
        separator = "=" * 60
        self.log(separator)