"""
配置文件 - 包含所有超参数和路径配置
"""
import torch
import os

class Config:
    # ============ 路径配置 ============
    BASE_PATH = r"C:\Users\73955\Desktop\data_figs\Data\陆态网插值"
    STATIONS_FILE = os.path.join(BASE_PATH, "stations.xlsx")
    TEST_STATIONS_FILE = os.path.join(BASE_PATH, "test.xlsx")  # 不参与训练的测试站
    TRAIN_TEST_FILE = os.path.join(BASE_PATH, "train_test.xlsx")  # 训练-测试站（可选）
    RES_ZTD_PATH = os.path.join(BASE_PATH, "RES_ZTD")
    HGPT2_PATH = os.path.join(BASE_PATH, "HGPT2")
    GNSS_PATH = os.path.join(BASE_PATH, "GNSS")
    OUTPUT_PATH = os.path.join(BASE_PATH, "results")

    # ============ 数据配置 ============
    YEARS = [2020, 2021]
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ============ 缺失实验配置 ============
    MISSING_CONFIGS = {
        '短时缺失_1h': {'hours': 1, 'history_multiplier': 4},
        '短时缺失_2h': {'hours': 2, 'history_multiplier': 4},
        '短时缺失_3h': {'hours': 3, 'history_multiplier': 4},
        '短时缺失_4h': {'hours': 4, 'history_multiplier': 4},
        '短时缺失_5h': {'hours': 5, 'history_multiplier': 4},
        '短时缺失_6h': {'hours': 6, 'history_multiplier': 4},
        '短时缺失_12h': {'hours': 12, 'history_multiplier': 4},
    }

    # ============ 模型超参数 ============
    # 模型选择: 'BiLSTM' 或 'Transformer'
    MODEL_TYPE = 'BiLSTM'  # 可选: 'BiLSTM', 'Transformer'

    # BiLSTM参数
    INPUT_SIZE = 13  # 输入特征数: ZTD残差(1) + HGPT2-ZTD(1) + 时间sin/cos(4) + 位置特征(3) + 空间相关性(4)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BIDIRECTIONAL = True
    SEQ_LEN = 24  # 序列长度（历史时间步数）

    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20

    # 优化器参数
    WEIGHT_DECAY = 1e-5

    # 学习率调度
    LR_SCHEDULER = 'ReduceLROnPlateau'
    LR_PATIENCE = 5
    LR_FACTOR = 0.5

    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============ 空间相关性配置 ============
    NUM_NEIGHBORS = 5  # 用于计算空间相关性的邻近站点数

    # ============ 其他配置 ============
    RANDOM_SEED = 42
    VERBOSE = True

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("="*60)
        print("配置信息:")
        print("="*60)
        print(f"设备: {cls.DEVICE}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"隐藏层大小: {cls.HIDDEN_SIZE}")
        print(f"LSTM层数: {cls.NUM_LAYERS}")
        print(f"Dropout: {cls.DROPOUT}")
        print("="*60)