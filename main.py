"""
主程序 - 陆态网ZTD缺失值内插实验
"""
import os
import time
from config import Config
from data_loader import DataLoader
from preprocessor import Preprocessor
from model import BiLSTMModel, TransformerModel
from trainer import Trainer, create_data_loaders
from tester import MissingValueTester
from utils import (
    set_seed,
    create_output_directory,
    save_training_losses,
    print_data_summary,
    check_cuda,
    format_time,
    Logger
)

def main():
    """主函数"""
    # 记录开始时间
    start_time = time.time()

    # 创建输出目录
    create_output_directory()

    # 创建日志
    log_file = os.path.join(Config.OUTPUT_PATH, 'experiment_log.txt')
    logger = Logger(log_file)

    logger.log_separator()
    logger.log("陆态网ZTD缺失值内插实验")
    logger.log_separator()

    # 设置随机种子
    set_seed()
    logger.log(f"随机种子设置为: {Config.RANDOM_SEED}")

    # 检查CUDA
    check_cuda()

    # 打印配置
    Config.print_config()

    # ============ 步骤1: 数据加载 ============
    logger.log_separator()
    logger.log("步骤1: 数据加载")
    logger.log_separator()

    data_loader = DataLoader()
    stations_info, test_stations_info, res_ztd_data, hgpt2_data, gnss_data = data_loader.load_all()

    logger.log(f"加载测站数: {len(stations_info)}")
    if test_stations_info is not None:
        logger.log(f"加载测试站数: {len(test_stations_info)}")
        test_station_names = list(test_stations_info['Station'].values)
    else:
        test_station_names = []
    logger.log(f"加载残差ZTD数据: {len(res_ztd_data)} 个测站年份")
    logger.log(f"加载HGPT2数据: {len(hgpt2_data)} 个测站年份")
    logger.log(f"加载GNSS数据: {len(gnss_data)} 个测站年份")

    # 检查是否读取到数据
    if len(res_ztd_data) == 0:
        logger.log("错误: 没有读取到任何残差ZTD数据!")
        logger.log("请检查:")
        logger.log(f"1. 路径是否正确: {Config.RES_ZTD_PATH}")
        logger.log("2. 文件格式是否为: station_name + year (如 ahaq2020)")
        logger.log("3. 文件内容是否为空格分隔的6列数据")
        return

    # 显示一些样例数据
    sample_key = list(res_ztd_data.keys())[0]
    sample_df = res_ztd_data[sample_key]
    logger.log(f"\n样例数据 ({sample_key}):")
    logger.log(f"数据形状: {sample_df.shape}")
    logger.log(f"前3行:\n{sample_df.head(3)}")

    # ============ 步骤2: 数据预处理 ============
    logger.log_separator()
    logger.log("步骤2: 数据预处理")
    logger.log_separator()

    preprocessor = Preprocessor(stations_info, res_ztd_data, test_station_names, hgpt2_data)

    # 准备特征
    all_features = preprocessor.prepare_all_features()
    logger.log(f"生成特征总数: {len(all_features)}")

    # 按测试站划分数据集
    train_data, val_data, test_data = preprocessor.split_data_by_stations(all_features)

    # 标准化
    train_data, val_data, test_data = preprocessor.normalize_features(
        train_data, val_data, test_data
    )

    print_data_summary(train_data, val_data, test_data)

    # ============ 步骤3: 创建数据加载器 ============
    logger.log_separator()
    logger.log("步骤3: 创建数据加载器")
    logger.log_separator()

    train_loader, val_loader = create_data_loaders(train_data, val_data, seq_len=24)
    logger.log(f"训练批次数: {len(train_loader)}")
    logger.log(f"验证批次数: {len(val_loader)}")

    # ============ 步骤4: 模型训练 ============
    logger.log_separator()
    logger.log("步骤4: 模型训练")
    logger.log_separator()

    # 根据配置选择模型
    if Config.MODEL_TYPE == 'Transformer':
        model = TransformerModel()
        logger.log(f"使用模型: Transformer")
    else:
        model = BiLSTMModel()
        logger.log(f"使用模型: BiLSTM")

    logger.log(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    trainer = Trainer(model, train_loader, val_loader)

    train_start = time.time()
    train_losses, val_losses = trainer.train()
    train_end = time.time()

    logger.log(f"训练耗时: {format_time(train_end - train_start)}")
    logger.log(f"最终训练损失: {train_losses[-1]:.6f}")
    logger.log(f"最终验证损失: {val_losses[-1]:.6f}")

    # 保存训练损失
    save_training_losses(train_losses, val_losses, Config.OUTPUT_PATH)

    # 保存模型
    model_path = os.path.join(Config.OUTPUT_PATH, 'bilstm_model.pth')
    trainer.save_model(model_path)

    # ============ 步骤5: 缺失值内插实验 ============
    logger.log_separator()
    logger.log("步骤5: 缺失值内插实验 (测试站 + 训练站)")
    logger.log_separator()

    # 获取固定的训练站测试列表
    train_test_station_names = None
    if hasattr(data_loader, 'train_test_stations_info') and data_loader.train_test_stations_info is not None:
        train_test_station_names = list(data_loader.train_test_stations_info['Station'].values)
        logger.log(f"使用固定训练站测试列表: {len(train_test_station_names)} 个站点")
    else:
        logger.log("未找到train_test.xlsx，训练站实验将随机选择站点")

    tester = MissingValueTester(model, preprocessor, hgpt2_data, gnss_data, train_test_station_names)

    test_start = time.time()
    # 同时运行测试站和训练站实验
    tester.run_all_experiments(train_data, test_data, Config.OUTPUT_PATH)
    test_end = time.time()

    logger.log(f"测试耗时: {format_time(test_end - test_start)}")

    # ============ 完成 ============
    end_time = time.time()
    total_time = end_time - start_time

    logger.log_separator()
    logger.log("实验完成!")
    logger.log_separator()
    logger.log(f"总耗时: {format_time(total_time)}")
    logger.log(f"所有结果已保存至: {Config.OUTPUT_PATH}")
    logger.log_separator()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()