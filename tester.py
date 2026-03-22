"""
测试和评估模块 - 缺失值内插实验
v4.0: 添加HGPT2-ZTD作为滚动预测的物理约束
"""
import torch
import numpy as np
import os
import pandas as pd
from config import Config
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MissingValueTester:
    def __init__(self, model, preprocessor, hgpt2_data, gnss_data, train_test_stations=None, device=Config.DEVICE):
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.hgpt2_data = hgpt2_data
        self.gnss_data = gnss_data
        self.device = device
        self.test_stations = preprocessor.test_stations  # 不参与训练的测试站
        self.train_test_stations = set(train_test_stations) if train_test_stations else None  # 固定的训练站测试列表

    def create_sequence_data(self, features_list, start_idx, seq_len):
        """创建序列数据"""
        if start_idx + seq_len > len(features_list):
            return None

        sequence = features_list[start_idx:start_idx + seq_len]
        X = np.array([item['features'] for item in sequence])
        return X, sequence

    def predict_single_step(self, history_seq):
        """预测单个时间步"""
        # history_seq: (seq_len, features) - 已经是标准化后的
        X = torch.FloatTensor(history_seq).unsqueeze(0).to(self.device)  # (1, seq_len, features)

        with torch.no_grad():
            pred_normalized = self.model(X).item()  # 模型输出的是标准化后的值

            # 反标准化预测值(只反标准化第0个特征,即ZTD残差)
            if hasattr(self.preprocessor, 'scaler') and self.preprocessor.scaler is not None:
                # 创建一个dummy数组用于反标准化
                n_features = self.preprocessor.scaler.n_features_in_
                dummy_features = np.zeros((1, n_features))
                dummy_features[0, 0] = pred_normalized

                # 反标准化
                denormalized = self.preprocessor.scaler.inverse_transform(dummy_features)
                pred_value = float(denormalized[0, 0])
            else:
                pred_value = pred_normalized

            return pred_value

    def get_hgpt2_value(self, station, year, doy, hour):
        """获取指定时间点的HGPT2-ZTD值"""
        key = f"{station}_{year}"
        if key not in self.hgpt2_data:
            return 0.0

        hgpt2_df = self.hgpt2_data[key]
        match = hgpt2_df[
            (hgpt2_df['year'] == year) &
            (hgpt2_df['doy'] == doy) &
            (hgpt2_df['hour'] == hour)
        ]

        if not match.empty:
            try:
                val = match.iloc[0]['ztd']
                if val is not None and str(val) not in ['NaT', 'nan', 'NaN', '']:
                    return float(val)
            except (ValueError, TypeError):
                pass
        return 0.0

    def rolling_forecast(self, history_data, n_steps, station, start_year, start_doy, start_hour):
        """
        滚动预测 - 前向预测（改进版，使用真实HGPT2-ZTD）

        参数:
            history_data: 历史数据列表
            n_steps: 预测步数
            station: 测站名
            start_year, start_doy, start_hour: 预测起始时间
        """
        predictions = []
        current_history = [item['features'].copy() for item in history_data]

        # 第一次预测前打印调试信息
        if len(current_history) > 0:
            print(f"      滚动预测调试:")
            print(f"        历史数据长度: {len(current_history)}")
            print(f"        特征维度: {len(current_history[0])}")
            print(f"        最后一个历史ZTD(标准化): {current_history[-1][0]:.4f}")
            if len(current_history[0]) > 1:
                print(f"        最后一个历史HGPT2(标准化): {current_history[-1][1]:.4f}")

        for step in range(n_steps):
            # 计算当前预测步的时间
            total_hours = start_doy * 24 + start_hour + step
            current_doy = total_hours // 24
            current_hour = total_hours % 24
            current_year = start_year

            # 处理跨年
            days_in_year = 366 if (current_year % 4 == 0 and (current_year % 100 != 0 or current_year % 400 == 0)) else 365
            if current_doy > days_in_year:
                current_doy -= days_in_year
                current_year += 1

            # 预测下一步残差(会自动反标准化)
            pred_residual = self.predict_single_step(np.array(current_history))
            predictions.append(pred_residual)

            # 只在第一步和最后一步打印
            if step == 0 or step == n_steps - 1:
                print(f"        步骤{step+1}: 预测残差(反标准化后) = {pred_residual:.4f}")

            # 更新历史数据(滑动窗口) - 关键改进：使用真实的HGPT2-ZTD
            # 创建新的特征向量（基于上一个时间步的特征）
            new_features = current_history[-1].copy()

            # 获取当前时间点的真实HGPT2-ZTD值
            hgpt2_value = self.get_hgpt2_value(station, current_year, current_doy, current_hour)

            # 更新特征[0]: 预测的ZTD残差（需要标准化）
            if hasattr(self.preprocessor, 'scaler') and self.preprocessor.scaler is not None:
                n_features = self.preprocessor.scaler.n_features_in_
                dummy_features = np.zeros((1, n_features))

                # 标准化预测的残差
                dummy_features[0, 0] = pred_residual
                normalized = self.preprocessor.scaler.transform(dummy_features)
                new_features[0] = float(normalized[0, 0])

                # 标准化HGPT2-ZTD值（特征[1]）
                dummy_features_hgpt2 = np.zeros((1, n_features))
                dummy_features_hgpt2[0, 1] = hgpt2_value
                normalized_hgpt2 = self.preprocessor.scaler.transform(dummy_features_hgpt2)
                new_features[1] = float(normalized_hgpt2[0, 1])
            else:
                new_features[0] = pred_residual
                new_features[1] = hgpt2_value

            # 更新时间特征（特征[2-5]：sin/cos编码）
            doy_normalized = current_doy / 366.0
            hour_normalized = current_hour / 24.0
            new_features[2] = np.sin(2 * np.pi * doy_normalized)  # doy_sin
            new_features[3] = np.cos(2 * np.pi * doy_normalized)  # doy_cos
            new_features[4] = np.sin(2 * np.pi * hour_normalized)  # hour_sin
            new_features[5] = np.cos(2 * np.pi * hour_normalized)  # hour_cos

            # 滑动窗口
            current_history = current_history[1:] + [new_features]

        return predictions

    def backward_correction(self, predictions, end_value):
        """
        后向误差修正 - 使用指数衰减

        使用指数函数分配误差,越靠近终点的预测点获得越大的修正
        指数函数比二次函数更激进,更符合LSTM误差加速累积的特性
        """
        if len(predictions) == 0:
            return predictions

        # 计算累积误差
        accumulated_error = predictions[-1] - end_value

        # 指数衰减分配误差
        corrected = []
        n = len(predictions)

        # 计算指数权重 (lambda控制衰减速度,越大越激进)
        lambda_param = 0.1  # 衰减参数,可调节
        weights = [np.exp(lambda_param * (i + 1)) for i in range(n)]
        total_weight = sum(weights)

        for i, pred in enumerate(predictions):
            # 指数修正: 越靠后修正越大,且增长更快
            correction = accumulated_error * weights[i] / total_weight
            corrected.append(pred - correction)

        return corrected

    def interpolate_missing_gap(self, station, year, gap_start_idx, gap_hours, features_list):
        """
        对单个缺失间隙进行内插

        参数:
            station: 测站名
            year: 年份(整数)
            gap_start_idx: 缺失开始的索引
            gap_hours: 缺失小时数
            features_list: 该测站的所有特征数据
        """
        # 确保year是整数
        year = int(year)

        history_multiplier = Config.MISSING_CONFIGS[list(Config.MISSING_CONFIGS.keys())[0]]['history_multiplier']
        history_len = gap_hours * history_multiplier

        # 检查是否有足够的历史数据
        if gap_start_idx < history_len:
            return None

        # 检查是否有足够的未来数据
        if gap_start_idx + gap_hours >= len(features_list):
            return None

        # 获取历史数据
        history_data = features_list[gap_start_idx - history_len:gap_start_idx]

        # 获取缺失起点的时间信息（用于生成连续时间序列）
        start_item = features_list[gap_start_idx]
        start_year = int(start_item['year'])
        start_doy = int(start_item['doy'])
        start_hour = int(start_item['hour'])

        # 获取缺失后的第一个真实值(用于后向修正)
        end_point = features_list[gap_start_idx + gap_hours]
        end_value = end_point['features'][0]  # 真实的ZTD残差

        # 前向滚动预测（使用真实HGPT2-ZTD）
        predictions = self.rolling_forecast(
            history_data, gap_hours, station, start_year, start_doy, start_hour
        )

        # 后向误差修正
        predictions_corrected = self.backward_correction(predictions, end_value)

        # 构建结果 - 使用计算的连续时间序列
        results = []
        key = f"{station}_{year}"

        for i, pred_residual in enumerate(predictions_corrected):
            # 计算当前预测点的时间（从起点开始连续递增）
            total_hours = start_doy * 24 + start_hour + i
            item_doy = total_hours // 24
            item_hour = total_hours % 24
            item_year = start_year

            # 处理跨年情况
            days_in_year = 366 if (item_year % 4 == 0 and (item_year % 100 != 0 or item_year % 400 == 0)) else 365
            if item_doy > days_in_year:
                item_doy -= days_in_year
                item_year += 1

            # 获取HGPT2值
            hgpt2_ztd = self.get_hgpt2_value(station, item_year, item_doy, item_hour)

            # 计算预测的ZTD = 预测的残差 + HGPT2
            pred_ztd = float(pred_residual) + hgpt2_ztd

            # 获取真实ZTD（用于评估，可能不存在）
            true_ztd = 0.0
            gnss_key = f"{station}_{item_year}"
            if gnss_key in self.gnss_data:
                gnss_df = self.gnss_data[gnss_key]
                gnss_match = gnss_df[
                    (gnss_df['year'] == item_year) &
                    (gnss_df['doy'] == item_doy) &
                    (gnss_df['hour'] == item_hour)
                ]
                if not gnss_match.empty:
                    try:
                        val = gnss_match.iloc[0]['ztd']
                        if val is not None and str(val) not in ['NaT', 'nan', 'NaN', '']:
                            true_ztd = float(val)
                    except (ValueError, TypeError):
                        pass  # 保持默认值0.0

            # 计算真实残差 = 真实ZTD - HGPT2
            true_residual = float(true_ztd - hgpt2_ztd) if true_ztd != 0 and hgpt2_ztd != 0 else 0.0

            results.append({
                'year': item_year,
                'doy': item_doy,
                'hour': item_hour,
                'true_ztd': true_ztd,
                'hgpt2_ztd': hgpt2_ztd,
                'pred_ztd': pred_ztd,
                'true_residual': true_residual,
                'pred_residual': float(pred_residual),
                'station': station
            })

        return results

    def save_experiment_results_by_station(self, config_name, station_results, output_dir):
        """按测站分别保存实验结果"""
        # 为每个测站保存预测结果文件
        for station, results in station_results.items():
            # 保存ZTD结果
            result_file = os.path.join(output_dir, f'{config_name}_{station}_预测结果.txt')
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write('年份\t年积日\t日内时\tGNSS-ZTD\tHGPT2-ZTD\t预测ZTD\n')
                for r in results:
                    year = int(r['year'])
                    doy = int(r['doy'])
                    hour = int(r['hour'])
                    true_ztd = float(r['true_ztd'])
                    hgpt2_ztd = float(r['hgpt2_ztd'])
                    pred_ztd = float(r['pred_ztd'])

                    f.write(f"{year}\t{doy}\t{hour}\t"
                           f"{true_ztd:.4f}\t{hgpt2_ztd:.4f}\t{pred_ztd:.4f}\n")

            # 保存残差结果
            residual_file = os.path.join(output_dir, f'{config_name}_{station}_残差结果.txt')
            with open(residual_file, 'w', encoding='utf-8') as f:
                f.write('年份\t年积日\t日内时\t真实ZTD残差\t预测ZTD残差\n')
                for r in results:
                    year = int(r['year'])
                    doy = int(r['doy'])
                    hour = int(r['hour'])
                    # 检查是否有残差字段,并确保不是简单的get默认值
                    true_residual = r.get('true_residual')
                    pred_residual = r.get('pred_residual')

                    # 如果残差字段存在且不是None,使用它们
                    if true_residual is not None and pred_residual is not None:
                        f.write(f"{year}\t{doy}\t{hour}\t"
                               f"{float(true_residual):.4f}\t{float(pred_residual):.4f}\n")
                    else:
                        # 如果没有残差字段,写入0(并打印警告)
                        if true_residual is None or pred_residual is None:
                            print(f"      警告: {year}/{doy}/{hour} 残差数据缺失")
                        f.write(f"{year}\t{doy}\t{hour}\t0.0000\t0.0000\n")

            print(f"  保存 {station}:")
            print(f"    - ZTD预测: {result_file}")
            print(f"    - 残差预测: {residual_file}")

        # 合并所有结果用于总体评估
        all_results = []
        for results in station_results.values():
            all_results.extend(results)

        # 计算总体评估指标
        metrics = self.evaluate_results(all_results)

        # 按测站分组评估
        station_metrics = {}
        for station, results in station_results.items():
            station_metrics[station] = self.evaluate_results(results)

        # 保存评估指标
        metrics_file = os.path.join(output_dir, f'{config_name}_评估指标.txt')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f'实验: {config_name}\n')
            f.write('='*60 + '\n\n')

            if metrics:
                f.write('总体评估指标:\n')
                f.write(f'  RMSE: {metrics["RMSE"]:.2f} mm\n')
                f.write(f'  MAE: {metrics["MAE"]:.2f} mm\n')
                f.write(f'  Bias: {metrics["Bias"]:.2f} mm\n')
                f.write(f'  R²: {metrics["R2"]:.4f}\n')
                f.write(f'  样本数: {metrics["n_samples"]}\n\n')

            # 各测站评估
            f.write('各测站评估:\n')
            f.write('-'*60 + '\n')
            for station in sorted(station_metrics.keys()):
                m = station_metrics[station]
                if m:
                    f.write(f'\n{station}:\n')
                    f.write(f'  RMSE: {m["RMSE"]:.2f} mm\n')
                    f.write(f'  MAE: {m["MAE"]:.2f} mm\n')
                    f.write(f'  Bias: {m["Bias"]:.2f} mm\n')
                    f.write(f'  R²: {m["R2"]:.4f}\n')
                    f.write(f'  样本数: {m["n_samples"]}\n')

        print(f'\n总体评估指标已保存: {metrics_file}')
        if metrics:
            print(f'  RMSE: {metrics["RMSE"]:.2f} mm')
            print(f'  MAE: {metrics["MAE"]:.2f} mm')
            print(f'  Bias: {metrics["Bias"]:.2f} mm')
            print(f'  R²: {metrics["R2"]:.4f}')
            print(f'  样本数: {metrics["n_samples"]}')

    def run_all_experiments(self, train_data, test_data, output_dir):
        """运行完整实验:训练站和测试站"""
        print("\n" + "="*60)
        print("开始缺失值内插实验")
        print("="*60)

        # 创建输出目录
        train_output_dir = os.path.join(output_dir, "训练站结果")
        test_output_dir = os.path.join(output_dir, "测试站结果")
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        # 1. 测试站实验
        print("\n" + "="*60)
        print("第一部分: 测试站实验 (未参与训练的27个站)")
        print("="*60)
        self.run_experiments_for_test_stations(test_data, test_output_dir)

        # 2. 训练站实验 (随机选择部分训练站)
        print("\n" + "="*60)
        print("第二部分: 训练站实验 (参与训练的站点)")
        print("="*60)
        self.run_experiments_for_train_stations(train_data, train_output_dir)

        print("\n" + "="*60)
        print("所有实验完成!")
        print("="*60)
        print(f"测试站结果保存在: {test_output_dir}")
        print(f"训练站结果保存在: {train_output_dir}")

    def run_experiments_for_train_stations(self, train_data, output_dir):
        """对训练站进行实验（使用固定的站点列表）"""
        os.makedirs(output_dir, exist_ok=True)

        # 按测站分组
        station_year_data = {}
        for item in train_data:
            year = int(item['year']) if isinstance(item['year'], float) else item['year']
            station = item['station']
            key = f"{station}_{year}"
            if key not in station_year_data:
                station_year_data[key] = []
            station_year_data[key].append(item)

        # 按时间排序
        for key in station_year_data:
            station_year_data[key].sort(key=lambda x: (x['doy'], x['hour']))

        available_stations = list(set(k.rsplit('_', 1)[0] for k in station_year_data.keys()))

        # 选择训练站进行测试
        if self.train_test_stations and len(self.train_test_stations) > 0:
            # 使用固定的训练站列表（来自train_test.xlsx）
            selected_stations = [s for s in self.train_test_stations if s in available_stations]
            selection_mode = "固定列表（来自train_test.xlsx）"
        else:
            # 回退到随机选择模式
            n_test_stations = len(self.test_stations)
            n_select = min(n_test_stations, len(available_stations))
            np.random.seed(Config.RANDOM_SEED)
            selected_stations = np.random.choice(available_stations, n_select, replace=False).tolist()
            selection_mode = f"随机选择（种子{Config.RANDOM_SEED}）"

        # 筛选选中的测站
        selected_data = {k: v for k, v in station_year_data.items()
                        if k.rsplit('_', 1)[0] in selected_stations}

        print(f"\n训练站数据:")
        print(f"  总训练站数: {len(available_stations)}")
        print(f"  选择模式: {selection_mode}")
        print(f"  选中站点数: {len(selected_stations)}")
        print(f"  选中的站点: {sorted(selected_stations)}")
        print(f"  共 {len(selected_data)} 个测站年份")
        for key, data in selected_data.items():
            print(f"  {key}: {len(data)} 条数据")

        # 对每种缺失配置进行实验
        for config_name, config in Config.MISSING_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"实验: {config_name}")
            print(f"缺失时长: {config['hours']} 小时")
            print(f"{'='*60}")

            gap_hours = config['hours']
            all_results = {}

            # 对每个选中的测站年份进行实验
            for station_year_key, features_list in selected_data.items():
                station, year_str = station_year_key.rsplit('_', 1)
                year = int(year_str)

                # 找到第一个0时的位置作为缺失起点
                gap_start = None
                history_len = gap_hours * config['history_multiplier']

                print(f"\n  检查 {station_year_key}:")
                print(f"    数据长度: {len(features_list)}")

                for idx, item in enumerate(features_list):
                    if item['hour'] == 0 and idx >= history_len and idx + gap_hours < len(features_list):
                        gap_start = idx
                        break

                if gap_start is None:
                    # 如果找不到0时,尝试从任意位置开始
                    if len(features_list) >= history_len + gap_hours:
                        gap_start = history_len
                        print(f"    ! 未找到0时起点,使用索引{gap_start}开始")
                    else:
                        print(f"    ✗ 跳过: 数据不足")
                        continue

                start_item = features_list[gap_start]
                print(f"    开始预测: 从第{start_item['doy']}天{start_item['hour']}时")

                # 进行内插
                results = self.interpolate_missing_gap(
                    station, year, gap_start, gap_hours, features_list
                )

                if results and len(results) > 0:
                    if station not in all_results:
                        all_results[station] = []
                    all_results[station].extend(results)
                    print(f"    ✓ 成功预测 {len(results)} 个时间点")
                else:
                    print(f"    ✗ 预测失败")

            # 保存结果
            if all_results:
                self.save_experiment_results_by_station(config_name, all_results, output_dir)
            else:
                print(f"  ✗ 没有成功的预测结果")

    def evaluate_results(self, results):
        """评估预测结果"""
        if not results:
            return None

        # 确保数据类型正确,并过滤无效值
        true_ztd = []
        pred_ztd = []

        for r in results:
            try:
                t = float(r['true_ztd'])
                p = float(r['pred_ztd'])
                # 只保留非零的真实值
                if t != 0 and not np.isnan(t) and not np.isnan(p):
                    true_ztd.append(t)
                    pred_ztd.append(p)
            except (ValueError, TypeError):
                continue

        if len(true_ztd) < 2:
            return None

        true_ztd = np.array(true_ztd)
        pred_ztd = np.array(pred_ztd)

        rmse = np.sqrt(mean_squared_error(true_ztd, pred_ztd))
        mae = mean_absolute_error(true_ztd, pred_ztd)
        bias = np.mean(pred_ztd - true_ztd)
        r2 = r2_score(true_ztd, pred_ztd)

        return {
            'RMSE': rmse,
            'MAE': mae,
            'Bias': bias,
            'R2': r2,
            'n_samples': len(true_ztd)
        }

    def run_experiments_for_test_stations(self, test_data, output_dir):
        """对测试站运行缺失值内插实验"""
        os.makedirs(output_dir, exist_ok=True)

        # 按测站和年份分组
        station_year_data = {}
        for item in test_data:
            # 确保year是整数
            year = int(item['year']) if isinstance(item['year'], float) else item['year']
            key = f"{item['station']}_{year}"
            if key not in station_year_data:
                station_year_data[key] = []
            station_year_data[key].append(item)

        # 按时间排序
        for key in station_year_data:
            station_year_data[key].sort(key=lambda x: (x['doy'], x['hour']))

        print(f"\n测试站数据:")
        print(f"  共 {len(station_year_data)} 个测站年份")
        for key, data in station_year_data.items():
            print(f"  {key}: {len(data)} 条数据")

        # 对每种缺失配置进行实验
        for config_name, config in Config.MISSING_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"实验: {config_name}")
            print(f"缺失时长: {config['hours']} 小时")
            print(f"{'='*60}")

            gap_hours = config['hours']
            all_results = []

            # 对每个测站年份进行实验
            for station_year_key, features_list in station_year_data.items():
                station, year_str = station_year_key.rsplit('_', 1)
                year = int(year_str)  # 确保年份是整数

                # 找到第一个0时的位置作为缺失起点（与训练站一致）
                history_len = gap_hours * config['history_multiplier']
                gap_start = None

                for idx, item in enumerate(features_list):
                    if item['hour'] == 0 and idx >= history_len and idx + gap_hours < len(features_list):
                        gap_start = idx
                        break

                if gap_start is None:
                    # 如果找不到0时,尝试从任意位置开始
                    if len(features_list) >= history_len + gap_hours + 1:
                        gap_start = history_len
                        print(f"  {station_year_key}: 未找到0时起点,使用索引{gap_start}开始")
                    else:
                        print(f"  跳过 {station_year_key}: 数据不足")
                        continue

                start_item = features_list[gap_start]
                print(f"  处理 {station_year_key}: 从第{int(start_item['doy'])}天{int(start_item['hour'])}时开始 (索引{gap_start}, 共{len(features_list)}条)")

                # 进行内插
                results = self.interpolate_missing_gap(
                    station, year, gap_start, gap_hours, features_list
                )

                if results:
                    all_results.extend(results)
                    print(f"    ✓ 成功预测 {len(results)} 个时间点")
                else:
                    print(f"    ✗ 预测失败")

            # 保存结果
            if all_results:
                self.save_experiment_results(config_name, all_results, output_dir)
            else:
                print(f"  ✗ 没有成功的预测结果")

    def save_experiment_results(self, config_name, results, output_dir):
        """保存实验结果 - 转换格式后调用按测站保存的方法"""
        # 将结果按测站分组
        station_results = {}
        for r in results:
            station = r['station']
            if station not in station_results:
                station_results[station] = []
            station_results[station].append(r)

        # 调用按测站保存的方法
        self.save_experiment_results_by_station(config_name, station_results, output_dir)