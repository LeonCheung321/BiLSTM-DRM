"""
数据预处理模块 - 特征工程和数据清洗
v2.0: 添加HGPT2-ZTD作为物理约束特征
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import Config


class Preprocessor:
    def __init__(self, stations_info, res_ztd_data, test_stations=None, hgpt2_data=None):
        self.stations_info = stations_info
        self.res_ztd_data = res_ztd_data
        self.hgpt2_data = hgpt2_data if hgpt2_data is not None else {}
        self.test_stations = set(test_stations) if test_stations else set()
        self.scaler = StandardScaler()
        self.station_coords = {}
        self._build_station_coords()

        if self.test_stations:
            print(f"测试站数量: {len(self.test_stations)}")
            print(f"测试站列表: {sorted(self.test_stations)}")

    def _build_station_coords(self):
        """构建测站坐标字典"""
        for _, row in self.stations_info.iterrows():
            self.station_coords[row['Station']] = {
                'lon': row['Lon'],
                'lat': row['Lat'],
                'height': row['Height']
            }

    def clean_data(self, df):
        """清洗数据 - 只移除NaN值,保留0值"""
        # 残差ZTD可能为0(表示GNSS和HGPT2一致),这是有效数据
        # 只移除NaN和明显异常的值
        df_clean = df.dropna(subset=['ztd']).copy()

        # 可选: 移除极端异常值(例如绝对值超过500mm的残差)
        # df_clean = df_clean[abs(df_clean['ztd']) < 500]

        return df_clean

    def calculate_spatial_distance(self, station1, station2):
        """计算两个测站之间的空间距离"""
        if station1 not in self.station_coords or station2 not in self.station_coords:
            return np.inf

        coord1 = self.station_coords[station1]
        coord2 = self.station_coords[station2]

        # 简化的欧氏距离
        dist = np.sqrt((coord1['lon'] - coord2['lon'])**2 +
                      (coord1['lat'] - coord2['lat'])**2 +
                      ((coord1['height'] - coord2['height'])/1000)**2)
        return dist

    def find_nearest_stations(self, station, year, n_neighbors=Config.NUM_NEIGHBORS):
        """找到最近的N个测站"""
        distances = []
        for key in self.res_ztd_data.keys():
            other_station, other_year = key.rsplit('_', 1)
            if other_station != station and int(other_year) == year:
                dist = self.calculate_spatial_distance(station, other_station)
                distances.append((other_station, dist))

        distances.sort(key=lambda x: x[1])
        return [s for s, d in distances[:n_neighbors]]

    def get_hgpt2_value(self, station, year, doy, hour):
        """获取HGPT2-ZTD值"""
        key = f"{station}_{year}"
        if key not in self.hgpt2_data:
            return 0.0  # 如果没有HGPT2数据，返回0

        hgpt2_df = self.hgpt2_data[key]
        # 查找匹配的时间点
        match = hgpt2_df[(hgpt2_df['doy'] == doy) & (hgpt2_df['hour'] == hour)]

        if len(match) > 0:
            return match.iloc[0]['ztd']
        else:
            return 0.0  # 如果没有找到匹配的时间点，返回0

    def create_features(self, station, year):
        """为单个测站年份创建特征"""
        key = f"{station}_{year}"
        if key not in self.res_ztd_data:
            return None

        df = self.clean_data(self.res_ztd_data[key])
        if len(df) < 100:  # 数据太少则跳过
            return None

        # 获取测站位置信息
        if station not in self.station_coords:
            return None

        coords = self.station_coords[station]

        # 预先计算该测站的邻近站点
        neighbors = self.find_nearest_stations(station, year)

        # 预先加载邻近站点的数据到内存
        neighbor_data_cache = {}
        for neighbor in neighbors:
            neighbor_key = f"{neighbor}_{year}"
            if neighbor_key in self.res_ztd_data:
                neighbor_df = self.res_ztd_data[neighbor_key]
                # 创建快速查找字典: (doy, hour) -> ztd
                lookup = {}
                for _, row in neighbor_df.iterrows():
                    lookup[(row['doy'], row['hour'])] = row['ztd']
                neighbor_data_cache[neighbor] = lookup

        features_list = []
        total_rows = len(df)
        print_interval = max(1, total_rows // 10)  # 每10%打印一次

        for idx, (_, row) in enumerate(df.iterrows()):
            if idx % print_interval == 0:
                progress = (idx / total_rows) * 100
                print(f"    处理 {station}_{year}: {progress:.0f}%", end='\r')

            # 获取HGPT2-ZTD值（物理约束）
            hgpt2_ztd = self.get_hgpt2_value(station, year, row['doy'], row['hour'])

            # 时间特征 - 使用sin/cos编码捕捉周期性
            doy_normalized = row['doy'] / 366.0
            hour_normalized = row['hour'] / 24.0

            # 年积日的周期性编码(捕捉季节变化)
            doy_sin = np.sin(2 * np.pi * doy_normalized)
            doy_cos = np.cos(2 * np.pi * doy_normalized)

            # 日内时的周期性编码(捕捉日变化)
            hour_sin = np.sin(2 * np.pi * hour_normalized)
            hour_cos = np.cos(2 * np.pi * hour_normalized)

            # 位置特征
            lon_normalized = coords['lon'] / 180.0
            lat_normalized = coords['lat'] / 90.0
            height_normalized = coords['height'] / 5000.0

            # 快速计算空间相关性特征(使用缓存)
            neighbor_ztd_values = []
            for neighbor, lookup in neighbor_data_cache.items():
                ztd_val = lookup.get((row['doy'], row['hour']))
                if ztd_val is not None and ztd_val != 0:
                    neighbor_ztd_values.append(ztd_val)

            # 计算统计特征
            neighbor_avg = np.mean(neighbor_ztd_values) if neighbor_ztd_values else 0
            gradient = np.std(neighbor_ztd_values) if len(neighbor_ztd_values) > 1 else 0
            n_neighbors = len(neighbor_ztd_values)
            neighbor_range = (max(neighbor_ztd_values) - min(neighbor_ztd_values)) if neighbor_ztd_values else 0

            features = [
                row['ztd'],  # 0: ZTD残差（预测目标）
                hgpt2_ztd,  # 1: HGPT2-ZTD（物理约束）
                doy_sin,  # 2: 年积日sin
                doy_cos,  # 3: 年积日cos
                hour_sin,  # 4: 日内时sin
                hour_cos,  # 5: 日内时cos
                lon_normalized,  # 6: 经度
                lat_normalized,  # 7: 纬度
                height_normalized,  # 8: 高程
                neighbor_avg,  # 9: 邻近站点平均ZTD
                gradient,  # 10: 梯度
                n_neighbors,  # 11: 有效邻近站点数
                neighbor_range,  # 12: 邻近站点ZTD范围
            ]

            features_list.append({
                'features': features,
                'year': row['year'],
                'doy': row['doy'],
                'hour': row['hour'],
                'station': station,
                'hgpt2_ztd': hgpt2_ztd  # 保存HGPT2值用于后续预测
            })

        print(f"    处理 {station}_{year}: 100% 完成")
        return features_list

    def prepare_all_features(self):
        """准备所有测站的特征"""
        print("准备特征...")
        all_features = []

        station_year_list = []
        for key in self.res_ztd_data.keys():
            station, year = key.rsplit('_', 1)
            year = int(year)
            station_year_list.append((station, year, key))

        total = len(station_year_list)
        print(f"需要处理 {total} 个测站年份")

        for idx, (station, year, key) in enumerate(station_year_list, 1):
            print(f"\n[{idx}/{total}] 处理测站: {station}_{year}")

            features = self.create_features(station, year)
            if features:
                all_features.extend(features)
                print(f"  ✓ 生成 {len(features)} 条特征")
            else:
                print(f"  ✗ 跳过(数据不足)")

        print(f"\n共生成 {len(all_features)} 条特征数据")
        return all_features

    def split_data_by_stations(self, features_data):
        """按测试站划分数据 - 测试站不参与训练"""
        train_data = []
        test_data = []

        for item in features_data:
            if item['station'] in self.test_stations:
                test_data.append(item)
            else:
                train_data.append(item)

        # 从训练数据中划分训练集和验证集
        np.random.seed(Config.RANDOM_SEED)
        np.random.shuffle(train_data)

        n_train = len(train_data)
        val_split = int(n_train * Config.TRAIN_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO))

        train_set = train_data[:val_split]
        val_set = train_data[val_split:]

        print(f"\n数据划分:")
        print(f"  训练集: {len(train_set)} (来自训练站)")
        print(f"  验证集: {len(val_set)} (来自训练站)")
        print(f"  测试集: {len(test_data)} (来自 {len(self.test_stations)} 个独立测试站)")

        # 统计测试站
        test_stations_used = set(d['station'] for d in test_data)
        print(f"  测试站: {sorted(test_stations_used)}")

        return train_set, val_set, test_data

    def normalize_features(self, train_data, val_data, test_data):
        """标准化特征"""
        train_features = np.array([d['features'] for d in train_data])
        val_features = np.array([d['features'] for d in val_data])
        test_features = np.array([d['features'] for d in test_data])

        # 使用训练集拟合标准化器
        self.scaler.fit(train_features)

        # 标准化所有数据
        train_features_normalized = self.scaler.transform(train_features)
        val_features_normalized = self.scaler.transform(val_features)
        test_features_normalized = self.scaler.transform(test_features)

        # 更新数据
        for i, d in enumerate(train_data):
            d['features'] = train_features_normalized[i]
        for i, d in enumerate(val_data):
            d['features'] = val_features_normalized[i]
        for i, d in enumerate(test_data):
            d['features'] = test_features_normalized[i]

        return train_data, val_data, test_data