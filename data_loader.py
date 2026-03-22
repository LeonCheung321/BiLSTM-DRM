"""
数据读取模块 - 负责读取所有原始数据
v4.0: 支持固定的训练-测试站划分
"""
import os
import pandas as pd
import numpy as np
from config import Config


class DataLoader:
    def __init__(self):
        self.stations_info = None
        self.train_test_stations_info = None  # 训练-测试站信息
        self.test_stations_info = None  # 不参与训练的测试站信息
        self.res_ztd_data = {}
        self.hgpt2_data = {}
        self.gnss_data = {}

    def load_stations(self):
        """读取全部测站信息"""
        print("读取全部测站信息...")
        self.stations_info = pd.read_excel(Config.STATIONS_FILE)
        print(f"共读取 {len(self.stations_info)} 个测站")
        return self.stations_info

    def load_train_test_stations(self):
        """读取训练-测试站信息（可选）"""
        print("读取训练-测试站信息...")
        try:
            if hasattr(Config, 'TRAIN_TEST_FILE') and os.path.exists(Config.TRAIN_TEST_FILE):
                self.train_test_stations_info = pd.read_excel(Config.TRAIN_TEST_FILE)
                print(f"共读取 {len(self.train_test_stations_info)} 个训练-测试站")
                print(f"训练-测试站列表: {list(self.train_test_stations_info['Station'].values)}")
            else:
                print("未找到train_test.xlsx文件，跳过")
                self.train_test_stations_info = None
        except Exception as e:
            print(f"警告: 无法读取训练-测试站文件 ({e})")
            self.train_test_stations_info = None
        return self.train_test_stations_info

    def load_test_stations(self):
        """读取不参与训练的测试站信息"""
        print("读取不参与训练的测试站信息...")
        try:
            self.test_stations_info = pd.read_excel(Config.TEST_STATIONS_FILE)
            print(f"共读取 {len(self.test_stations_info)} 个不参与训练的测试站")
            print(f"不参与训练的测试站列表: {list(self.test_stations_info['Station'].values)}")
        except Exception as e:
            print(f"警告: 无法读取测试站文件 ({e})")
            print("将使用默认测试站")
            self.test_stations_info = None
        return self.test_stations_info

    def load_ztd_file(self, filepath):
        """读取单个ZTD文件"""
        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None,
                           names=['year', 'doy', 'hour', 'minute', 'second', 'ztd'],
                           skipinitialspace=True, engine='python')
            if len(df) > 0:
                return df
            else:
                return None
        except Exception as e:
            return None

    def load_res_ztd(self):
        """读取所有残差ZTD数据"""
        print("读取残差ZTD数据...")
        for year in Config.YEARS:
            year_path = os.path.join(Config.RES_ZTD_PATH, str(year))
            if not os.path.exists(year_path):
                continue

            files = os.listdir(year_path)
            print(f"{year}年文件夹中有 {len(files)} 个文件")

            for filename in files:
                if filename.endswith(f"{year}.txt") or filename.endswith(str(year)):
                    if filename.endswith('.txt'):
                        station = filename[:-8]
                    else:
                        station = filename[:-4]

                    filepath = os.path.join(year_path, filename)

                    if not os.path.isfile(filepath):
                        continue

                    df = self.load_ztd_file(filepath)

                    if df is not None and len(df) > 0:
                        df_clean = df.dropna(subset=['ztd'])
                        if len(df_clean) > 0:
                            key = f"{station}_{year}"
                            self.res_ztd_data[key] = df_clean
                            if len(self.res_ztd_data) <= 5:
                                print(f"  成功读取: {key}, 数据行数: {len(df_clean)}")

        print(f"共读取 {len(self.res_ztd_data)} 个测站年份的残差ZTD数据")
        return self.res_ztd_data

    def load_hgpt2(self):
        """读取HGPT2-ZTD数据"""
        print("读取HGPT2-ZTD数据...")
        for year in Config.YEARS:
            year_path = os.path.join(Config.HGPT2_PATH, str(year))
            if not os.path.exists(year_path):
                continue

            files = os.listdir(year_path)
            print(f"{year}年文件夹中有 {len(files)} 个文件")

            for filename in files:
                if filename.endswith(f"{year}.txt") or filename.endswith(str(year)):
                    if filename.endswith('.txt'):
                        station = filename[:-8]
                    else:
                        station = filename[:-4]

                    filepath = os.path.join(year_path, filename)

                    if not os.path.isfile(filepath):
                        continue

                    df = self.load_ztd_file(filepath)

                    if df is not None and len(df) > 0:
                        df_clean = df.dropna(subset=['ztd'])
                        if len(df_clean) > 0:
                            key = f"{station}_{year}"
                            self.hgpt2_data[key] = df_clean

        print(f"共读取 {len(self.hgpt2_data)} 个测站年份的HGPT2数据")
        return self.hgpt2_data

    def load_gnss(self):
        """读取GNSS真实ZTD数据"""
        print("读取GNSS真实ZTD数据...")
        for year in Config.YEARS:
            year_path = os.path.join(Config.GNSS_PATH, str(year))
            if not os.path.exists(year_path):
                continue

            files = os.listdir(year_path)
            print(f"{year}年文件夹中有 {len(files)} 个文件")

            for filename in files:
                if filename.endswith(f"{year}.txt") or filename.endswith(str(year)):
                    if filename.endswith('.txt'):
                        station = filename[:-8]
                    else:
                        station = filename[:-4]

                    filepath = os.path.join(year_path, filename)

                    if not os.path.isfile(filepath):
                        continue

                    df = self.load_ztd_file(filepath)

                    if df is not None and len(df) > 0:
                        df_clean = df.dropna(subset=['ztd'])
                        if len(df_clean) > 0:
                            key = f"{station}_{year}"
                            self.gnss_data[key] = df_clean

        print(f"共读取 {len(self.gnss_data)} 个测站年份的GNSS数据")
        return self.gnss_data

    def load_all(self):
        """加载所有数据"""
        self.load_stations()
        self.load_train_test_stations()  # 新增
        self.load_test_stations()
        self.load_res_ztd()
        self.load_hgpt2()
        self.load_gnss()
        return self.stations_info, self.test_stations_info, self.res_ztd_data, self.hgpt2_data, self.gnss_data