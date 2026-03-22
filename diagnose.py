"""
数据诊断工具 - 检查数据文件结构和格式
"""
import os
import pandas as pd

# 直接定义路径,不依赖Config避免循环导入
BASE_PATH = r"C:\Users\73955\Desktop\data_figs\Data\陆态网插值"
STATIONS_FILE = os.path.join(BASE_PATH, "stations.xlsx")
RES_ZTD_PATH = os.path.join(BASE_PATH, "RES_ZTD")
HGPT2_PATH = os.path.join(BASE_PATH, "HGPT2")
GNSS_PATH = os.path.join(BASE_PATH, "GNSS")
YEARS = [2020, 2021]

def check_directory_structure():
    """检查目录结构"""
    print("="*60)
    print("目录结构检查")
    print("="*60)

    paths = {
        'BASE_PATH': BASE_PATH,
        'STATIONS_FILE': STATIONS_FILE,
        'RES_ZTD_PATH': RES_ZTD_PATH,
        'HGPT2_PATH': HGPT2_PATH,
        'GNSS_PATH': GNSS_PATH,
    }

    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")

        if exists and os.path.isdir(path):
            try:
                items = os.listdir(path)
                print(f"  包含 {len(items)} 个项目")
                if len(items) <= 10:
                    for item in items[:10]:
                        print(f"    - {item}")
            except:
                print(f"  无法列出内容")
    print()

def check_year_folders():
    """检查年份文件夹"""
    print("="*60)
    print("年份文件夹检查")
    print("="*60)

    for year in YEARS:
        print(f"\n{year}年:")
        for folder_name, folder_path in [
            ('RES_ZTD', RES_ZTD_PATH),
            ('HGPT2', HGPT2_PATH),
            ('GNSS', GNSS_PATH)
        ]:
            year_path = os.path.join(folder_path, str(year))
            exists = os.path.exists(year_path)
            status = "✓" if exists else "✗"
            print(f"  {status} {folder_name}/{year}: {year_path}")

            if exists:
                try:
                    all_items = os.listdir(year_path)
                    files = [f for f in all_items if os.path.isfile(os.path.join(year_path, f))]
                    print(f"     文件数: {len(files)}")
                    if len(files) > 0:
                        # 显示前5个文件
                        print(f"     示例文件:")
                        for f in files[:5]:
                            file_path = os.path.join(year_path, f)
                            size = os.path.getsize(file_path)
                            # 检查文件名格式: .txt扩展名也是可以的
                            if f.endswith(f"{year}.txt") or f.endswith(str(year)):
                                format_status = "✓"
                            else:
                                format_status = "✗ (格式错误)"
                            print(f"       {format_status} {f} ({size} bytes)")
                    else:
                        print(f"     ✗ 没有文件!")
                except Exception as e:
                    print(f"     错误: {e}")
    print()

def check_file_format():
    """检查文件格式"""
    print("="*60)
    print("文件格式检查")
    print("="*60)

    # 检查第一个可用的文件
    for year in YEARS:
        year_path = os.path.join(RES_ZTD_PATH, str(year))
        if os.path.exists(year_path):
            all_items = os.listdir(year_path)
            files = [f for f in all_items if os.path.isfile(os.path.join(year_path, f))]

            if files:
                # 找到第一个以年份结尾的文件(.txt也可以)
                test_file_name = None
                for f in files:
                    if f.endswith(f"{year}.txt") or f.endswith(str(year)):
                        test_file_name = f
                        break

                if not test_file_name:
                    print(f"✗ 警告: 没有找到以{year}或{year}.txt结尾的文件!")
                    print(f"  文件示例: {files[0]}")
                    test_file_name = files[0]

                test_file = os.path.join(year_path, test_file_name)
                print(f"测试文件: {test_file_name}")
                print(f"完整路径: {test_file}")
                print(f"文件大小: {os.path.getsize(test_file)} bytes")

                # 检查文件命名
                if test_file_name.endswith(f"{year}.txt"):
                    station_name = test_file_name[:-8]  # 移除year.txt
                    print(f"✓ 文件命名正确: 测站={station_name}, 年份={year}, 扩展名=.txt")
                elif test_file_name.endswith(str(year)):
                    station_name = test_file_name[:-4]  # 移除year
                    print(f"✓ 文件命名正确: 测站={station_name}, 年份={year}, 无扩展名")
                else:
                    print(f"✗ 文件命名错误: 应该以{year}或{year}.txt结尾")
                print()

                # 尝试读取
                print("尝试读取文件内容(前10行):")
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i >= 10:
                                break
                            print(f"  {i+1}: {line.rstrip()}")
                    print()
                except UnicodeDecodeError:
                    print("  UTF-8编码失败,尝试GBK...")
                    try:
                        with open(test_file, 'r', encoding='gbk') as f:
                            for i, line in enumerate(f):
                                if i >= 10:
                                    break
                                print(f"  {i+1}: {line.rstrip()}")
                        print()
                    except Exception as e2:
                        print(f"  GBK编码也失败: {e2}")
                except Exception as e:
                    print(f"  读取失败: {e}")
                    print()

                # 尝试用pandas读取
                print("尝试用pandas读取:")
                try:
                    df = pd.read_csv(test_file, sep=r'\s+', header=None,
                                   names=['year', 'doy', 'hour', 'minute', 'second', 'ztd'],
                                   skipinitialspace=True, engine='python', nrows=10)
                    print(df)
                    print(f"\n数据类型:\n{df.dtypes}")
                    print(f"\n✓ 成功! 可以正确读取数据")
                    print(f"  列数: {len(df.columns)}")
                    print(f"  行数: {len(df)}")

                    # 检查ZTD为0的情况
                    zero_count = (df['ztd'] == 0).sum()
                    nan_count = df['ztd'].isna().sum()
                    if zero_count > 0:
                        print(f"\n  警告: 有 {zero_count}/{len(df)} 行ZTD为0 (将被过滤)")
                    if nan_count > 0:
                        print(f"  警告: 有 {nan_count}/{len(df)} 行ZTD为NaN (将被过滤)")

                    valid_count = len(df) - zero_count - nan_count
                    print(f"  有效数据行数: {valid_count}/{len(df)}")

                    if valid_count == 0:
                        print(f"\n  ✗✗✗ 严重警告: 该文件没有有效的ZTD数据!")
                        print(f"  请检查数据文件是否正确!")

                except Exception as e:
                    print(f"✗ 失败: {e}")
                    print("\n可能的问题:")
                    print("  1. 列数不是6列")
                    print("  2. 分隔符不是空格")
                    print("  3. 包含非数字字符")
                print()
                break
        else:
            print(f"✗ 路径不存在: {year_path}")

def check_stations_file():
    """检查测站文件"""
    print("="*60)
    print("测站文件检查")
    print("="*60)

    if os.path.exists(STATIONS_FILE):
        print(f"✓ 文件存在: {STATIONS_FILE}")
        try:
            df = pd.read_excel(STATIONS_FILE)
            print(f"  测站数量: {len(df)}")
            print(f"  列名: {list(df.columns)}")

            # 检查必需的列
            required_cols = ['Station', 'Lon', 'Lat', 'Height']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  ✗ 缺少必需列: {missing_cols}")
            else:
                print(f"  ✓ 所有必需列都存在")

            print(f"\n前5个测站:")
            print(df.head())
        except Exception as e:
            print(f"✗ 读取失败: {e}")
    else:
        print(f"✗ 文件不存在: {STATIONS_FILE}")
    print()

def check_data_consistency():
    """检查数据一致性"""
    print("="*60)
    print("数据一致性检查")
    print("="*60)

    # 统计每个文件夹中的文件
    res_files = set()
    hgpt2_files = set()
    gnss_files = set()

    for year in YEARS:
        # RES_ZTD
        res_path = os.path.join(RES_ZTD_PATH, str(year))
        if os.path.exists(res_path):
            files = [f for f in os.listdir(res_path)
                    if os.path.isfile(os.path.join(res_path, f)) and
                    (f.endswith(f"{year}.txt") or f.endswith(str(year)))]
            if files:
                stations = [f[:-8] if f.endswith('.txt') else f[:-4] for f in files]
                res_files.update([f"{s}_{year}" for s in stations])

        # HGPT2
        hgpt2_path = os.path.join(HGPT2_PATH, str(year))
        if os.path.exists(hgpt2_path):
            files = [f for f in os.listdir(hgpt2_path)
                    if os.path.isfile(os.path.join(hgpt2_path, f)) and
                    (f.endswith(f"{year}.txt") or f.endswith(str(year)))]
            if files:
                stations = [f[:-8] if f.endswith('.txt') else f[:-4] for f in files]
                hgpt2_files.update([f"{s}_{year}" for s in stations])

        # GNSS
        gnss_path = os.path.join(GNSS_PATH, str(year))
        if os.path.exists(gnss_path):
            files = [f for f in os.listdir(gnss_path)
                    if os.path.isfile(os.path.join(gnss_path, f)) and
                    (f.endswith(f"{year}.txt") or f.endswith(str(year)))]
            if files:
                stations = [f[:-8] if f.endswith('.txt') else f[:-4] for f in files]
                gnss_files.update([f"{s}_{year}" for s in stations])

    print(f"RES_ZTD文件数: {len(res_files)}")
    print(f"HGPT2文件数: {len(hgpt2_files)}")
    print(f"GNSS文件数: {len(gnss_files)}")
    print()

    # 检查是否有缺失
    if res_files:
        missing_hgpt2 = res_files - hgpt2_files
        missing_gnss = res_files - gnss_files

        if missing_hgpt2:
            print(f"✗ 警告: {len(missing_hgpt2)} 个测站在HGPT2中缺失")
            if len(missing_hgpt2) <= 5:
                print(f"  示例: {list(missing_hgpt2)[:5]}")
        else:
            print(f"✓ 所有RES_ZTD测站在HGPT2中都有对应文件")

        if missing_gnss:
            print(f"✗ 警告: {len(missing_gnss)} 个测站在GNSS中缺失")
            if len(missing_gnss) <= 5:
                print(f"  示例: {list(missing_gnss)[:5]}")
        else:
            print(f"✓ 所有RES_ZTD测站在GNSS中都有对应文件")

    print()

def main():
    """主诊断函数"""
    print("\n" + "="*60)
    print("陆态网数据诊断工具")
    print("="*60 + "\n")

    check_directory_structure()
    check_stations_file()
    check_year_folders()
    check_file_format()
    check_data_consistency()

    print("="*60)
    print("诊断完成!")
    print("="*60)
    print("\n如果发现问题,请按照提示修复后再运行 main.py")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()