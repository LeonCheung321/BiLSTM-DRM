# 陆态网ZTD缺失值内插实验

基于BiLSTM的对流层延迟(ZTD)缺失值智能内插系统

## 项目结构

```
project/
│
├── config.py              # 配置文件(所有超参数)
├── data_loader.py         # 数据读取模块
├── preprocessor.py        # 数据预处理和特征工程
├── model.py              # BiLSTM模型定义
├── trainer.py            # 模型训练模块
├── tester.py             # 测试和评估模块(完全重写)
├── utils.py              # 工具函数
├── diagnose.py           # 数据诊断工具(首次运行)
├── check_imports.py      # 导入检查工具
├── quick_test.py         # 快速测试脚本(推荐先运行)
├── compare_results.py    # 结果对比分析工具
├── visualize_correction.py  # 误差修正策略可视化
├── main.py               # 主程序入口
├── requirements.txt      # 依赖包列表
└── README.md             # 项目说明
```

## 重要改进 (v3.0)

### ✨ 核心算法升级

1. **时间特征编码改进**
   - 使用sin/cos编码替代线性归一化
   - 更好地捕捉时间的周期性特征
   - 年积日: sin/cos编码捕捉季节变化
   - 日内时: sin/cos编码捕捉日变化

2. **误差修正策略升级**
   - 线性 → 二次 → **指数衰减**
   - 使用 `exp(λ·t)` 权重分配
   - 更符合LSTM误差加速累积特性
   - λ参数可调节(默认0.1)

3. **模型架构扩展**
   - 新增 Transformer 模型支持
   - 可在 `config.py` 中选择: `MODEL_TYPE = 'BiLSTM'` 或 `'Transformer'`
   - Transformer特性: 
     - 多头注意力机制(8头)
     - 位置编码
     - 更强的长程依赖建模

4. **输出增强**
   - 每个测站保存两个文件:
     - `*_预测结果.txt`: ZTD预测(GNSS, HGPT2, 预测)
     - `*_残差结果.txt`: 残差预测(真实, 预测)

### 🔧 技术细节

**特征数量变化**: 10维 → 12维
- 原: ZTD(1) + 时间(2) + 位置(3) + 空间(4) = 10
- 新: ZTD(1) + 时间sin/cos(4) + 位置(3) + 空间(4) = 12

## 环境要求

```bash
Python >= 3.8
PyTorch >= 1.10 (CUDA版本)
pandas
numpy
scikit-learn
openpyxl
```

安装依赖:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn openpyxl
```

## 数据目录结构

```
C:\Users\73955\Desktop\data_figs\Data\陆态网插值\
│
├── stations.xlsx          # 测站信息(经纬度、高程)
├── RES_ZTD/              # 残差ZTD数据
│   ├── 2020/
│   └── 2021/
├── HGPT2/                # HGPT2模型ZTD
│   ├── 2020/
│   └── 2021/
├── GNSS/                 # 真实GNSS ZTD
│   ├── 2020/
│   └── 2021/
└── results/              # 输出结果(自动创建)
    ├── 测试站结果/       # 未参与训练的测试站
    │   ├── 短时缺失_12h_ahaq_预测结果.txt
    │   ├── 短时缺失_12h_评估指标.txt
    │   └── ...
    └── 训练站结果/       # 参与训练的站点
        ├── 短时缺失_12h_ahbb_预测结果.txt
        ├── 短时缺失_12h_评估指标.txt
        └── ...
```

## 使用方法

### 步骤0: 快速测试(强烈推荐!)

```bash
# 用少量数据快速测试整个流程(5-10分钟)
python quick_test.py
```

这会:
- 只处理15个测站
- 只训练5轮
- 只测试24小时缺失
- 验证整个流程是否正常

### 步骤1: 检查环境

```bash
# 检查所有模块是否正确安装和导入
python check_imports.py
```

如果有导入错误,先解决依赖问题。

### 步骤1: 数据诊断(首次运行推荐)

```bash
python diagnose.py
```

这个诊断工具会检查:
- 所有目录是否存在
- 文件数量和格式
- 能否正确读取数据
- 测站信息是否正确
- 数据一致性

### 步骤2: 配置参数

编辑 `config.py` 调整超参数和路径:

```python
# 路径配置(根据你的实际路径修改)
BASE_PATH = r"C:\Users\73955\Desktop\data_figs\Data\陆态网插值"

# 模型参数
HIDDEN_SIZE = 128        # LSTM隐藏层大小
NUM_LAYERS = 2          # LSTM层数
DROPOUT = 0.2           # Dropout率

# 训练参数
BATCH_SIZE = 64         # 批次大小
LEARNING_RATE = 0.001   # 学习率
NUM_EPOCHS = 100        # 训练轮数
```

### 步骤4: 查看结果

```bash
python main.py
```

### 步骤5: 对比分析(可选)

```bash
# 对比训练站和测试站的性能差异
python compare_results.py
```

这会生成:
- 控制台输出: 训练站vs测试站对比表格
- `性能对比总结.txt`: 详细分析报告

结果保存在 `results/` 目录:

- `experiment_log.txt` - 实验日志
- `training_losses.txt` - 训练损失曲线
- `bilstm_model.pth` - 训练好的模型
- `短时缺失_12h_预测结果.txt` - 各实验预测结果
- `短时缺失_12h_评估指标.txt` - 各实验评估指标

## 实验设计

### 缺失场景

1. **短时缺失**: 12小时、24小时
2. **中短时缺失**: 3天(72小时)
3. **中时缺失**: 5天(120小时)
4. **中长时缺失**: 7天(168小时)
5. **长时缺失**: 10天(240小时)
6. **超长时缺失**: 15天(360小时)

### 特征工程

**输入特征(10维)**:
1. ZTD残差
2. 年积日(归一化)
3. 日内时(归一化)
4. 经度(归一化)
5. 纬度(归一化)
6. 高程(归一化)
7. 邻近站点平均ZTD
8. 空间梯度
9. 有效邻近站点数
10. 邻近站点ZTD范围

### 预测策略

**前向预测 + 后向修正**:
1. 使用历史3倍时长的数据进行滚动预测
2. 预测到缺失区间末端时计算累积误差
3. 线性分配误差进行后向修正

### 评估指标

- **RMSE** (均方根误差)
- **MAE** (平均绝对误差)
- **Bias** (偏差)
- **R²** (决定系数)

## 模型架构

```
输入 (batch, seq_len, 10)
    ↓
BiLSTM层 (2层, hidden=128)
    ↓
全连接层1 (256 → 128)
    ↓
ReLU + Dropout
    ↓
全连接层2 (128 → 64)
    ↓
ReLU + Dropout
    ↓
全连接层3 (64 → 1)
    ↓
输出 (ZTD预测值)
```

## 关键特性

✅ 双向LSTM捕获时序依赖
✅ 空间相关性特征提取
✅ 前向预测+后向误差修正
✅ 滚动窗口滑动预测
✅ 早停机制防止过拟合
✅ 学习率自适应调整
✅ GPU加速训练
✅ 模块化代码结构

## 输出说明

### 预测结果文件格式

```
年份  年积日  日内时  真实ZTD  预测ZTD
2020  100    12     2450.23  2448.56
2020  100    13     2451.12  2449.87
...
```

### 评估指标文件格式

```
实验: 短时缺失_12h
============================================================
平均RMSE: 15.2345
平均MAE: 12.3456
平均Bias: -0.1234
平均R²: 0.9567
测试样本数: 1200
```

## 注意事项

1. **CUDA要求**: 确保安装正确的PyTorch CUDA版本
2. **内存管理**: 大数据集可能需要调整batch_size
3. **计算时间**: 完整实验可能需要数小时
4. **数据质量**: 自动过滤ZTD=0的异常值
5. **路径配置**: 在config.py中修改数据路径

## 故障排除

### 导入错误

**问题**: `ImportError: cannot import name 'create_data_loaders'`

**解决**:
```bash
# 运行导入检查工具
python check_imports.py
```

确保所有模块都正确复制到项目文件夹,没有语法错误。

### 数据读取失败

**问题**: "成功读取 0 个测站的数据"

**解决步骤**:
1. 运行诊断工具查看问题:
   ```bash
   python diagnose.py
   ```

2. 检查文件命名格式:
   - 正确: `ahaq2020`, `ynsm2021` (测站名+年份,无扩展名)
   - 错误: `ahaq2020.txt`, `ahaq_2020`

3. 检查文件内容格式(空格分隔):
   ```
   2020 2 14 0 0 78.853
   2020 2 15 0 0 84.125
   ```

4. 检查文件编码:
   - 应该是UTF-8或ASCII
   - 不应包含BOM或特殊字符

5. 验证路径:
   - 在`config.py`中检查`BASE_PATH`是否正确
   - 确保路径使用双反斜杠`\\`或原始字符串`r"..."`

### CUDA不可用
```python
# 检查CUDA
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

### 内存不足
降低batch_size或hidden_size:
```python
BATCH_SIZE = 32  # 改为32
HIDDEN_SIZE = 64  # 改为64
```

### 收敛慢
调整学习率和优化器参数:
```python
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
```

## 作者

陆态网ZTD缺失值内插研究项目

## 许可

仅供学术研究使用