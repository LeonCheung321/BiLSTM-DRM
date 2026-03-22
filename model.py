"""
模型定义 - BiLSTM和Transformer网络
"""
import torch
import torch.nn as nn
import math
from config import Config


class PositionalEncoding(nn.Module):
    """位置编码 for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer模型用于ZTD预测"""
    def __init__(self):
        super(TransformerModel, self).__init__()

        self.input_size = Config.INPUT_SIZE
        self.d_model = Config.HIDDEN_SIZE
        self.nhead = 8  # 注意力头数
        self.num_layers = Config.NUM_LAYERS
        self.dropout = Config.DROPOUT

        # 输入投影
        self.input_projection = nn.Linear(self.input_size, self.d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=self.num_layers
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 取最后一个时间步
        x = x[:, -1, :]

        # 输出
        out = self.fc(x)

        return out.squeeze(-1)


class BiLSTMModel(nn.Module):
    def __init__(self):
        super(BiLSTMModel, self).__init__()

        self.hidden_size = Config.HIDDEN_SIZE
        self.num_layers = Config.NUM_LAYERS

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=Config.INPUT_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            batch_first=True,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0,
            bidirectional=Config.BIDIRECTIONAL
        )

        # 全连接层
        lstm_output_size = Config.HIDDEN_SIZE * 2 if Config.BIDIRECTIONAL else Config.HIDDEN_SIZE

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(lstm_output_size // 4, 1)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        # LSTM层
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 全连接层
        out = self.fc(last_output)

        return out.squeeze(-1)

    def predict_sequence(self, x, seq_len):
        """
        滚动预测序列
        x: (1, history_len, input_size) - 历史数据
        seq_len: 需要预测的长度
        """
        predictions = []
        current_seq = x.clone()

        for _ in range(seq_len):
            # 预测下一个时间步
            with torch.no_grad():
                pred = self.forward(current_seq)
                predictions.append(pred.item())

            # 更新序列（滑动窗口）
            # 创建新的输入：将预测值作为新的ZTD特征
            new_input = current_seq[:, -1:, :].clone()
            new_input[:, 0, 0] = pred  # 更新ZTD特征

            # 滑动窗口
            current_seq = torch.cat([current_seq[:, 1:, :], new_input], dim=1)

        return predictions

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=Config.EARLY_STOPPING_PATIENCE, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """保存模型"""
        self.best_model = model.state_dict().copy()