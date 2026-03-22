"""
模型训练模块
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import Config
from model import BiLSTMModel, EarlyStopping


class ZTDDataset(Dataset):
    """ZTD数据集"""
    def __init__(self, data, seq_len=24):
        self.data = data
        self.seq_len = seq_len
        self.samples = self._create_sequences()

    def _create_sequences(self):
        """创建序列样本"""
        samples = []

        # 按测站分组
        station_data = {}
        for item in self.data:
            station = item['station']
            if station not in station_data:
                station_data[station] = []
            station_data[station].append(item)

        # 为每个测站创建序列
        for station, items in station_data.items():
            # 按时间排序
            items.sort(key=lambda x: (x['year'], x['doy'], x['hour']))

            # 创建滑动窗口序列
            for i in range(len(items) - self.seq_len):
                seq = items[i:i+self.seq_len+1]
                X = np.array([s['features'] for s in seq[:-1]])
                y = seq[-1]['features'][0]  # 下一个时间步的ZTD
                samples.append((X, y, seq[-1]))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y, meta = self.samples[idx]
        return torch.FloatTensor(X), torch.FloatTensor([y]), meta

class Trainer:
    def __init__(self, model, train_loader, val_loader, device=Config.DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=Config.LR_FACTOR,
            patience=Config.LR_PATIENCE,
            verbose=True
        )

        self.early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            verbose=Config.VERBOSE
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        n_batches = 0

        for X_batch, y_batch, _ in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).squeeze()

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # 调试信息: 打印第一个epoch的数据范围
        if len(self.train_losses) == 0:
            print(f"\n  训练数据检查:")
            print(f"    输入范围: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
            print(f"    标签范围: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
            print(f"    预测范围: [{outputs.min():.4f}, {outputs.max():.4f}]\n")

        return avg_loss

    def validate(self):
        """验证"""
        self.model.eval()
        epoch_loss = 0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch, _ in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze()

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                epoch_loss += loss.item()
                n_batches += 1

        avg_loss = epoch_loss / n_batches
        return avg_loss

    def train(self):
        """完整训练流程"""
        print("开始训练...")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if Config.VERBOSE and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], '
                      f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # 学习率调度
            self.scheduler.step(val_loss)

            # 早停
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 加载最佳模型
        if self.early_stopping.best_model is not None:
            self.model.load_state_dict(self.early_stopping.best_model)

        print("训练完成!")
        return self.train_losses, self.val_losses

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"模型已保存至 {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"模型已从 {path} 加载")


def create_data_loaders(train_data, val_data, seq_len=24):
    """创建数据加载器"""
    train_dataset = ZTDDataset(train_data, seq_len)
    val_dataset = ZTDDataset(val_data, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Windows下设为0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader