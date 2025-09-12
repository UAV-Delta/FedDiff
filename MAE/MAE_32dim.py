import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


# 1. 数据加载与预处理函数
def load_and_preprocess_data(file_path):
    """
    从Excel文件加载并预处理换电站需求数据
    返回形状为 (num_weeks, num_stations, 168) 的张量
    """
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=0)

    # 获取时间戳和换电站ID
    timestamps = df.iloc[:, 0].values
    # print(timestamps)
    station_ids = df.columns[1:31].get_level_values(0).tolist()
    num_stations = len(station_ids)

    # 提取需求数据 (去除时间戳列)
    demand_data = df.iloc[0:168, 1:31].values.astype(float)

    # # 计算周数 (39周)
    # num_weeks = 39
    weekly_data = np.zeros((1, num_stations, 168))

    # 按周组织数据


    weekly_data[0] = demand_data.T

    return weekly_data, station_ids


# 新增函数：加载知识图谱特征向量
def load_feature_vectors(file_path, num_stations):
    """
    从文本文件加载时空特征向量（支持32维）
    返回形状为 (num_stations, 32) 的numpy数组
    """
    features = []
    with open(file_path, 'r') as f:
        for line in f:
            # 解析每行的32个浮点数
            values = [float(x) for x in line.split()]
            features.append(values)

    features = np.array(features)
    features = features[0:30]

    # 确保特征数量与换电站数量一致
    if len(features) != num_stations:
        raise ValueError(f"特征数量({len(features)})与换电站数量({num_stations})不匹配")

    # 确保特征维度是32
    if features.shape[1] != 32:
        raise ValueError(f"特征维度({features.shape[1]})应为32")

    return features


# 2. 掩码策略模块
class MaskingStrategies:
    @staticmethod
    def random_mask(data, mask_ratio=0.3    ):
        """随机掩码策略"""
        mask = np.random.rand(*data.shape) > mask_ratio
        masked_data = data.copy()
        masked_data[~mask] = -1
        return masked_data, mask

    @staticmethod
    def block_mask(data, block_size=24, num_blocks=3):
        """块状掩码策略 (掩码连续时间段)"""
        masked_data = data.copy()
        mask = np.ones_like(data, dtype=bool)

        for i in range(data.shape[0]):  # 遍历每个换电站
            for _ in range(num_blocks):  # 每个站掩码num_blocks个块
                start = np.random.randint(0, 168 - block_size)
                end = start + block_size
                masked_data[i, start:end] = -1
                mask[i, start:end] = False

        return masked_data, mask

    @staticmethod
    def station_mask(data, station_ratio=0.3):
        """换电站掩码策略 (掩码整个换电站)"""
        num_stations = data.shape[0]
        num_mask = int(num_stations * station_ratio)

        masked_data = data.copy()
        mask = np.ones_like(data, dtype=bool)

        mask_stations = np.random.choice(num_stations, num_mask, replace=False)
        masked_data[mask_stations] = -1
        mask[mask_stations] = False

        return masked_data, mask

    @staticmethod
    def time_slot_mask(data, time_slots=[(8, 12), (18, 22)], mask_ratio=0.8):
        """时间段掩码策略 (掩码特定时间段)"""
        masked_data = data.copy()
        mask = np.ones_like(data, dtype=bool)

        for start, end in time_slots:
            # 创建时间段掩码
            time_mask = np.zeros(168, dtype=bool)
            for day in range(7):
                time_mask[day * 24 + start: day * 24 + end] = True

            # 应用掩码
            mask_indices = np.where(time_mask)[0]
            num_mask = int(len(mask_indices) * mask_ratio)
            mask_stations = np.random.choice(data.shape[0], num_mask, replace=False)

            masked_data[mask_stations[:, None], mask_indices] = -1
            mask[mask_stations[:, None], mask_indices] = False

        return masked_data, mask

    @staticmethod
    def apply_all_strategies(data):
        """应用所有四种掩码策略"""
        results = {}

        # 应用四种掩码策略
        results['random'], _ = MaskingStrategies.random_mask(data)
        results['block'], _ = MaskingStrategies.block_mask(data)
        results['station'], _ = MaskingStrategies.station_mask(data)
        results['time_slot'], _ = MaskingStrategies.time_slot_mask(data)

        # 添加原始数据
        results['original'] = data

        return results


# 3. 数据集类
class ChargingStationDataset(Dataset):
    def __init__(self, weekly_data, feature_data, mask_strategy='random'):
        """
        weekly_data: 周数据张量 (num_weeks, num_stations, 168)
        feature_data: 时空特征张量 (num_stations, 32)  # 维度变为32
        mask_strategy: 掩码策略名称
        """
        self.num_weeks, self.num_stations, self.seq_len = weekly_data.shape
        self.weekly_data = weekly_data
        self.mask_strategy = mask_strategy
        self.feature_data = feature_data  # 直接使用传入的特征数据

        # 标准化特征
        self.scaler = StandardScaler()
        self.feature_data = self.scaler.fit_transform(self.feature_data)

    def __len__(self):
        return self.num_weeks * self.num_stations

    def __getitem__(self, idx):
        # 计算周索引和换电站索引
        week_idx = idx // self.num_stations
        station_idx = idx % self.num_stations

        # 获取原始需求序列
        original_demand = self.weekly_data[week_idx, station_idx]

        # 获取特征
        features = self.feature_data[station_idx]

        # 应用掩码策略
        if self.mask_strategy == 'random':
            masked_demand, mask = MaskingStrategies.random_mask(original_demand.reshape(1, -1))
        elif self.mask_strategy == 'block':
            masked_demand, mask = MaskingStrategies.block_mask(original_demand.reshape(1, -1))
        elif self.mask_strategy == 'station':
            masked_demand, mask = MaskingStrategies.station_mask(original_demand.reshape(1, -1))
        elif self.mask_strategy == 'time_slot':
            masked_demand, mask = MaskingStrategies.time_slot_mask(original_demand.reshape(1, -1))
        else:
            # 默认使用随机掩码
            masked_demand, mask = MaskingStrategies.random_mask(original_demand.reshape(1, -1))

        # 转换为1D数组
        masked_demand = masked_demand.flatten()
        mask = mask.flatten()

        return {
            'masked_demand': torch.tensor(masked_demand, dtype=torch.float32),
            'features': torch.tensor(features, dtype=torch.float32),
            'original_demand': torch.tensor(original_demand, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }


# 4. MAE模型架构
class DemandMAE(nn.Module):
    def __init__(self, input_dim=168, feature_dim=32, latent_dim=64, hidden_dim=256):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, masked_demand, features):
        # 拼接掩码需求与特征
        x = torch.cat([masked_demand, features], dim=1)

        # 编码过程
        latent = self.encoder(x)

        # 解码过程 (拼接潜在向量与特征)
        z = torch.cat([latent, features], dim=1)
        reconstructed = self.decoder(z)

        return reconstructed, latent


# 5. 训练函数
def train_model(model, dataloaders, val_loader, epochs=100, lr=0.001, save_path="best_mae_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_train_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        # 循环所有掩码策略的数据加载器
        for strategy_name, train_loader in dataloaders.items():
            if strategy_name == 'original':  # 跳过原始数据
                continue

            for batch in train_loader:
                masked_demand = batch['masked_demand'].to(device)
                features = batch['features'].to(device)
                original_demand = batch['original_demand'].to(device)
                mask = batch['mask'].to(device)

                optimizer.zero_grad()

                # 前向传播
                reconstructed, _ = model(masked_demand, features)

                # 原来只计算 masked loss
                mask_loss = criterion(reconstructed[~mask], original_demand[~mask])

                # 新增 full loss
                full_loss = criterion(reconstructed, original_demand)

                # 总损失：加权和
                loss = 0.3 * mask_loss + 0.7 * full_loss

                # 反向传播
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

        # 计算平均训练损失
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0

        # 验证步骤
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                masked_demand = batch['masked_demand'].to(device)
                features = batch['features'].to(device)
                original_demand = batch['original_demand'].to(device)
                mask = batch['mask'].to(device)

                reconstructed, _ = model(masked_demand, features)
                # 原来只计算 masked loss
                mask_loss = criterion(reconstructed[~mask], original_demand[~mask])

                # 新增 full loss
                full_loss = criterion(reconstructed, original_demand)

                # 总损失：加权和
                loss = 0.3 * mask_loss + 0.7 * full_loss
                # loss = criterion(reconstructed[~mask], original_demand[~mask])
                val_loss += loss.item()

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 打印训练进度
        print(
            f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存最佳模型
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved with train loss: {best_train_loss:.4f}')

    print('Training completed!')
    return model



current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

# 6. 主流程
if __name__ == "__main__":
    # 1. 加载数据
    excel_file = os.path.join(base_dir, "battery swap datasets", "chengdu_hourly_demand.xlsx")
    weekly_data, station_ids = load_and_preprocess_data(excel_file)
    num_week, num_stations, seq_len = weekly_data.shape
    # print(f"Loaded data: {num_weeks} weeks, {num_stations} stations, sequence length: {seq_len}")

    # 2. 加载知识图谱特征向量
    feature_file = os.path.join(base_dir, "ukg_embeddings", "32_chengdu_station_feature.txt")
    feature_data = load_feature_vectors(feature_file, num_stations)
    # print(f"Loaded feature vectors: shape {feature_data.shape}")

    train_weekly_data = weekly_data

    # 4. 为每种掩码策略创建数据加载器
    strategies = ['random', 'random', 'random', 'random']
    train_dataloaders = {}
    batch_size = 32

    for strategy in strategies:
        ds = ChargingStationDataset(
            train_weekly_data,
            feature_data,
            mask_strategy=strategy
        )
        train_dataloaders[strategy] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )


    val_loader = train_dataloaders['random']  # 或者任意一个 loader

    # 6. 初始化模型 (注意特征维度改为4)
    model = DemandMAE(
        input_dim=seq_len,
        feature_dim=feature_data.shape[1],  # 特征维度改为4
        latent_dim=128,
        hidden_dim=256
    )

    # 7. 训练模型
    trained_model = train_model(
        model,
        train_dataloaders,
        val_loader,
        epochs=150,
        lr=0.001,
        save_path="32_mae_model_1.pth"
    )

