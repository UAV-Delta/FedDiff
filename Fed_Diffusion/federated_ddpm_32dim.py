import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 定义数据集类
class DemandDataset(Dataset):
    """加载需求嵌入特征 (39*n,128) 和空间特征 (39*n,4) 的数据集"""

    def __init__(self, demand_npy, spatial_txt):
        arr = np.load(demand_npy)  # (39, n, 128)
        weeks, n, feat_dim = arr.shape
        self.demand = torch.from_numpy(arr.reshape(weeks * n, feat_dim)).float()

        spatial = np.loadtxt(spatial_txt)  # (n, 4)
        spatial_rep = np.tile(spatial[0:30], (weeks, 1))  # (39*n, 4)
        self.spatial = torch.from_numpy(spatial_rep).float()

        assert self.demand.shape[0] == self.spatial.shape[0], \
            f"样本数不匹配: {self.demand.shape[0]} vs {self.spatial.shape[0]}"

    def __len__(self):
        return self.demand.shape[0]

    def __getitem__(self, idx):
        return self.demand[idx], self.spatial[idx]

    def get_sample_count(self):
        """获取样本数量（用于联邦学习的加权聚合）"""
        return len(self)


# 2. 定义模型组件（与原始代码相同）
class TimeEmbedding(nn.Module):
    """Sinusoidal 时间步嵌入"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=device) / (half - 1)
        )  # (half,)
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class ConditionedBlock(nn.Module):
    """单层条件块，融合时间与空间特征"""

    def __init__(self, hidden_dim, time_dim, cond_dim=32):
        super().__init__()
        self.proj_x = nn.Linear(hidden_dim, hidden_dim)
        self.proj_t = nn.Linear(time_dim, hidden_dim)
        self.proj_cond = nn.Linear(cond_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x, t_emb, cond):
        h = self.proj_x(x) + self.proj_t(t_emb) + self.proj_cond(cond)
        h = self.norm(h)
        return self.act(h)


class DDPM(nn.Module):
    """条件扩散模型"""

    def __init__(self, num_steps=500, feature_dim=128, time_dim=128, hidden_dim=512, cond_dim=32):
        super().__init__()
        self.num_steps = num_steps
        self.time_emb = TimeEmbedding(time_dim)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ConditionedBlock(hidden_dim, time_dim, cond_dim=cond_dim)
            for _ in range(4)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self._setup_diffusion()

    def _setup_diffusion(self):
        betas = torch.linspace(1e-4, 0.01, self.num_steps, device=device)
        alphas = 1 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cum = alphas_cum
        self.alphas_cum_prev = F.pad(alphas_cum[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cum = torch.sqrt(alphas_cum)
        self.sqrt_one_minus_alphas_cum = torch.sqrt(1 - alphas_cum)
        self.sqrt_recip_alphas_cum = torch.sqrt(1. / alphas_cum)

        post_var = betas * (1 - self.alphas_cum_prev) / (1 - alphas_cum)
        self.post_var = post_var
        self.post_log_var = torch.log(torch.clamp(post_var, min=1e-20))

    def forward(self, x, t, cond):
        t_emb = self.time_emb(t)  # [B, time_dim]
        h = self.input_proj(x)  # [B, hidden]
        for blk in self.blocks:
            h = blk(h, t_emb, cond)
        return self.output_proj(h)  # [B, 128]

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_t = extract(self.sqrt_alphas_cum, t, x0.shape)
        bm = extract(self.sqrt_one_minus_alphas_cum, t, x0.shape)
        return a_t * x0 + bm * noise

    def p_losses(self, x0, t, cond):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self(x_t, t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddim_sample(self, cond, n_steps=None):
        """
        使用 DDIM 采样（无噪声注入），返回 (M, feature_dim)
        """
        M = cond.shape[0]
        n_steps = self.num_steps if n_steps is None else n_steps
        x = torch.randn(M, 128, device=device)

        # 逆向一步步去噪
        for i in reversed(range(n_steps)):
            t = torch.full((M,), i, device=device, dtype=torch.long)
            beta_t = extract(self.betas, t, x.shape)
            bm = extract(self.sqrt_one_minus_alphas_cum, t, x.shape).clamp(min=1e-5)
            ir = extract(self.sqrt_recip_alphas_cum, t, x.shape)

            # 预测噪声
            pred_noise = self(x, t, cond)
            # 直接用 DDIM 公式去掉噪声项
            x0_pred = (x - bm * pred_noise) / ir
            x = extract(self.sqrt_alphas_cum, t, x.shape) * x0_pred

        return x


def extract(a, t, shape):
    batch = t.shape[0]
    out = a.gather(0, t)  # [B]
    return out.view(batch, *([1] * (len(shape) - 1))).to(t.device)

    # 3. 联邦学习框架


class FederatedDDPM:
    def __init__(self, city_configs, num_steps=500, local_epochs=300, communication_rounds=10, cond_dim=32):
        """
        city_configs: 城市配置列表，每个元素为元组 (city_name, demand_npy_path, spatial_txt_path)
        num_steps: 扩散步数
        local_epochs: 每次通信前的本地训练轮数
        communication_rounds: 总通信轮数
        """
        self.city_configs = city_configs
        self.num_cities = len(city_configs)
        self.num_steps = num_steps
        self.local_epochs = local_epochs
        self.communication_rounds = communication_rounds
        self.cond_dim = cond_dim

        # 初始化客户端
        self.clients = []
        for config in city_configs:
            city_name, demand_npy, spatial_txt = config
            client = {
                'name': city_name,
                'dataset': DemandDataset(demand_npy, spatial_txt),
                'sample_count': None,  # 将在初始化时填充
                'model': DDPM(num_steps=num_steps, cond_dim=cond_dim).to(device),
                'optimizer': None,  # 将在训练时初始化
                'best_loss': float('inf'),
                'consecutive_misses': 0,  # 连续未被选中的次数
                'active': True,  # 是否参与训练
                'loss_history': []  # 记录每轮训练损失
            }
            client['sample_count'] = client['dataset'].get_sample_count()
            self.clients.append(client)

        # 记录联邦学习过程
        self.final_global_model = None
        self.global_loss_history = []
        self.participation_history = {client['name']: [] for client in self.clients}

    def train_local(self, client_idx, epochs):
        """在单个客户端上训练模型"""
        client = self.clients[client_idx]
        if not client['active']:
            return float('inf')

        # 初始化优化器
        if client['optimizer'] is None:
            client['optimizer'] = torch.optim.Adam(client['model'].parameters(), lr=1e-3)

        # 创建数据加载器
        dataloader = DataLoader(
            client['dataset'],
            batch_size=256,
            shuffle=True,
            pin_memory=True
        )

        model = client['model']
        optimizer = client['optimizer']

        model.train()
        best_loss = float('inf')
        epoch_losses = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for x0, cond in dataloader:
                x0, cond = x0.to(device), cond.to(device)
                t = torch.randint(0, self.num_steps, (x0.size(0),), device=device)

                optimizer.zero_grad()
                loss = model.p_losses(x0, t, cond)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)

            # 更新最佳损失
            if avg_loss < best_loss:
                best_loss = avg_loss

            client['loss_history'].extend(epoch_losses)
            if best_loss < client['best_loss']:
                client['best_loss'] = best_loss

        return best_loss

    def aggregate_models(self, selected_indices):
        """聚合选中的客户端模型"""
        # 计算总样本数和权重
        total_samples = sum(self.clients[i]['sample_count'] for i in selected_indices)
        weights = [self.clients[i]['sample_count'] / total_samples for i in selected_indices]

        # 创建全局模型
        global_model = DDPM(num_steps=self.num_steps, cond_dim=self.cond_dim).to(device)
        global_state_dict = global_model.state_dict()

        # 初始化聚合参数
        for key in global_state_dict:
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])

        # 加权聚合参数
        for idx, weight in zip(selected_indices, weights):
            client_state_dict = self.clients[idx]['model'].state_dict()
            for key in global_state_dict:
                global_state_dict[key] += weight * client_state_dict[key]

        # 更新全局模型
        global_model.load_state_dict(global_state_dict)
        return global_model

    def distribute_model(self, global_model):
        """分发全局模型到所有活跃客户端"""
        global_state_dict = global_model.state_dict()
        for client in self.clients:
            if client['active']:
                client['model'].load_state_dict(copy.deepcopy(global_state_dict))

    def run_federated_learning(self):
        """运行联邦学习过程"""
        print(f"开始联邦学习: {self.num_cities}个城市, {self.communication_rounds}轮通信")
        print(f"每轮本地训练: {self.local_epochs}个epoch")

        active_clients = [i for i, c in enumerate(self.clients) if c['active']]

        for round_idx in range(1, self.communication_rounds + 1):
            print(f"\n===== 通信轮次 [{round_idx}/{self.communication_rounds}] =====")

            # 1. 本地训练
            round_losses = []
            for client_idx in active_clients:
                client = self.clients[client_idx]
                print(f"城市 {client['name']} 本地训练中...")
                best_loss = self.train_local(client_idx, self.local_epochs)
                round_losses.append(best_loss)
                print(f"  完成! 最佳损失: {best_loss:.6f}")

            # 2. 计算平均损失并选择参与者
            avg_loss = np.mean(round_losses)
            self.global_loss_history.append(avg_loss)

            # 选择损失低于平均值的客户端
            selected_indices = []
            for i, client_idx in enumerate(active_clients):
                client = self.clients[client_idx]
                if client['best_loss'] <= avg_loss:
                    selected_indices.append(client_idx)
                    client['consecutive_misses'] = 0
                    self.participation_history[client['name']].append(round_idx)
                else:
                    client['consecutive_misses'] += 1
                    # 检查是否应停用客户端
                    if client['consecutive_misses'] >= 5:
                        client['active'] = False
                        print(f"城市 {client['name']} 已停用 (连续5轮未参与)")

            # 3. 聚合模型
            if selected_indices:
                print(f"聚合模型: 参与城市 {len(selected_indices)}个")
                global_model = self.aggregate_models(selected_indices)
                self.distribute_model(global_model)
                self.final_global_model = global_model

            # 4. 更新活跃客户端列表
            active_clients = [i for i, c in enumerate(self.clients) if c['active']]
            if not active_clients:
                print("所有客户端均已停用，提前终止训练")
                break

            print(f"本轮平均损失: {avg_loss:.6f}, 活跃客户端: {len(active_clients)}个")

        print("\n联邦学习完成!")

    def save_models(self, output_dir="32_federated_models_1"):
        """保存所有客户端模型"""
        os.makedirs(output_dir, exist_ok=True)
        for client in self.clients:
            if client['active'] or client['loss_history']:  # 保存所有训练过的模型
                path = os.path.join(output_dir, f"{client['name']}_ddpm.pth")
                torch.save(client['model'].state_dict(), path)
                print(f"保存模型: {path}")

        # 保存最终聚合得到的全局模型
        if hasattr(self, "final_global_model"):
            global_path = os.path.join(output_dir, "final_global_model.pth")
            torch.save(self.final_global_model.state_dict(), global_path)
            print(f"保存最终聚合模型: {global_path}")


current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

# 4. 主执行函数
def main():
    # 城市配置
    city_configs = [
        ("beijing",
         os.path.join(base_dir, "demand_embeddings", "32_beijing_demand_embeddings.npy"),
         os.path.join(base_dir, "ukg_embeddings", "32_beijing_station_feature.txt")),

        ("nanjing",
         os.path.join(base_dir, "demand_embeddings", "32_nanjing_demand_embeddings.npy"),
         os.path.join(base_dir, "ukg_embeddings", "32_nanjing_station_feature.txt")),

        ("nanchong",
         os.path.join(base_dir, "demand_embeddings", "32_nanchong_demand_embeddings.npy"),
         os.path.join(base_dir, "ukg_embeddings", "32_nanchong_station_feature.txt")),

        ("yichang",
         os.path.join(base_dir, "demand_embeddings", "32_yichang_demand_embeddings.npy"),
         os.path.join(base_dir, "ukg_embeddings", "32_yichang_station_feature.txt")),

        ("deyang",
         os.path.join(base_dir, "demand_embeddings", "32_deyang_demand_embeddings.npy"),
         os.path.join(base_dir, "ukg_embeddings", "32_deyang_station_feature.txt")),

        ("Hohhot",
         os.path.join(base_dir, "demand_embeddings", "32_Hohhot_demand_embeddings.npy"),
         os.path.join(base_dir, "ukg_embeddings", "32_Hohhot_station_feature.txt"))
    ]

    # 初始化联邦学习系统
    federated_system = FederatedDDPM(
        city_configs=city_configs,
        num_steps=500,
        local_epochs=500,  # 每轮本地训练500次
        communication_rounds=10,  # 共10轮通信
        cond_dim=32
    )

    # 运行联邦学习
    federated_system.run_federated_learning()

    # 保存模型并可视化结果
    federated_system.save_models()
    # federated_system.plot_training_history()


if __name__ == "__main__":
    main()