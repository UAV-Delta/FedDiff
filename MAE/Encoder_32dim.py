import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# from MAE_model import DemandMAE
from MAE_32dim import DemandMAE

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

def get_data_path(filename, subfolder=""):
    """获取数据文件的相对路径"""
    if subfolder:
        return os.path.join(base_dir, subfolder, filename)
    return os.path.join(base_dir, filename)


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

    # 计算周数 (39周)
    num_weeks = 39
    weekly_data = np.zeros((1, num_stations, 168))

    # 按周组织数据

    weekly_data[0] = demand_data.T

    return weekly_data, station_ids

# 新增函数：加载知识图谱特征向量
def load_feature_vectors(file_path, num_stations):
    """
    从文本文件加载时空特征向量
    返回形状为 (num_stations, 4) 的numpy数组
    """
    features = []
    with open(file_path, 'r') as f:
        for line in f:
            # 解析每行的4个浮点数
            values = [float(x) for x in line.split()]
            if len(values) == 32:
                features.append(values)

    features = np.array(features)
    features = features[0:30]

    # 确保特征数量与换电站数量一致
    if len(features) != num_stations:
        raise ValueError(f"特征数量({len(features)})与换电站数量({num_stations})不匹配")

    return features

def encode_demand(model, demand, features, device):
    """
    demand: numpy (168,), features: numpy (4,)
    返回 latent: numpy (latent_dim,)
    """
    model.eval()
    with torch.no_grad():
        d = torch.tensor(demand, dtype=torch.float32, device=device).unsqueeze(0)
        f = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        _, latent = model(d, f)
    return latent.cpu().numpy().squeeze(0)


def main(excel_file, feature_file, model_path, output_path):
    weekly_data, station_ids = load_and_preprocess_data(excel_file)
    num_weeks, num_stations, seq_len = weekly_data.shape
    print(f"Loaded data: {num_weeks} weeks, {num_stations} stations, sequence length: {seq_len}")

    # 2. 加载知识图谱特征向量
    feature_data = load_feature_vectors(feature_file, num_stations)
    print(f"Loaded feature vectors: shape {feature_data.shape}")

    # 构建模型并加载参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandMAE(input_dim=seq_len, feature_dim=feature_data.shape[1],
                      latent_dim=128, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("模型参数加载完毕，开始计算嵌入...")

    # 准备输出数组
    embeddings = np.zeros((num_weeks, num_stations, 128), dtype=float)

    # 逐周、逐站点计算 latent
    for w in range(num_weeks):
        for s in range(num_stations):
            d = weekly_data[w, s]  # shape (168,)
            f = feature_data[s]  # shape (4,)
            embeddings[w, s] = encode_demand(model, d, f, device)
        print(f"Week {w + 1}/{num_weeks} embeddings done")

    # 保存到 .npy
    # output_path="D:\\code\\python\\Infocom2025\\data\\demand_embeddings\\shanghai_demand_embeddings.npy"
    np.save(output_path, embeddings)
    print(f"All embeddings saved to {output_path} with shape {embeddings.shape}")

# 6. 主流程
if __name__ == "__main__":
    main(get_data_path("nanchong_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_nanchong_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_nanchong_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("yichang_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_yichang_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_yichang_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("deyang_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_deyang_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_deyang_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("hohhot_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_Hohhot_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_Hohhot_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("beijing_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_beijing_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_beijing_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("nanjing_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_nanjing_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_nanjing_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("dazhou_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_dazhou_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_dazhou_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("suining_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_suining_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_suining_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("chengdu_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_chengdu_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_chengdu_demand_embeddings.npy", "demand_embeddings"))

    main(get_data_path("shanghai_hourly_demand.xlsx", "battery swap datasets"),
         get_data_path("32_shanghai_station_feature.txt", "ukg_embeddings"),
         "32_mae_model.pth",
         get_data_path("32_shanghai_demand_embeddings.npy", "demand_embeddings"))



