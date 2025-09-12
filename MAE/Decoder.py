"""
decode_embeddings.py

功能：
1. 读取 3D 嵌入特征 (x, n, 128) 的 .npy 文件
2. 读取 n×4 的空间特征 .txt 文件
3. 加载训练好的 best_mae_model.pth，并用其 **decoder** 把嵌入还原成
   (x, n, 168) 的一周需求序列
4. 按“168*x 行 × n 列” 展平，保存为 Excel 工作簿
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from MAE_model import DemandMAE
from MAE_32dim import DemandMAE

# --------- 2. 批量解码函数 ---------
@torch.no_grad()
def decode_latents(model,
                   latents: np.ndarray,
                   feats: np.ndarray,
                   batch_size: int = 4096,
                   device: str = "cpu") -> np.ndarray:
    """
    latents:  (N, 128)
    feats:    (N,   4)
    返回      (N, 168)
    """
    model.to(device).eval()
    dec = model.decoder  # 直接拿 decoder
    outs = []

    total = latents.shape[0]
    for st in range(0, total, batch_size):
        ed = min(st + batch_size, total)

        l_batch = torch.tensor(latents[st:ed], dtype=torch.float32, device=device)
        f_batch = torch.tensor(feats[st:ed],   dtype=torch.float32, device=device)

        z = torch.cat([l_batch, f_batch], dim=1)  # (B, 128+4)
        outs.append(dec(z).cpu().numpy())

    return np.vstack(outs)  # (N, 168)


# --------- 3. 主流程 ---------
def main(embeddings_npy, spatial_txt, mae_ckpt, excel_out):
    # # ======= 路径配置  =======
    # # embeddings_npy = "D:\\code\\python\\Infocom2025\\data\\demand_embeddings\\chengdu_demand_embeddings.npy"   # (x, n, 128)
    # embeddings_npy = "D:\\code\\python\\Infocom2025\\diffusion_model\\generated_demand_embeddings.npy"  # (x, n, 128)
    # spatial_txt    = "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\chengdu_station_feature.txt"       # (n,   4)
    # mae_ckpt       = "best_mae_model.pth"                # 训练好的 MAE
    # # excel_out      = "chengdu_decoded_demand.xlsx"               # 输出
    # excel_out = "chengdu_ddpm_decoded_demand.xlsx"

    # ======= 读取数据  =======
    emb = np.load(embeddings_npy)        # (x, n, 128)
    weeks, n, latent_dim = emb.shape
    spatial = np.loadtxt(spatial_txt)    # (n, 4)
    # spatial = spatial[0:30]
    assert spatial.shape[0] == n, "站点数量和空间特征不一致"

    # spatial = np.ones_like(spatial)  # 创建与原始空间特征形状相同的全1数组

    # print(spatial)

    # 复制空间特征, 展平成 (x*n, 4)
    spatial_rep = np.tile(spatial, (weeks, 1))
    latents_flat = emb.reshape(-1, latent_dim)           # (x*n, 128)

    # ======= 加载模型  =======
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mae = DemandMAE(
        input_dim=168,
        feature_dim=32,
        latent_dim=latent_dim,
        hidden_dim=256
    )
    mae.load_state_dict(torch.load(mae_ckpt, map_location=device))

    # ======= 解码  =======
    decoded_flat = decode_latents(
        mae,
        latents_flat,
        spatial_rep,
        batch_size=4096,
        device=device
    )                                                   # (x*n, 168)

    # ======= 重塑形状并写 Excel =======
    decoded = decoded_flat.reshape(weeks, n, 168)       # (x, n, 168)
    # 转成 (168*x, n)：把每周 168 行顺序拼接
    data_2d = decoded.transpose(1, 0, 2)                # (n, x, 168)
    data_2d = data_2d.reshape(n, weeks * 168).T         # (weeks*168, n)

    # 创建不带列名的 DataFrame
    df = pd.DataFrame(data_2d)

    # 保存
    df.to_excel(excel_out, index=False, header=False)
    print(f"已保存到 {excel_out}  (shape={df.shape})")


if __name__ == "__main__":


    # main("D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\ours\\32_chengdu_generated_demand_embeddings.npy",
    #      "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #      "32_mae_model.pth",
    #      "D:\\code\\python\\Infocom2025\\results\\ours\\chengdu_ddpm_decoded_demand.xlsx")
    #
    # main("D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\ours\\32_shanghai_generated_demand_embeddings.npy",
    #      "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #      "32_mae_model.pth",
    #      "D:\\code\\python\\Infocom2025\\results\\ours\\shanghai_ddpm_decoded_demand.xlsx")
    #
    # main("D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\ours\\32_dazhou_generated_demand_embeddings.npy",
    #      "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_dazhou_station_feature.txt",
    #      "32_mae_model.pth",
    #      "D:\\code\\python\\Infocom2025\\results\\ours\\dazhou_ddpm_decoded_demand.xlsx")
    #
    # main("D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\ours\\32_suining_generated_demand_embeddings.npy",
    #      "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_suining_station_feature.txt",
    #      "32_mae_model.pth",
    #      "D:\\code\\python\\Infocom2025\\results\\ours\\suining_ddpm_decoded_demand.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\pfedctp\\chengdu_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\chengdu_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\pfedctp\\chengdu_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\pfedctp\\shanghai_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\shanghai_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\pfedctp\\shanghai_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\pfedctp\\dazhou_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\dazhou_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\pfedctp\\dazhou_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\pfedctp\\suining_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\suining_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\pfedctp\\suining_ddpm_decoded_demand.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\fededm\\chengdu_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\chengdu_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\fededm\\chengdu_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\fededm\\shanghai_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\shanghai_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\fededm\\shanghai_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\fededm\\dazhou_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\dazhou_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\fededm\\dazhou_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\fededm\\suining_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\suining_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\fededm\\suining_ddpm_decoded_demand.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\w_o_u\\chengdu_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\chengdu_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\w_o_u\\chengdu_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\w_o_u\\shanghai_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\shanghai_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\w_o_u\\shanghai_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\w_o_u\\dazhou_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\dazhou_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\w_o_u\\dazhou_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\w_o_u\\suining_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\suining_station_feature.txt",
    #     "chengdu_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\w_o_u\\suining_ddpm_decoded_demand.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\test\\32_chengdu_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\test\\32_chengdu_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\test\\32_shanghai_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\test\\32_shanghai_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\test\\32_dazhou_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_dazhou_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\test\\32_dazhou_ddpm_decoded_demand.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\test\\32_suining_generated_demand_embeddings.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_suining_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\test\\32_suining_ddpm_decoded_demand.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings10.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand10.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings20.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand20.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings30.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand30.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings40.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand40.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings50.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand50.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings60.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand60.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings70.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand70.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings80.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand80.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\32_demand_embeddings\\32_chengdu_demand_embeddings90.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\data\\masked_demand_excel1\\32_chengdu_decoded_demand90.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_1.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_1.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_2.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_2.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_3.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_3.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_4.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_4.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_5.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_5.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_6.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_6.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_7.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_7.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_8.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_8.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_9.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_9.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\each_round\\32_shanghai_generated_demand_embeddings_10.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\each_round\\32_shanghai_ddpm_decoded_demand_10.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city\\32_chengdu_generated_demand_embeddings_2.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city\\32_chengdu_ddpm_decoded_demand_2.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city\\32_chengdu_generated_demand_embeddings_3.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city\\32_chengdu_ddpm_decoded_demand_3.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city\\32_chengdu_generated_demand_embeddings_4.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city\\32_chengdu_ddpm_decoded_demand_4.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city\\32_chengdu_generated_demand_embeddings_5.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city\\32_chengdu_ddpm_decoded_demand_5.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city\\32_chengdu_generated_demand_embeddings_6.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city\\32_chengdu_ddpm_decoded_demand_6.xlsx")
    #
    main(
        "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city_v3\\32_shanghai_generated_demand_embeddings_bjnj.npy",
        "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
        "32_mae_model.pth",
        "D:\\code\\python\\Infocom2025\\results\\different_city_v3\\32_shanghai_ddpm_decoded_demand_bjnj.xlsx")

    main(
        "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city_v3\\32_shanghai_generated_demand_embeddings_ncyc.npy",
        "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_shanghai_station_feature.txt",
        "32_mae_model.pth",
        "D:\\code\\python\\Infocom2025\\results\\different_city_v3\\32_shanghai_ddpm_decoded_demand_ncyc.xlsx")

    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city_v2\\32_chengdu_generated_demand_embeddings_2.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city_v2\\32_chengdu_ddpm_decoded_demand_2.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city_v2\\32_chengdu_generated_demand_embeddings_3.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city_v2\\32_chengdu_ddpm_decoded_demand_3.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city_v2\\32_chengdu_generated_demand_embeddings_4.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city_v2\\32_chengdu_ddpm_decoded_demand_4.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city_v2\\32_chengdu_generated_demand_embeddings_5.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city_v2\\32_chengdu_ddpm_decoded_demand_5.xlsx")
    #
    # main(
    #     "D:\\code\\python\\Infocom2025\\data\\generated_demand_embeddings\\different_city_v2\\32_chengdu_generated_demand_embeddings_6.npy",
    #     "D:\\code\\python\\Infocom2025\\data\\ukg_feature\\32_chengdu_station_feature.txt",
    #     "32_mae_model.pth",
    #     "D:\\code\\python\\Infocom2025\\results\\different_city_v2\\32_chengdu_ddpm_decoded_demand_6.xlsx")





