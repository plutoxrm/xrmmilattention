import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_patient import PatientBagDataset
from mil_train import PatientMILMultiLabel


def collate(batch):
    """
    和 train.py 里基本一致的 collate：
    - batch_size 建议为 1
    - 保留患者 id 和原始图像路径列表
    """
    bags, ys, pids, paths = zip(*batch)
    bags = bags[0].unsqueeze(0)  # [1, N, 3, H, W]
    ys = torch.stack(ys, dim=0)  # [B, L]
    return bags, ys, pids, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--labels_csv', type=str, default='./labels.xlsx')
    parser.add_argument('--label_cols', nargs='+', required=True,
                         help='多标签列名，例如: 消化系统疾病 内分泌代谢疾病 泌尿生殖系统疾病 血液及造血器官疾病和涉及免疫机制的某些疾患')
    parser.add_argument('--max_images', type=int, default=0,
                        help='<=0 使用全部图像；>0 时每个病人最多取这么多张图')
    parser.add_argument('--encoder', type=str, default='efficientnet_b0')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='是否使用离线预训练权重(1/0)')
    parser.add_argument('--weights_path', type=str, default='../weight/efficientnet_b0_rwightman-3dd342df.pth',
                        help='离线权重 .pth 文件路径')
    parser.add_argument('--freeze_encoder', type=int, default=1,
                        help='是否冻结编码器权重 (1/0)')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='可选：训练好的 best_foldX.ckpt 路径，若提供则加载其中的 encoder 权重')
    parser.add_argument('--out_xlsx', type=str, default='./outputs/encoder_instance_features.xlsx')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)

    # ====== 1. 构建数据集（只为了拿到 patient_id 和图像顺序） ======
    # 和 train.py / PatientBagDataset 的用法保持一致
    ds = PatientBagDataset(
        root_dir=args.data_root,
        labels_csv=args.labels_csv,
        label_cols=args.label_cols,
        max_images=args.max_images if args.max_images > 0 else 0,
        train=False   # 特征导出用验证模式，不做数据增强
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("✅ Using device:", device)

    # ====== 2. 构建模型并加载权重 ======
    model = PatientMILMultiLabel(
        n_labels=len(args.label_cols),
        encoder_name=args.encoder,
        d_hidden_attn=512,
        dropout=0.0,
        pretrained=bool(args.pretrained),
        weights_path=args.weights_path,
        freeze_encoder=bool(args.freeze_encoder)
    ).to(device)

    model.eval()

    # ====== 3. 遍历所有病人，提取每张图的 encoder 特征 ======
    rows = []
    with torch.no_grad():
        for bags, ys, pids, paths in loader:
            pid = pids[0]
            img_paths = paths[0]
            bags = bags.to(device)       # [1, N, 3, H, W]

            # ===== 直接用 model.encoder 提取特征 =====
            B, N, C, H, W = bags.shape
            x = bags.view(B * N, C, H, W)        # [B*N, 3, H, W]
            feats = model.encoder(x)             # [B*N, D]
            D = feats.shape[-1]
            feats = feats.view(B, N, D)          # [B, N, D]
            feats_np = feats.squeeze(0).cpu().numpy()   # [N, D]
            # ======================================

            assert feats_np.shape[0] == len(img_paths)

            for i in range(feats_np.shape[0]):
                row = {
                    "patient_id": pid,
                    "image_path": img_paths[i]
                }
                for d in range(D):
                    row[f"feat_{d}"] = feats_np[i, d]
                rows.append(row)

            print(f"已处理患者 {pid}，图像数 {feats_np.shape[0]}，特征维度 {D}")


    # ====== 4. 保存为 xlsx ======
    df_out = pd.DataFrame(rows)
    df_out.to_excel(args.out_xlsx, index=False)
    print(f"\n✅ 特征文件已保存到: {args.out_xlsx}")
    print(f"总样本数（行数）: {len(df_out)}，特征维度: {df_out.shape[1] - 2}")
    

if __name__ == "__main__":
    main()
"""
python extract_instance_features.py \
  --data_root ../newData \
  --labels_csv ./labels03.xlsx \
  --label_cols 代谢慢病 \
  --max_images -1 \
  --encoder vit_base_patch14_dinov2 \
  --pretrained 1 \
  --weights_path ../weight/dinov2_vitb14_pretrain.pth \
  --freeze_encoder 1 \
  --out_xlsx ./outputs/encoder_instance_features.xlsx

"""

