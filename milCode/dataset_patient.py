# dataset_patient.py
import os
import glob
import random
from typing import List, Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PatientBagDataset(Dataset):
    """
    读取患者级“图像袋”（bag）。

    - root_dir: data 根目录，子目录为 patient_id（例如 '1','2','189'）
    - labels_csv: 含 id 列 和 多标签列（0/1）
    - max_images:
        >0: 最多取这么多张图（训练随机采样 / 验证取前 max_images）
       <=0: 使用全部图像（推荐你现在用的方案）
    - train: 训练模式会做更强的数据增强
    - 递归搜索图片：支持 data/id/**/*.jpg 结构
    """
    def __init__(self,
                 root_dir: str,
                 labels_csv: str,
                 label_cols: List[str],
                 max_images: int = 0,
                 train: bool = True,
                 img_exts=(".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff",".PNG", ".JPEG")):
        super().__init__()
        self.root_dir = root_dir
        self.label_cols = label_cols
        self.max_images = max_images
        self.train = train
        self.img_exts = img_exts

        # 读取标签（支持 csv/xlsx）
        if labels_csv.lower().endswith(".xlsx") or labels_csv.lower().endswith(".xls"):
            df = pd.read_excel(labels_csv)
        else:
            df = pd.read_csv(labels_csv)
        # 统一用 'id' 做患者标识
        if "id" not in df.columns:
            raise ValueError("标签文件中必须有一列名为 'id' 的患者编号列。")
        self.df = df
        self.patient_ids = self.df["id"].astype(str).tolist()
        self.labels = self.df[self.label_cols].values.astype("float32")

        # 图像预处理/增强
        size = 518
        if train:
            self.tx = T.Compose([
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
                # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tx = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def _load_images_for_patient(self, pid: str) -> List[str]:
        """
        递归搜索某个患者所有图片路径。
        支持 data/<id>/**/*.jpg 这种多级目录。
        """
        pdir = os.path.join(self.root_dir, pid)
        files = []
        if os.path.isdir(pdir):
            for ext in self.img_exts:
                files.extend(glob.glob(os.path.join(pdir, "**", f"*{ext}"), recursive=True))
        files = sorted(files)
        return files

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, List[str]]:
        pid = self.patient_ids[idx]
        y = torch.from_numpy(self.labels[idx])  # [n_labels]
        paths = self._load_images_for_patient(pid)
        if len(paths) == 0:
            raise FileNotFoundError(f"No images found for patient id={pid} in {os.path.join(self.root_dir, pid)}")

        # ✅ 使用全部图像，或者按 max_images 裁剪
        if self.max_images is not None and self.max_images > 0 and len(paths) > self.max_images:
            if self.train:
                # 训练时随机采样 max_images
                paths_sampled = random.sample(paths, self.max_images)
            else:
                # 验证/解释时取前 max_images，保证可复现
                paths_sampled = paths[:self.max_images]
        else:
            # max_images<=0 或者 总图像数 <= max_images 时，全部使用
            paths_sampled = paths

        imgs = []
        orig_paths = []
        for p in paths_sampled:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                # 避免个别坏图中断，简单跳过
                continue
            imgs.append(self.tx(img))
            orig_paths.append(p)

        if len(imgs) == 0:
            raise RuntimeError(f"所有图像均读取失败，患者 id={pid}")

        bag = torch.stack(imgs, dim=0)  # [N, 3, H, W]
        
        return bag, y, pid, orig_paths
