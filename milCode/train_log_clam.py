import argparse
import builtins
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', message='Input data has no positive sample')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


# ================= LOGGING =================
def enable_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w', encoding='utf-8')
    original_print = builtins.print

    def logged_print(*args, **kwargs):
        original_print(*args, **kwargs)
        log_kwargs = kwargs.copy()
        log_kwargs.pop('file', None)
        original_print(*args, file=log_file, **log_kwargs)
        log_file.flush()

    builtins.print = logged_print
    return log_file, original_print


# ================= SEED =================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= DATASET =================
class PatientFeatureDataset(Dataset):
    """
    每个患者一个 .pt 文件，文件中至少包含:
      {
        'patient_id': xxx,
        'feats': Tensor [N, D]
      }

    训练阶段可随机/固定随机/多样性/范数方式选取一部分实例。
    验证阶段可设置 all，使用全部实例 full-bag 推断。
    """
    def __init__(
        self,
        feat_dir: str,
        labels_df: pd.DataFrame,
        label_cols: List[str],
        max_feats: int = -1,
        instance_strategy: str = 'all',
        random_seed: int = 42,
    ):
        super().__init__()
        self.feat_dir = feat_dir
        self.df = labels_df.copy().reset_index(drop=True)
        self.label_cols = label_cols
        self.max_feats = max_feats
        self.instance_strategy = instance_strategy
        self.random_seed = random_seed

        if 'id' not in self.df.columns:
            raise KeyError("Labels file must contain column 'id'.")

        self.df['id'] = self.df['id'].astype(str)
        self.patient_ids = self.df['id'].tolist()
        self.labels = self.df[self.label_cols].values.astype('float32')

    def __len__(self):
        return len(self.patient_ids)

    @staticmethod
    def _safe_load_pt(path: str):
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            return torch.load(path, map_location='cpu')

    def _select_diverse_instances(self, feats: torch.Tensor, max_feats: int) -> torch.Tensor:
        n = feats.shape[0]
        norms = torch.norm(feats, dim=1)
        selected = [norms.argmax().item()]

        for _ in range(min(max_feats - 1, n - 1)):
            best_div = -1.0
            best_idx = None
            for idx in range(n):
                if idx in selected:
                    continue
                min_div = float('inf')
                for sidx in selected:
                    a = feats[idx]
                    b = feats[sidx]
                    a = a - a.mean()
                    b = b - b.mean()
                    na = a.norm()
                    nb = b.norm()
                    if na < 1e-6 or nb < 1e-6:
                        div = 1.0
                    else:
                        corr = (a * b).sum() / (na * nb)
                        corr = torch.clamp(corr, -1.0, 1.0)
                        div = 1.0 - abs(corr.item())
                    min_div = min(min_div, div)
                if min_div > best_div:
                    best_div = min_div
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
        return torch.tensor(selected)

    def _select_instances(self, feats: torch.Tensor, pid: str) -> torch.Tensor:
        n = feats.shape[0]
        if self.instance_strategy == 'all' or self.max_feats is None or self.max_feats <= 0:
            return feats
        if n <= self.max_feats:
            return feats

        if self.instance_strategy == 'random':
            idxs = torch.randperm(n)[:self.max_feats]
        elif self.instance_strategy == 'fixed_random':
            g = torch.Generator()
            try:
                seed = int(pid) % (2 ** 32)
            except ValueError:
                seed = self.random_seed
            g.manual_seed(seed)
            idxs = torch.randperm(n, generator=g)[:self.max_feats]
        elif self.instance_strategy == 'top_norm':
            idxs = torch.topk(torch.norm(feats, dim=1), self.max_feats)[1]
        elif self.instance_strategy == 'diversity':
            idxs = self._select_diverse_instances(feats, self.max_feats)
        else:
            raise ValueError(f'Unknown instance_strategy: {self.instance_strategy}')
        return feats[idxs]

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        path = os.path.join(self.feat_dir, f'{pid}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Feature file not found: {path}')

        data = self._safe_load_pt(path)
        if 'feats' not in data:
            raise KeyError(f"Feature file {path} does not contain key 'feats'.")

        feats = data['feats']
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float32)
        feats = feats.float()
        feats = self._select_instances(feats, pid)
        return feats, label, pid


# ================= COLLATE =================
def pad_collate(batch):
    feats_list, ys, pids = zip(*batch)
    lengths = [x.shape[0] for x in feats_list]
    max_len = max(lengths)
    feat_dim = feats_list[0].shape[1]

    padded = torch.zeros(len(batch), max_len, feat_dim, dtype=feats_list[0].dtype)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, x in enumerate(feats_list):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = True

    return padded, torch.stack(ys), mask, pids


# ================= MODEL: CLAM =================
class AttentionNetwork(nn.Module):
    """
    CLAM 的注意力网络。
    - simple: 线性 + tanh + dropout + 线性
    - gated : CLAM 常用的 gated attention
    输出每个实例的注意力 logit，再经过 masked softmax 得到权重。
    """
    def __init__(self, in_dim: int, attn_dim: int = 256, dropout: float = 0.25, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.dropout = nn.Dropout(dropout)

        if gated:
            self.attn_v = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh())
            self.attn_u = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid())
            self.attn_w = nn.Linear(attn_dim, 1)
        else:
            self.attn = nn.Sequential(
                nn.Linear(in_dim, attn_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(attn_dim, 1),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if self.gated:
            a_v = self.attn_v(x)
            a_u = self.attn_u(x)
            logits = self.attn_w(self.dropout(a_v * a_u)).squeeze(-1)
        else:
            logits = self.attn(x).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        attn = torch.softmax(logits, dim=1)
        if mask is not None:
            attn = attn * mask.float()
            attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return attn, logits


class CLAMBinary(nn.Module):
    """
    面向单标签二分类的简化 CLAM。

    模块包含：
    1) 特征投影
    2) 注意力网络
    3) 基于注意力的加权聚合（bag representation）
    4) bag-level 分类器
    5) instance-level 分类器（背景 vs 典型病理特征）

    约束聚类 / 实例级辅助监督做法：
    - 正样本 bag:
        top-k 最高注意力实例 -> 伪标签 1（典型病理特征）
        low-k 最低注意力实例 -> 伪标签 0（背景特征）
    - 负样本 bag:
        top-k 和 low-k 实例都标记为 0（背景 / 非病理）
    """
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.25,
        gated_attention: bool = True,
        k_sample: int = 8,
    ):
        super().__init__()
        self.k_sample = k_sample

        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_net = AttentionNetwork(
            in_dim=embed_dim,
            attn_dim=attn_dim,
            dropout=dropout,
            gated=gated_attention,
        )
        self.bag_classifier = nn.Linear(embed_dim, 1)
        self.instance_classifier = nn.Linear(embed_dim, 2)
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def forward(self, feats: torch.Tensor, mask: torch.Tensor = None):
        h = self.feature_proj(feats)                       # [B, N, E]
        attn, attn_logits = self.attention_net(h, mask)   # [B, N]
        bag_feat = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # [B, E]
        bag_logits = self.bag_classifier(bag_feat)        # [B, 1]
        return bag_logits, bag_feat, attn, h, attn_logits

    def instance_clustering_loss(
        self,
        h: torch.Tensor,
        attn: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        根据 attention 选出 top-k / low-k 实例做伪标签监督。
        返回实例损失，以及一些可打印统计量。
        """
        device = h.device
        total_loss = torch.tensor(0.0, device=device)
        valid_bags = 0
        selected_total = 0
        pos_selected = 0
        neg_selected = 0

        batch_size = h.shape[0]
        for b in range(batch_size):
            valid_idx = torch.where(mask[b])[0]
            if valid_idx.numel() == 0:
                continue

            h_valid = h[b, valid_idx]          # [n_valid, E]
            a_valid = attn[b, valid_idx]       # [n_valid]
            n_valid = h_valid.shape[0]

            k = min(self.k_sample, max(1, n_valid // 2))
            order_desc = torch.argsort(a_valid, descending=True)
            top_idx = order_desc[:k]

            if n_valid > 1:
                top_set = set(top_idx.detach().cpu().tolist())
                bottom_list = []
                for idx in torch.argsort(a_valid, descending=False).detach().cpu().tolist():
                    if idx not in top_set:
                        bottom_list.append(idx)
                    if len(bottom_list) >= k:
                        break
                bottom_idx = torch.tensor(bottom_list, device=device, dtype=torch.long)
            else:
                bottom_idx = torch.empty(0, device=device, dtype=torch.long)

            logits_list = []
            target_list = []

            # top-k: 正 bag -> 1, 负 bag -> 0
            top_logits = self.instance_classifier(h_valid[top_idx])
            if labels[b, 0] >= 0.5:
                top_targets = torch.ones(top_logits.shape[0], dtype=torch.long, device=device)
            else:
                top_targets = torch.zeros(top_logits.shape[0], dtype=torch.long, device=device)
            logits_list.append(top_logits)
            target_list.append(top_targets)

            pos_selected += int((top_targets == 1).sum().item())
            neg_selected += int((top_targets == 0).sum().item())

            # low-k: 一律视作背景 0
            if bottom_idx.numel() > 0:
                low_logits = self.instance_classifier(h_valid[bottom_idx])
                low_targets = torch.zeros(low_logits.shape[0], dtype=torch.long, device=device)
                logits_list.append(low_logits)
                target_list.append(low_targets)
                neg_selected += int(low_targets.shape[0])

            inst_logits = torch.cat(logits_list, dim=0)
            inst_targets = torch.cat(target_list, dim=0)
            loss_b = self.instance_loss_fn(inst_logits, inst_targets)

            total_loss = total_loss + loss_b
            valid_bags += 1
            selected_total += int(inst_targets.shape[0])

        if valid_bags > 0:
            total_loss = total_loss / valid_bags
        stats = {
            'inst_selected': float(selected_total),
            'inst_pos_pseudo': float(pos_selected),
            'inst_neg_pseudo': float(neg_selected),
            'inst_valid_bags': float(valid_bags),
        }
        return total_loss, stats


# ================= LOSS =================
class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=None, auc_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.auc_weight = auc_weight
        try:
            from libauc.losses import AUCMLoss
        except ImportError as exc:
            raise ImportError(
                'You passed --use_combined_loss, but libauc is not installed. '
                'Please install libauc or remove --use_combined_loss.'
            ) from exc
        self.auc = AUCMLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        return (
            (1.0 - self.auc_weight) * self.bce(logits, labels)
            + self.auc_weight * self.auc(torch.sigmoid(logits), labels)
        )


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    m, l = labels.shape
    weights = []
    for j in range(l):
        p = labels[:, j].sum()
        n = m - p
        weights.append((n + 1e-6) / (p + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)


# ================= METRICS =================
def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return np.nan
    return tpr[idx[-1]]


def safe_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    if denom == 0:
        return np.nan
    return tn / denom


def evaluate_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    out = {}
    try:
        out['auc'] = roc_auc_score(y_true, y_score)
    except ValueError:
        out['auc'] = np.nan

    try:
        out['auprc'] = average_precision_score(y_true, y_score)
    except ValueError:
        out['auprc'] = np.nan

    try:
        out['sens95'] = sensitivity_at_specificity(y_true, y_score, target_spec=0.95)
    except ValueError:
        out['sens95'] = np.nan

    try:
        out['brier'] = brier_score_loss(y_true, y_score)
    except ValueError:
        out['brier'] = np.nan

    y_pred = (y_score >= threshold).astype(int)
    out['acc'] = float((y_pred == y_true).mean())
    out['f1'] = f1_score(y_true, y_pred, zero_division=0)
    out['precision'] = precision_score(y_true, y_pred, zero_division=0)
    out['recall'] = recall_score(y_true, y_pred, zero_division=0)
    out['specificity'] = safe_specificity(y_true, y_pred)
    out['threshold'] = threshold
    return out


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = 'youden') -> float:
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return 0.5

    thresholds = np.linspace(0.0, 1.0, 1001)
    best_thr = 0.5
    best_val = -np.inf

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            recall = recall_score(y_true, y_pred, zero_division=0)
            spec = safe_specificity(y_true, y_pred)
            if np.isnan(spec):
                continue
            score = recall + spec - 1.0
        else:
            raise ValueError(f'Unknown threshold_metric: {metric}. Use youden / f1.')

        if score > best_val:
            best_val = score
            best_thr = float(thr)

    return best_thr


# ================= UTILS =================
def infer_feature_dim(feat_dir: str) -> int:
    pt_files = [f for f in os.listdir(feat_dir) if f.endswith('.pt')]
    if not pt_files:
        raise FileNotFoundError(f'No .pt files found in feat_dir: {feat_dir}')
    sample_path = os.path.join(feat_dir, pt_files[0])
    try:
        data = torch.load(sample_path, map_location='cpu', weights_only=True)
    except TypeError:
        data = torch.load(sample_path, map_location='cpu')
    feats = data['feats']
    return int(feats.shape[1])


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    scores = []
    labels = []
    pids = []

    with torch.no_grad():
        for feats, ys, mask, batch_pids in loader:
            feats = feats.to(device)
            mask = mask.to(device)
            bag_logits, _, _, _, _ = model(feats, mask)
            prob = torch.sigmoid(bag_logits).cpu().numpy()
            scores.append(prob)
            labels.append(ys.numpy())
            pids.extend(batch_pids)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels, pids


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser(description='CLAM-style MIL training for patient-level pre-extracted features.')
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--label_cols', nargs='+', required=True)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--train_max_feats', type=int, default=30)
    parser.add_argument('--train_instance_strategy', type=str, default='random')
    parser.add_argument('--valid_max_feats', type=int, default=-1)
    parser.add_argument('--valid_instance_strategy', type=str, default='all')

    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--attn_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--gated_attention', action='store_true')
    parser.add_argument('--k_sample', type=int, default=8)
    parser.add_argument('--instance_loss_weight', type=float, default=1.0)

    parser.add_argument('--use_combined_loss', action='store_true')
    parser.add_argument('--auc_weight', type=float, default=0.5)
    parser.add_argument('--threshold_metric', type=str, default='youden', choices=['youden', 'f1'])
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--save_ckpt', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.out_dir, f'train_clam_{ts}.log')
    _, original_print = enable_logging(log_path)

    set_seed(args.seed)

    print('===== CLAM TRAINING START =====')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('================================\n')

    df = pd.read_excel(args.labels_csv) if args.labels_csv.endswith(('xls', 'xlsx')) else pd.read_csv(args.labels_csv)
    if 'id' not in df.columns:
        raise KeyError("labels file must contain an 'id' column")
    df['id'] = df['id'].astype(str)

    y_all = df[args.label_cols].values.astype(int)
    l = y_all.shape[1]
    if l != 1:
        raise ValueError(
            'This CLAM script currently supports only single-label binary classification. '
            'Please pass exactly one label column, e.g. --label_cols 代谢慢病'
        )

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y_target = y_all[:, 0]
    print('Single-label binary task: StratifiedKFold')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    in_dim = infer_feature_dim(args.feat_dir)
    print(f'Inferred feature dimension: {in_dim}')

    all_folds_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df, y_target), 1):
        print(f'\n===== Fold {fold}/{args.folds} =====')

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        valid_df = df.iloc[va_idx].reset_index(drop=True)

        train_ds = PatientFeatureDataset(
            args.feat_dir,
            train_df,
            args.label_cols,
            max_feats=args.train_max_feats,
            instance_strategy=args.train_instance_strategy,
            random_seed=args.seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=pad_collate,
        )

        train_eval_ds = PatientFeatureDataset(
            args.feat_dir,
            train_df,
            args.label_cols,
            max_feats=args.valid_max_feats,
            instance_strategy=args.valid_instance_strategy,
            random_seed=args.seed,
        )
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pad_collate,
        )

        valid_ds = PatientFeatureDataset(
            args.feat_dir,
            valid_df,
            args.label_cols,
            max_feats=args.valid_max_feats,
            instance_strategy=args.valid_instance_strategy,
            random_seed=args.seed,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pad_collate,
        )

        model = CLAMBinary(
            in_dim=in_dim,
            embed_dim=args.embed_dim,
            attn_dim=args.attn_dim,
            dropout=args.dropout,
            gated_attention=args.gated_attention,
            k_sample=args.k_sample,
        ).to(device)

        pos_weight = compute_pos_weight(train_df[args.label_cols].values.astype('float32')).to(device)
        bag_criterion = (
            CombinedLoss(pos_weight=pos_weight, auc_weight=args.auc_weight)
            if args.use_combined_loss
            else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            threshold=0.002,
            cooldown=3,
            min_lr=1e-6,
        )

        best_metrics = {
            'auc': -1.0,
            'auprc': np.nan,
            'acc': np.nan,
            'f1': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'specificity': np.nan,
            'sens95': np.nan,
            'brier': np.nan,
            'threshold': 0.5,
            'best_epoch': 0,
        }

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_bag_loss = 0.0
            epoch_inst_loss = 0.0
            epoch_inst_selected = 0.0
            epoch_pos_pseudo = 0.0
            epoch_neg_pseudo = 0.0

            for feats, ys, mask, _ in train_loader:
                feats = feats.to(device)
                ys = ys.to(device)
                mask = mask.to(device)

                bag_logits, _, attn, h, _ = model(feats, mask)
                bag_loss = bag_criterion(bag_logits, ys)
                inst_loss, inst_stats = model.instance_clustering_loss(h, attn, ys, mask)
                loss = bag_loss + args.instance_loss_weight * inst_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = feats.size(0)
                epoch_loss += loss.item() * bs
                epoch_bag_loss += bag_loss.item() * bs
                epoch_inst_loss += inst_loss.item() * bs
                epoch_inst_selected += inst_stats['inst_selected']
                epoch_pos_pseudo += inst_stats['inst_pos_pseudo']
                epoch_neg_pseudo += inst_stats['inst_neg_pseudo']

            epoch_loss /= len(train_loader.dataset)
            epoch_bag_loss /= len(train_loader.dataset)
            epoch_inst_loss /= len(train_loader.dataset)

            # 训练 fold 内用 full-bag 选阈值
            train_scores, train_labels, _ = run_inference(model, train_eval_loader, device)
            train_score_1 = train_scores[:, 0]
            train_label_1 = train_labels[:, 0].astype(int)
            threshold = find_best_threshold(train_label_1, train_score_1, metric=args.threshold_metric)

            # 验证 fold full-bag 推断
            valid_scores, valid_labels, _ = run_inference(model, valid_loader, device)
            valid_score_1 = valid_scores[:, 0]
            valid_label_1 = valid_labels[:, 0].astype(int)
            metrics = evaluate_binary_metrics(valid_label_1, valid_score_1, threshold)

            is_best = (not np.isnan(metrics['auc'])) and (metrics['auc'] > best_metrics['auc'])
            if is_best:
                best_metrics.update(metrics)
                best_metrics['best_epoch'] = epoch

                if args.save_ckpt:
                    ckpt_path = os.path.join(args.out_dir, f'best_clam_fold{fold}.pt')
                    torch.save(
                        {
                            'fold': fold,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_metrics': best_metrics,
                            'model_type': 'CLAMBinary',
                            'in_dim': in_dim,
                            'embed_dim': args.embed_dim,
                            'attn_dim': args.attn_dim,
                            'dropout': args.dropout,
                            'gated_attention': args.gated_attention,
                            'k_sample': args.k_sample,
                            'instance_loss_weight': args.instance_loss_weight,
                            'label_cols': args.label_cols,
                            'threshold_metric': args.threshold_metric,
                        },
                        ckpt_path,
                    )

            msg = (
                f"Epoch {epoch:03d}: "
                f"loss={epoch_loss:.4f}  "
                f"bag_loss={epoch_bag_loss:.4f}  "
                f"inst_loss={epoch_inst_loss:.4f}  "
                f"AUROC={metrics['auc']:.4f}  "
                f"AUPRC={metrics['auprc']:.4f}  "
                f"ACC={metrics['acc']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}  "
                f"Spec={metrics['specificity']:.4f}  "
                f"Sens@95Spec={metrics['sens95']:.4f}  "
                f"Brier={metrics['brier']:.4f}  "
                f"Thr={threshold:.3f}  "
                f"InstSel={int(epoch_inst_selected)}  "
                f"PseudoPos={int(epoch_pos_pseudo)}  "
                f"PseudoNeg={int(epoch_neg_pseudo)}"
            )
            if is_best:
                msg += '  ✓ NEW BEST'
            print(msg)

            scheduler.step(metrics['auc'] if not np.isnan(metrics['auc']) else -1.0)

        print(
            f"Fold {fold} BEST | "
            f"Epoch={best_metrics['best_epoch']:03d}  "
            f"AUROC={best_metrics['auc']:.4f}  "
            f"AUPRC={best_metrics['auprc']:.4f}  "
            f"ACC={best_metrics['acc']:.4f}  "
            f"F1={best_metrics['f1']:.4f}  "
            f"Prec={best_metrics['precision']:.4f}  "
            f"Recall={best_metrics['recall']:.4f}  "
            f"Spec={best_metrics['specificity']:.4f}  "
            f"Sens@95Spec={best_metrics['sens95']:.4f}  "
            f"Brier={best_metrics['brier']:.4f}  "
            f"Thr={best_metrics['threshold']:.3f}"
        )
        all_folds_metrics.append(best_metrics.copy())

    summary_df = pd.DataFrame(all_folds_metrics)
    summary_path = os.path.join(args.out_dir, f'cv_summary_clam_{ts}.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    print('\n===== CROSS-VALIDATION SUMMARY (mean ± std) =====')
    for key in ['auc', 'auprc', 'acc', 'f1', 'precision', 'recall', 'specificity', 'sens95', 'brier', 'threshold']:
        vals = np.array(summary_df[key], dtype=float)
        print(f'{key.upper():<12}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}')

    print(f'\nPer-fold summary saved to: {summary_path}')
    print(f'Full training log saved to: {log_path}')
    print('\nTraining complete.')

    builtins.print = original_print


if __name__ == '__main__':
    main()
