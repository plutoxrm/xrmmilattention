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

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except Exception:
    MultilabelStratifiedKFold = None


# ================= JUPYTER/TERMINAL-SAFE LOGGING =================
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
    读取每个患者的 pt 特征文件。

    每个 pt 期望至少包含:
      {
        'patient_id': xxx,
        'feats': Tensor [N, D]
      }
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
        N = feats.shape[0]
        norms = torch.norm(feats, dim=1)
        selected_indices = [norms.argmax().item()]

        for _ in range(min(max_feats - 1, N - 1)):
            best_diversity = -1.0
            best_idx = None

            for idx in range(N):
                if idx in selected_indices:
                    continue

                min_diversity = float('inf')
                for sel_idx in selected_indices:
                    feat_a = feats[idx]
                    feat_b = feats[sel_idx]

                    feat_a_centered = feat_a - feat_a.mean()
                    feat_b_centered = feat_b - feat_b.mean()

                    norm_a = feat_a_centered.norm()
                    norm_b = feat_b_centered.norm()

                    if norm_a < 1e-6 or norm_b < 1e-6:
                        diversity = 1.0
                    else:
                        corr = (feat_a_centered * feat_b_centered).sum() / (norm_a * norm_b)
                        corr = torch.clamp(corr, -1.0, 1.0)
                        diversity = 1.0 - abs(corr.item())
                    min_diversity = min(min_diversity, diversity)

                if min_diversity > best_diversity:
                    best_diversity = min_diversity
                    best_idx = idx

            if best_idx is None:
                for idx in range(N):
                    if idx not in selected_indices:
                        best_idx = idx
                        break

            if best_idx is not None:
                selected_indices.append(best_idx)
            else:
                break

        return torch.tensor(selected_indices)

    def _select_instances(self, feats: torch.Tensor, pid: str) -> torch.Tensor:
        N = feats.shape[0]

        if self.instance_strategy == 'all' or self.max_feats is None or self.max_feats <= 0:
            return feats

        if N <= self.max_feats:
            return feats

        if self.instance_strategy == 'top_norm':
            idxs = torch.topk(torch.norm(feats, dim=1), self.max_feats)[1]
        elif self.instance_strategy == 'fixed_random':
            seed = int(pid) % (2 ** 32)
            g = torch.Generator()
            g.manual_seed(seed)
            idxs = torch.randperm(N, generator=g)[:self.max_feats]
        elif self.instance_strategy == 'random':
            idxs = torch.randperm(N)[:self.max_feats]
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


# ================= POOLING MODELS =================
class MeanMILPool(nn.Module):
    def forward(self, feats: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            pooled = feats.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).float()
            summed = (feats * mask_f).sum(dim=1)
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
        return pooled, None


class MaxMILPool(nn.Module):
    def forward(self, feats: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            pooled = feats.max(dim=1)[0]
        else:
            very_small = torch.finfo(feats.dtype).min
            masked_feats = feats.masked_fill(~mask.unsqueeze(-1), very_small)
            pooled = masked_feats.max(dim=1)[0]
        return pooled, None


class AttentionMILPool(nn.Module):
    def __init__(self, in_dim: int, d_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor = None):
        h = torch.tanh(self.fc1(feats))
        h = self.dropout(h)
        logits = self.fc2(h).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        attn = torch.softmax(logits, dim=1)
        if mask is not None:
            attn = attn * mask.float()
            attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-8)

        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)
        return pooled, attn


class PatientMILFeatures(nn.Module):
    """
    支持三种患者级聚合方式:
      - mean      : 所有实例特征直接平均
      - max       : 所有实例特征直接 max-pooling
      - attention : 学习实例重要性权重
    """
    def __init__(
        self,
        in_dim: int,
        n_labels: int = 1,
        d_hidden_attn: int = 128,
        dropout: float = 0.3,
        architecture: str = 'attention',
    ):
        super().__init__()

        if architecture == 'mean':
            self.pool = MeanMILPool()
            classifier_in_dim = in_dim
        elif architecture == 'max':
            self.pool = MaxMILPool()
            classifier_in_dim = in_dim
        elif architecture == 'attention':
            self.pool = AttentionMILPool(in_dim, d_hidden_attn, dropout=0.2)
            classifier_in_dim = in_dim
        else:
            raise ValueError(f'Unknown architecture: {architecture}. Use mean / max / attention.')

        self.classifier = nn.Linear(classifier_in_dim, n_labels)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor = None):
        pooled, attn = self.pool(feats, mask)
        logits = self.classifier(pooled)
        return logits, pooled, attn


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
    M, L = labels.shape
    weights = []
    for j in range(L):
        p = labels[:, j].sum()
        n = M - p
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
            logits, _, _ = model(feats, mask)
            prob = torch.sigmoid(logits).cpu().numpy()
            scores.append(prob)
            labels.append(ys.numpy())
            pids.extend(batch_pids)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels, pids


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser(description='MIL training with mean/max/attention pooling baselines.')
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

    parser.add_argument('--architecture', type=str, default='attention', choices=['mean', 'max', 'attention'])
    parser.add_argument('--d_hidden_attn', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--use_combined_loss', action='store_true')
    parser.add_argument('--auc_weight', type=float, default=0.5)
    parser.add_argument('--threshold_metric', type=str, default='youden', choices=['youden', 'f1'])
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--save_ckpt', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.out_dir, f'train_{args.architecture}_{ts}.log')
    _, original_print = enable_logging(log_path)

    set_seed(args.seed)

    print('===== TRAINING START =====')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('==========================\n')

    df = pd.read_excel(args.labels_csv) if args.labels_csv.endswith(('xls', 'xlsx')) else pd.read_csv(args.labels_csv)
    if 'id' not in df.columns:
        raise KeyError("labels file must contain an 'id' column")
    df['id'] = df['id'].astype(str)

    y_all = df[args.label_cols].values.astype(int)
    L = y_all.shape[1]
    if L < 1:
        raise ValueError('No label columns provided.')

    if L == 1:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        y_target = y_all[:, 0]
        print('Single-label task: StratifiedKFold')
    else:
        if MultilabelStratifiedKFold is None:
            raise ImportError('iterstrat is required for multi-label splitting. Please install iterative-stratification.')
        splitter = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        y_target = y_all
        print('Multi-label task: MultilabelStratifiedKFold')

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

        model = PatientMILFeatures(
            in_dim=in_dim,
            n_labels=L,
            d_hidden_attn=args.d_hidden_attn,
            dropout=args.dropout,
            architecture=args.architecture,
        ).to(device)

        pos_weight = compute_pos_weight(train_df[args.label_cols].values.astype('float32')).to(device)
        criterion = (
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
            for feats, ys, mask, _ in train_loader:
                feats = feats.to(device)
                ys = ys.to(device)
                mask = mask.to(device)

                logits, _, _ = model(feats, mask)
                loss = criterion(logits, ys)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * feats.size(0)

            epoch_loss /= len(train_loader.dataset)

            train_scores, train_labels, _ = run_inference(model, train_eval_loader, device)
            train_score_1 = train_scores[:, 0]
            train_label_1 = train_labels[:, 0].astype(int)
            threshold = find_best_threshold(train_label_1, train_score_1, metric=args.threshold_metric)

            valid_scores, valid_labels, _ = run_inference(model, valid_loader, device)
            valid_score_1 = valid_scores[:, 0]
            valid_label_1 = valid_labels[:, 0].astype(int)
            metrics = evaluate_binary_metrics(valid_label_1, valid_score_1, threshold)

            is_best = (not np.isnan(metrics['auc'])) and (metrics['auc'] > best_metrics['auc'])
            if is_best:
                best_metrics.update(metrics)
                best_metrics['best_epoch'] = epoch

                if args.save_ckpt:
                    ckpt_path = os.path.join(args.out_dir, f'best_{args.architecture}_fold{fold}.pt')
                    torch.save(
                        {
                            'fold': fold,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_metrics': best_metrics,
                            'architecture': args.architecture,
                            'in_dim': in_dim,
                            'label_cols': args.label_cols,
                            'threshold_metric': args.threshold_metric,
                        },
                        ckpt_path,
                    )

            msg = (
                f"Epoch {epoch:03d}: "
                f"loss={epoch_loss:.4f}  "
                f"AUROC={metrics['auc']:.4f}  "
                f"AUPRC={metrics['auprc']:.4f}  "
                f"ACC={metrics['acc']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}  "
                f"Spec={metrics['specificity']:.4f}  "
                f"Sens@95Spec={metrics['sens95']:.4f}  "
                f"Brier={metrics['brier']:.4f}  "
                f"Thr={threshold:.3f}"
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
    summary_path = os.path.join(args.out_dir, f'cv_summary_{args.architecture}_{ts}.csv')
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
