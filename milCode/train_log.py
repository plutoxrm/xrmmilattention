import argparse
import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime
import builtins

warnings.filterwarnings('ignore', message='Input data has no positive sample')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    brier_score_loss
)
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from libauc.losses import AUCMLoss

from dataset_patient_pt import PatientFeatureDataset
from mil_train_pt import PatientMILFeatures


# ================= JUPYTER-SAFE LOGGING =================
def enable_jupyter_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    original_print = builtins.print

    def logged_print(*args, **kwargs):
        # 1. 保持原有的屏幕输出行为
        original_print(*args, **kwargs)
        
        # 2. 复制 kwargs 并安全地移除 'file' 键，防止参数冲突
        log_kwargs = kwargs.copy()
        log_kwargs.pop('file', None)
        
        # 3. 将内容写入你的本地日志文件
        original_print(*args, file=log_file, **log_kwargs)
        log_file.flush()

    builtins.print = logged_print


# ================= SEED =================
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= METRICS =================
def sensitivity_at_specificity(y_true, y_score, target_spec=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return np.nan
    return tpr[idx[-1]]


# ================= LOSS =================
class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=None, auc_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.auc = AUCMLoss()
        self.auc_weight = auc_weight

    def forward(self, logits, labels):
        return (
            (1 - self.auc_weight) * self.bce(logits, labels)
            + self.auc_weight * self.auc(torch.sigmoid(logits), labels)
        )


def compute_pos_weight(labels):
    M, L = labels.shape
    weights = []
    for j in range(L):
        p = labels[:, j].sum()
        n = M - p
        weights.append((n + 1e-6) / (p + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)


def collate(batch):
    feats, ys, pids = zip(*batch)
    return torch.stack(feats), torch.stack(ys), pids


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--label_cols', nargs='+', required=True)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_feats', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--instance_strategy', type=str, default='random')
    parser.add_argument('--architecture', type=str, default='attention')

    parser.add_argument('--use_combined_loss', action='store_true')
    parser.add_argument('--auc_weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # ---- logging ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    enable_jupyter_logging(f"outputs/train_{ts}.log")

    set_seed(args.seed)

    print("===== TRAINING START =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==========================\n")

    df = pd.read_excel(args.labels_csv) if args.labels_csv.endswith(('xls', 'xlsx')) else pd.read_csv(args.labels_csv)
    y_all = df[args.label_cols].values.astype(int)
    L = y_all.shape[1]

    if L == 1:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        y_target = y_all[:, 0]
        print("Single-label task: StratifiedKFold")
    else:
        splitter = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        y_target = y_all
        print("Multi-label task: MultilabelStratifiedKFold")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_folds_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df, y_target), 1):
        print(f"\n===== Fold {fold}/{args.folds} =====")

        train_ds = PatientFeatureDataset(
            args.feat_dir, df.iloc[tr_idx], args.label_cols, args.max_feats,
            instance_strategy=args.instance_strategy
        )
        valid_ds = PatientFeatureDataset(
            args.feat_dir, df.iloc[va_idx], args.label_cols, args.max_feats,
            instance_strategy=args.instance_strategy
        )

        tr_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate)
        va_loader = DataLoader(valid_ds, args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate)

        model = PatientMILFeatures(
            in_dim=768,
            n_labels=L,
            d_hidden_attn=128,
            dropout=0.3,
            architecture=args.architecture
        ).to(device)

        pos_weight = compute_pos_weight(
            df.iloc[tr_idx][args.label_cols].values.astype("float32")
        ).to(device)

        criterion = (
            CombinedLoss(pos_weight, args.auc_weight)
            if args.use_combined_loss
            else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5,
            patience=10, threshold=0.002,
            cooldown=3, min_lr=1e-6
        )

        best_metrics = {
            "auc": -1.0,
            "auprc": np.nan,
            "sens95": np.nan,
            "brier": np.nan,
        }

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for feats, ys, _ in tr_loader:
                feats, ys = feats.to(device), ys.to(device)
                logits, _, _ = model(feats)
                loss = criterion(logits, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * feats.size(0)
            epoch_loss /= len(tr_loader.dataset)

            model.eval()
            scores, labels = [], []
            with torch.no_grad():
                for feats, ys, _ in va_loader:
                    feats = feats.to(device)
                    logits, _, _ = model(feats)
                    scores.append(logits.sigmoid().cpu().numpy())
                    labels.append(ys.numpy())

            scores = np.concatenate(scores)
            labels = np.concatenate(labels)

            try:
                auc = roc_auc_score(labels[:, 0], scores[:, 0])
                auprc = average_precision_score(labels[:, 0], scores[:, 0])
                sens95 = sensitivity_at_specificity(labels[:, 0], scores[:, 0])
                brier = brier_score_loss(labels[:, 0], scores[:, 0])
            except ValueError:
                auc, auprc, sens95, brier = np.nan, np.nan, np.nan, np.nan

            is_best = (not np.isnan(auc)) and (auc > best_metrics["auc"])
            if is_best:
                best_metrics["auc"] = auc
                best_metrics["auprc"] = auprc
                best_metrics["sens95"] = sens95
                best_metrics["brier"] = brier

            msg = (
                f"Epoch {epoch:03d}: "
                f"loss={epoch_loss:.4f}  "
                f"AUROC={auc:.4f}  "
                f"AUPRC={auprc:.4f}  "
                f"Sens@95Spec={sens95:.4f}  "
                f"Brier={brier:.4f}"
            )
            if is_best:
                msg += "  ✓ NEW BEST"
            print(msg)

            scheduler.step(auc)

        print(
            f"Fold {fold} BEST | "
            f"AUROC={best_metrics['auc']:.4f}  "
            f"AUPRC={best_metrics['auprc']:.4f}  "
            f"Sens@95Spec={best_metrics['sens95']:.4f}  "
            f"Brier={best_metrics['brier']:.4f}"
        )

        all_folds_metrics.append(best_metrics)

    print("\n===== CROSS-VALIDATION SUMMARY (mean ± std) =====")
    for key in ["auc", "auprc", "sens95", "brier"]:
        vals = [m[key] for m in all_folds_metrics]
        print(f"{key.upper():<10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
