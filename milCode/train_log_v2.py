import argparse
import builtins
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', message='Input data has no positive sample')

from dataset_patient_pt import PatientFeatureDataset
from mil_train_pt import PatientMILFeatures
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from libauc.losses import AUCMLoss


def enable_jupyter_logging(log_path: str):
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



def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=None, auc_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.auc = AUCMLoss()
        self.auc_weight = auc_weight

    def forward(self, logits, labels):
        return (1 - self.auc_weight) * self.bce(logits, labels) + self.auc_weight * self.auc(
            torch.sigmoid(logits), labels
        )



def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    m, l = labels.shape
    weights = []
    for j in range(l):
        p = labels[:, j].sum()
        n = m - p
        weights.append((n + 1e-6) / (p + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)



def collate_train_fixed(batch):
    feats, ys, pids = zip(*batch)
    return torch.stack(feats), torch.stack(ys), pids



def collate_eval_variable(batch):
    assert len(batch) == 1, 'Variable-length eval collate requires batch_size=1.'
    feats, ys, pids = zip(*batch)
    return feats[0].unsqueeze(0), ys[0].unsqueeze(0), pids



def sensitivity_at_specificity(y_true, y_score, target_spec=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return np.nan
    return tpr[idx[-1]]



def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    if denom == 0:
        return np.nan
    return tn / denom



def find_best_threshold_binary(y_true, y_score, metric='youden'):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    unique_scores = np.unique(y_score)
    if len(unique_scores) == 1:
        return 0.5

    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique_scores)))
    best_thr = 0.5
    best_val = -np.inf

    for thr in candidates:
        y_pred = (y_score >= thr).astype(int)
        if metric == 'f1':
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            rec = recall_score(y_true, y_pred, zero_division=0)
            spec = specificity_score(y_true, y_pred)
            spec = -1e9 if np.isnan(spec) else spec
            val = rec + spec - 1
        else:
            raise ValueError(f'Unknown threshold metric: {metric}')

        if val > best_val:
            best_val = val
            best_thr = float(thr)

    return best_thr



def compute_binary_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    out = {}
    try:
        out['auc'] = roc_auc_score(y_true, y_score)
    except ValueError:
        out['auc'] = np.nan

    try:
        out['auprc'] = average_precision_score(y_true, y_score)
    except ValueError:
        out['auprc'] = np.nan

    out['acc'] = float((y_pred == y_true).mean())
    out['f1'] = f1_score(y_true, y_pred, zero_division=0)
    out['precision'] = precision_score(y_true, y_pred, zero_division=0)
    out['recall'] = recall_score(y_true, y_pred, zero_division=0)
    out['specificity'] = specificity_score(y_true, y_pred)

    try:
        out['sens95'] = sensitivity_at_specificity(y_true, y_score)
    except ValueError:
        out['sens95'] = np.nan

    try:
        out['brier'] = brier_score_loss(y_true, y_score)
    except ValueError:
        out['brier'] = np.nan

    out['threshold'] = float(threshold)
    return out



def infer_feature_dim(feat_dir: str, ids):
    for pid in ids:
        path = os.path.join(feat_dir, f'{pid}.pt')
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu', weights_only=True)
            feats = data['feats']
            return int(feats.shape[1])
    raise FileNotFoundError('Could not infer in_dim: no valid .pt feature file found.')



def run_fullbag_inference(model, loader, device):
    scores, labels, pids = [], [], []
    model.eval()
    with torch.no_grad():
        for feats, ys, batch_pids in loader:
            feats = feats.to(device)
            logits, _, _ = model(feats)
            prob = torch.sigmoid(logits).cpu().numpy()
            scores.append(prob)
            labels.append(ys.numpy())
            pids.extend(list(batch_pids))

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels, pids



def save_best_checkpoint(path, model, optimizer, epoch, threshold, metrics, args, fold):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'fold': fold,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'thresholds': threshold,
            'best_metrics': metrics,
            'args': vars(args),
        },
        path,
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--label_cols', nargs='+', required=True)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--train_max_feats', type=int, default=32)
    parser.add_argument('--train_instance_strategy', type=str, default='random')
    parser.add_argument('--valid_max_feats', type=int, default=-1)
    parser.add_argument('--valid_instance_strategy', type=str, default='all')

    parser.add_argument('--architecture', type=str, default='attention')
    parser.add_argument('--threshold_metric', type=str, default='youden', choices=['youden', 'f1'])
    parser.add_argument('--use_combined_loss', action='store_true')
    parser.add_argument('--auc_weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_ckpt', action='store_true')

    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    enable_jupyter_logging(f'outputs/train_{ts}.log')
    set_seed(args.seed)

    print('===== TRAINING START =====')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('==========================\n')

    df = pd.read_excel(args.labels_csv) if args.labels_csv.endswith(('xls', 'xlsx')) else pd.read_csv(args.labels_csv)
    if 'id' not in df.columns:
        raise KeyError("Labels file must contain column 'id'.")
    df['id'] = df['id'].astype(str)

    y_all = df[args.label_cols].values.astype(int)
    l = y_all.shape[1]
    if l != 1:
        print('⚠️ 当前脚本完整支持多标签划分，但阈值选择与 ACC/F1 主要按单标签场景设计。')

    if l == 1:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        y_target = y_all[:, 0]
        print('Single-label task: StratifiedKFold')
    else:
        splitter = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        y_target = y_all
        print('Multi-label task: MultilabelStratifiedKFold')

    in_dim = infer_feature_dim(args.feat_dir, df['id'].tolist())
    print(f'Inferred feature dimension: {in_dim}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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
        train_eval_ds = PatientFeatureDataset(
            args.feat_dir,
            train_df,
            args.label_cols,
            max_feats=args.valid_max_feats,
            instance_strategy=args.valid_instance_strategy,
            random_seed=args.seed,
        )
        valid_ds = PatientFeatureDataset(
            args.feat_dir,
            valid_df,
            args.label_cols,
            max_feats=args.valid_max_feats,
            instance_strategy=args.valid_instance_strategy,
            random_seed=args.seed,
        )

        tr_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_train_fixed,
        )
        tr_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_eval_variable,
        )
        va_loader = DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_eval_variable,
        )

        model = PatientMILFeatures(
            in_dim=in_dim,
            n_labels=l,
            d_hidden_attn=128,
            dropout=0.3,
            architecture=args.architecture,
        ).to(device)

        pos_weight = compute_pos_weight(train_df[args.label_cols].values.astype('float32')).to(device)
        criterion = (
            CombinedLoss(pos_weight, args.auc_weight)
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
            'threshold': np.nan,
            'best_epoch': -1,
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

            train_scores, train_labels, _ = run_fullbag_inference(model, tr_eval_loader, device)
            valid_scores, valid_labels, _ = run_fullbag_inference(model, va_loader, device)

            if l == 1:
                threshold = find_best_threshold_binary(
                    train_labels[:, 0], train_scores[:, 0], metric=args.threshold_metric
                )
                metrics = compute_binary_metrics(valid_labels[:, 0], valid_scores[:, 0], threshold)
            else:
                threshold = 0.5
                metrics = compute_binary_metrics(valid_labels[:, 0], valid_scores[:, 0], threshold)

            is_best = (not np.isnan(metrics['auc'])) and (metrics['auc'] > best_metrics['auc'])
            if is_best:
                for k, v in metrics.items():
                    best_metrics[k] = v
                best_metrics['best_epoch'] = epoch
                if args.save_ckpt:
                    save_best_checkpoint(
                        f'outputs/best_fold{fold}.pt',
                        model,
                        optimizer,
                        epoch,
                        threshold,
                        best_metrics,
                        args,
                        fold,
                    )

            msg = (
                f"Epoch {epoch:03d}: loss={epoch_loss:.4f}  "
                f"AUROC={metrics['auc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
                f"ACC={metrics['acc']:.4f}  F1={metrics['f1']:.4f}  "
                f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
                f"Specificity={metrics['specificity']:.4f}  Sens@95Spec={metrics['sens95']:.4f}  "
                f"Brier={metrics['brier']:.4f}  Thr={metrics['threshold']:.4f}"
            )
            if is_best:
                msg += '  ✓ NEW BEST'
            print(msg)

            scheduler.step(metrics['auc'] if not np.isnan(metrics['auc']) else -1.0)

        print(
            f"Fold {fold} BEST | "
            f"Epoch={best_metrics['best_epoch']:03d}  "
            f"AUROC={best_metrics['auc']:.4f}  AUPRC={best_metrics['auprc']:.4f}  "
            f"ACC={best_metrics['acc']:.4f}  F1={best_metrics['f1']:.4f}  "
            f"Precision={best_metrics['precision']:.4f}  Recall={best_metrics['recall']:.4f}  "
            f"Specificity={best_metrics['specificity']:.4f}  Sens@95Spec={best_metrics['sens95']:.4f}  "
            f"Brier={best_metrics['brier']:.4f}  Thr={best_metrics['threshold']:.4f}"
        )

        row = {'fold': fold}
        row.update(best_metrics)
        all_folds_metrics.append(row)

    metrics_df = pd.DataFrame(all_folds_metrics)
    os.makedirs('outputs', exist_ok=True)
    summary_csv = f'outputs/cv_summary_{ts}.csv'
    metrics_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    print('\n===== CROSS-VALIDATION SUMMARY (mean ± std) =====')
    for key in ['auc', 'auprc', 'acc', 'f1', 'precision', 'recall', 'specificity', 'sens95', 'brier']:
        vals = metrics_df[key].astype(float).values
        print(f'{key.upper():<12}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}')
    print(f'Threshold mean: {np.nanmean(metrics_df["threshold"]):.4f}')
    print(f'Summary csv saved to: {summary_csv}')
    print('\nTraining complete.')


if __name__ == '__main__':
    main()
