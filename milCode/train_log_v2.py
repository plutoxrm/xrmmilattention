import argparse
import builtins
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

warnings.filterwarnings("ignore", message="Input data has no positive sample")

from dataset_patient_pt import PatientFeatureDataset
from mil_train_pt import PatientMILFeatures
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from libauc.losses import AUCMLoss


def enable_jupyter_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    original_print = builtins.print

    def logged_print(*args, **kwargs):
        original_print(*args, **kwargs)
        log_kwargs = kwargs.copy()
        log_kwargs.pop("file", None)
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


def collate_pad_bags(batch):
    """
    支持训练/验证都使用全部实例。
    不同患者实例数不同，因此在 batch 内按最长 bag 做 padding。
    """
    feats_list, ys, pids, masks_list = zip(*batch)

    batch_size = len(feats_list)
    max_n = max(x.shape[0] for x in feats_list)
    feat_dim = feats_list[0].shape[1]
    dtype = feats_list[0].dtype

    feats_padded = torch.zeros(batch_size, max_n, feat_dim, dtype=dtype)
    masks_padded = torch.zeros(batch_size, max_n, dtype=torch.bool)

    for i, (feats, mask) in enumerate(zip(feats_list, masks_list)):
        n = feats.shape[0]
        feats_padded[i, :n] = feats
        masks_padded[i, :n] = mask

    ys = torch.stack(ys, dim=0)
    return feats_padded, ys, pids, masks_padded


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


def find_best_threshold_binary(y_true, y_score, metric="youden"):
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
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            rec = recall_score(y_true, y_pred, zero_division=0)
            spec = specificity_score(y_true, y_pred)
            spec = -1e9 if np.isnan(spec) else spec
            val = rec + spec - 1
        else:
            raise ValueError(f"Unknown threshold metric: {metric}")

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
        out["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        out["auc"] = np.nan

    try:
        out["auprc"] = average_precision_score(y_true, y_score)
    except ValueError:
        out["auprc"] = np.nan

    out["acc"] = float((y_pred == y_true).mean())
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["specificity"] = specificity_score(y_true, y_pred)

    try:
        out["sens95"] = sensitivity_at_specificity(y_true, y_score)
    except ValueError:
        out["sens95"] = np.nan

    try:
        out["brier"] = brier_score_loss(y_true, y_score)
    except ValueError:
        out["brier"] = np.nan

    out["threshold"] = float(threshold)
    return out


def infer_feature_dim(feat_dir: str, ids):
    for pid in ids:
        path = os.path.join(feat_dir, f"{pid}.pt")
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=True)
            feats = data["feats"]
            return int(feats.shape[1])
    raise FileNotFoundError("Could not infer in_dim: no valid .pt feature file found.")


def run_fullbag_inference(model, loader, device):
    scores, labels, pids = [], [], []
    model.eval()

    with torch.no_grad():
        for feats, ys, batch_pids, masks in loader:
            feats = feats.to(device)
            ys = ys.to(device)
            masks = masks.to(device)

            out = model(feats, mask=masks)
            prob = torch.sigmoid(out["logits"]).cpu().numpy()

            scores.append(prob)
            labels.append(ys.cpu().numpy())
            pids.extend(list(batch_pids))

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels, pids


def save_best_checkpoint(path, model, optimizer, epoch, threshold, metrics, args, fold):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "fold": fold,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "thresholds": threshold,
            "best_metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def _masked_topk_indices(values_1d, valid_mask_1d, k, largest=True):
    """
    values_1d: [N]
    valid_mask_1d: [N] bool
    return: indices on original axis
    """
    valid_indices = torch.where(valid_mask_1d)[0]
    if valid_indices.numel() == 0:
        return None

    k = min(k, valid_indices.numel())
    valid_values = values_1d[valid_indices]
    top_local = torch.topk(valid_values, k=k, largest=largest).indices
    return valid_indices[top_local]


def compute_proto_loss(
    attn,
    s_pos,
    s_neg,
    labels,
    mask,
    k_pos=3,
    k_neg=3,
):
    """
    单标签二分类版本：
    - 正 bag：取 attention top-k 作为弱正实例，目标=正类
    - 负 bag：取 s_pos top-k 作为 hard negatives，目标=负类
    """
    device = attn.device
    losses = []

    y = labels[:, 0]

    for b in range(attn.size(0)):
        valid = mask[b].bool()
        if valid.sum() == 0:
            continue

        if y[b] >= 0.5:
            idx = _masked_topk_indices(attn[b], valid, k=k_pos, largest=True)
            if idx is None:
                continue
            logits_inst = torch.stack([s_neg[b, idx], s_pos[b, idx]], dim=1)  # [K, 2]
            targets = torch.ones(logits_inst.size(0), dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits_inst, targets))
        else:
            idx = _masked_topk_indices(s_pos[b], valid, k=k_neg, largest=True)
            if idx is None:
                continue
            logits_inst = torch.stack([s_neg[b, idx], s_pos[b, idx]], dim=1)  # [K, 2]
            targets = torch.zeros(logits_inst.size(0), dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits_inst, targets))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def compute_margin_loss(
    attn,
    s_pos,
    margin,
    labels,
    mask,
    k_pos=3,
    k_neg=3,
    gamma_pos=0.2,
    gamma_neg=0.2,
):
    """
    - 正 bag：attention top-k，希望 margin = s_pos - s_neg 足够大
    - 负 bag：s_pos top-k hard negatives，希望 margin 足够小
    """
    device = attn.device
    losses = []

    y = labels[:, 0]

    for b in range(attn.size(0)):
        valid = mask[b].bool()
        if valid.sum() == 0:
            continue

        if y[b] >= 0.5:
            idx = _masked_topk_indices(attn[b], valid, k=k_pos, largest=True)
            if idx is None:
                continue
            loss_pos = F.relu(gamma_pos - margin[b, idx]).mean()
            losses.append(loss_pos)
        else:
            idx = _masked_topk_indices(s_pos[b], valid, k=k_neg, largest=True)
            if idx is None:
                continue
            loss_neg = F.relu(margin[b, idx] + gamma_neg).mean()
            losses.append(loss_neg)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def attention_entropy_regularizer(attn, mask):
    """
    返回 sum(p log p) 的均值（<= 0）
    把这个量加到总 loss 中，会鼓励更高熵、避免 attention 过于尖锐。
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    p = attn.clamp_min(1e-12) * mask.float()
    reg = (p * torch.log(p.clamp_min(1e-12))).sum(dim=1)  # <= 0
    return reg.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--label_cols", nargs="+", required=True)

    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)

    # 按你的要求：训练和验证都使用全部实例
    parser.add_argument("--train_max_feats", type=int, default=-1)
    parser.add_argument("--train_instance_strategy", type=str, default="all")
    parser.add_argument("--valid_max_feats", type=int, default=-1)
    parser.add_argument("--valid_instance_strategy", type=str, default="all")

    parser.add_argument("--architecture", type=str, default="attention", choices=["attention"])
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="linear",
        choices=["linear", "mlp", "mlp_ln_dropout"],
    )
    parser.add_argument("--classifier_hidden_dim", type=int, default=256)
    parser.add_argument("--classifier_dropout", type=float, default=0.3)
    parser.add_argument("--attn_hidden_dim", type=int, default=128)
    parser.add_argument("--attn_dropout", type=float, default=0.0)

    parser.add_argument("--threshold_metric", type=str, default="youden", choices=["youden", "f1"])
    parser.add_argument("--use_combined_loss", action="store_true")
    parser.add_argument("--auc_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_ckpt", action="store_true")

    # prototype branch
    parser.add_argument("--use_prototype", action="store_true")
    parser.add_argument("--proto_dim", type=int, default=128)
    parser.add_argument("--num_pos_prototypes", type=int, default=4)
    parser.add_argument("--num_neg_prototypes", type=int, default=4)
    parser.add_argument("--proto_temperature", type=float, default=0.1)

    parser.add_argument("--proto_topk_pos", type=int, default=2)
    parser.add_argument("--proto_topk_neg", type=int, default=2)
    parser.add_argument("--lambda_proto", type=float, default=0.2)
    parser.add_argument("--lambda_margin", type=float, default=0.1)
    parser.add_argument("--lambda_ent", type=float, default=1e-3)
    parser.add_argument("--gamma_pos", type=float, default=0.2)
    parser.add_argument("--gamma_neg", type=float, default=0.2)
    parser.add_argument("--proto_warmup_epochs", type=int, default=0)

    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    enable_jupyter_logging(f"outputs/train_{ts}.log")
    set_seed(args.seed)

    print("===== TRAINING START =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==========================\n")

    df = pd.read_excel(args.labels_csv) if args.labels_csv.endswith(("xls", "xlsx")) else pd.read_csv(args.labels_csv)
    if "id" not in df.columns:
        raise KeyError("Labels file must contain column 'id'.")
    df["id"] = df["id"].astype(str)

    y_all = df[args.label_cols].values.astype(int)
    l = y_all.shape[1]
    if l != 1:
        print("⚠️ 当前 prototype 辅助监督主要按单标签二分类设计。若多标签，请先只跑单标签任务。")

    if l == 1:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        y_target = y_all[:, 0]
        print("Single-label task: StratifiedKFold")
    else:
        splitter = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        y_target = y_all
        print("Multi-label task: MultilabelStratifiedKFold")

    in_dim = infer_feature_dim(args.feat_dir, df["id"].tolist())
    print(f"Inferred feature dimension: {in_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_folds_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df, y_target), 1):
        print(f"\n===== Fold {fold}/{args.folds} =====")

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
            collate_fn=collate_pad_bags,
        )
        tr_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_pad_bags,
        )
        va_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_pad_bags,
        )

        model = PatientMILFeatures(
            in_dim=in_dim,
            n_labels=l,
            d_hidden_attn=args.attn_hidden_dim,
            architecture=args.architecture,
            classifier_type=args.classifier_type,
            classifier_hidden_dim=args.classifier_hidden_dim,
            classifier_dropout=args.classifier_dropout,
            attn_dropout=args.attn_dropout,
            use_prototype=args.use_prototype,
            proto_dim=args.proto_dim,
            num_pos_prototypes=args.num_pos_prototypes,
            num_neg_prototypes=args.num_neg_prototypes,
            proto_temperature=args.proto_temperature,
        ).to(device)

        pos_weight = compute_pos_weight(train_df[args.label_cols].values.astype("float32")).to(device)
        criterion = (
            CombinedLoss(pos_weight, args.auc_weight)
            if args.use_combined_loss
            else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            threshold=0.002,
            cooldown=3,
            min_lr=1e-6,
        )

        best_metrics = {
            "auc": -1.0,
            "auprc": np.nan,
            "acc": np.nan,
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "specificity": np.nan,
            "sens95": np.nan,
            "brier": np.nan,
            "threshold": np.nan,
            "best_epoch": -1,
        }

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_bag_loss = 0.0
            epoch_proto_loss = 0.0
            epoch_margin_loss = 0.0
            epoch_ent_reg = 0.0

            for feats, ys, _, masks in tr_loader:
                feats = feats.to(device)
                ys = ys.to(device)
                masks = masks.to(device)

                out = model(feats, mask=masks)

                bag_loss = criterion(out["logits"], ys)
                proto_loss = torch.tensor(0.0, device=device)
                margin_loss = torch.tensor(0.0, device=device)
                ent_reg = torch.tensor(0.0, device=device)

                if args.use_prototype and epoch > args.proto_warmup_epochs:
                    proto_loss = compute_proto_loss(
                        attn=out["attn"],
                        s_pos=out["s_pos"],
                        s_neg=out["s_neg"],
                        labels=ys,
                        mask=masks,
                        k_pos=args.proto_topk_pos,
                        k_neg=args.proto_topk_neg,
                    )
                    margin_loss = compute_margin_loss(
                        attn=out["attn"],
                        s_pos=out["s_pos"],
                        margin=out["margin"],
                        labels=ys,
                        mask=masks,
                        k_pos=args.proto_topk_pos,
                        k_neg=args.proto_topk_neg,
                        gamma_pos=args.gamma_pos,
                        gamma_neg=args.gamma_neg,
                    )
                    ent_reg = attention_entropy_regularizer(out["attn"], masks)

                loss = (
                    bag_loss
                    + args.lambda_proto * proto_loss
                    + args.lambda_margin * margin_loss
                    + args.lambda_ent * ent_reg
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = feats.size(0)
                epoch_loss += loss.item() * bs
                epoch_bag_loss += bag_loss.item() * bs
                epoch_proto_loss += proto_loss.item() * bs
                epoch_margin_loss += margin_loss.item() * bs
                epoch_ent_reg += ent_reg.item() * bs

            epoch_loss /= len(tr_loader.dataset)
            epoch_bag_loss /= len(tr_loader.dataset)
            epoch_proto_loss /= len(tr_loader.dataset)
            epoch_margin_loss /= len(tr_loader.dataset)
            epoch_ent_reg /= len(tr_loader.dataset)

            train_scores, train_labels, _ = run_fullbag_inference(model, tr_eval_loader, device)
            valid_scores, valid_labels, _ = run_fullbag_inference(model, va_loader, device)

            if l == 1:
                threshold = find_best_threshold_binary(
                    train_labels[:, 0],
                    train_scores[:, 0],
                    metric=args.threshold_metric,
                )
                metrics = compute_binary_metrics(
                    valid_labels[:, 0],
                    valid_scores[:, 0],
                    threshold,
                )
            else:
                threshold = 0.5
                metrics = compute_binary_metrics(
                    valid_labels[:, 0],
                    valid_scores[:, 0],
                    threshold,
                )

            is_best = (not np.isnan(metrics["auc"])) and (metrics["auc"] > best_metrics["auc"])
            if is_best:
                for k, v in metrics.items():
                    best_metrics[k] = v
                best_metrics["best_epoch"] = epoch

                if args.save_ckpt:
                    save_best_checkpoint(
                        f"outputs/best_fold{fold}.pt",
                        model,
                        optimizer,
                        epoch,
                        threshold,
                        best_metrics,
                        args,
                        fold,
                    )

            msg = (
                f"Epoch {epoch:03d}: "
                f"loss={epoch_loss:.4f}  "
                f"bag={epoch_bag_loss:.4f}  "
                f"proto={epoch_proto_loss:.4f}  "
                f"margin={epoch_margin_loss:.4f}  "
                f"entReg={epoch_ent_reg:.4f}  "
                f"AUROC={metrics['auc']:.4f}  "
                f"AUPRC={metrics['auprc']:.4f}  "
                f"ACC={metrics['acc']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"Precision={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}  "
                f"Specificity={metrics['specificity']:.4f}  "
                f"Sens@95Spec={metrics['sens95']:.4f}  "
                f"Brier={metrics['brier']:.4f}  "
                f"Thr={metrics['threshold']:.4f}"
            )
            if is_best:
                msg += "  ✓ NEW BEST"
            print(msg)

            scheduler.step(metrics["auc"] if not np.isnan(metrics["auc"]) else -1.0)

        print(
            f"Fold {fold} BEST | "
            f"Epoch={best_metrics['best_epoch']:03d}  "
            f"AUROC={best_metrics['auc']:.4f}  "
            f"AUPRC={best_metrics['auprc']:.4f}  "
            f"ACC={best_metrics['acc']:.4f}  "
            f"F1={best_metrics['f1']:.4f}  "
            f"Precision={best_metrics['precision']:.4f}  "
            f"Recall={best_metrics['recall']:.4f}  "
            f"Specificity={best_metrics['specificity']:.4f}  "
            f"Sens@95Spec={best_metrics['sens95']:.4f}  "
            f"Brier={best_metrics['brier']:.4f}  "
            f"Thr={best_metrics['threshold']:.4f}"
        )

        row = {"fold": fold}
        row.update(best_metrics)
        all_folds_metrics.append(row)

    metrics_df = pd.DataFrame(all_folds_metrics)
    os.makedirs("outputs", exist_ok=True)
    summary_csv = f"outputs/cv_summary_{ts}.csv"
    metrics_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n===== CROSS-VALIDATION SUMMARY (mean ± std) =====")
    for key in ["auc", "auprc", "acc", "f1", "precision", "recall", "specificity", "sens95", "brier"]:
        vals = metrics_df[key].astype(float).values
        print(f"{key.upper():<12}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
    print(f"Threshold mean: {np.nanmean(metrics_df['threshold']):.4f}")
    print(f"Summary csv saved to: {summary_csv}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()


"""
python train_log_v2.py ^
  --feat_dir ".\encoder_features" ^
  --labels_csv ".\labels03.xlsx" ^
  --label_cols "代谢慢病" ^
  --architecture attention ^
  --classifier_type linear ^
  --epochs 70 ^
  --batch_size 2 ^
  --lr 5e-4 ^
  --weight_decay 1e-4 ^
  --folds 5 ^
  --train_max_feats -1 ^
  --train_instance_strategy all ^
  --valid_max_feats -1 ^
  --valid_instance_strategy all ^
  --use_combined_loss ^
  --auc_weight 0.5 ^
  --use_prototype
"""