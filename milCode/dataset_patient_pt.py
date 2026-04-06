import os
import torch
from torch.utils.data import Dataset


class PatientFeatureDataset(Dataset):
    def __init__(
        self,
        feat_dir: str,
        labels_df,
        label_cols,
        max_feats: int = -1,
        instance_strategy: str = "top_norm",
        random_seed: int = 42,
    ):
        super().__init__()
        self.feat_dir = feat_dir
        self.df = labels_df.reset_index(drop=True)
        self.label_cols = label_cols
        self.max_feats = max_feats
        self.instance_strategy = instance_strategy
        self.random_seed = random_seed

        if "id" not in self.df.columns:
            raise KeyError("Labels file must contain column 'id'.")

        self.patient_ids = self.df["id"].astype(str).tolist()
        self.labels = self.df[self.label_cols].values.astype("float32")

    def __len__(self):
        return len(self.patient_ids)

    def _select_diverse_instances(self, feats, max_feats):
        """
        贪心式选择“彼此尽量不相似”的实例：
        先选范数最大的，再不断选与已选集合最不相似的。
        """
        n = feats.shape[0]
        norms = torch.norm(feats, dim=1)
        selected_indices = [norms.argmax().item()]

        for _ in range(min(max_feats - 1, n - 1)):
            best_diversity = -1.0
            best_idx = None

            for idx in range(n):
                if idx in selected_indices:
                    continue

                min_diversity = float("inf")
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
                for idx in range(n):
                    if idx not in selected_indices:
                        best_idx = idx
                        break

            if best_idx is not None:
                selected_indices.append(best_idx)
            else:
                break

        return torch.tensor(selected_indices, dtype=torch.long)

    def _sample_indices(self, feats, pid):
        """
        当 N >= max_feats 时，根据策略选出 max_feats 个实例索引。
        """
        n = feats.shape[0]

        if self.instance_strategy == "top_norm":
            norms = torch.norm(feats, dim=1)
            idxs = torch.topk(norms, self.max_feats)[1]

        elif self.instance_strategy == "fixed_random":
            # 同一个 pid 每次都固定随机，保证可复现
            try:
                seed = int(pid) % (2**32)
            except ValueError:
                seed = abs(hash(pid)) % (2**32)
            g = torch.Generator()
            g.manual_seed(seed + self.random_seed)
            idxs = torch.randperm(n, generator=g)[: self.max_feats]

        elif self.instance_strategy == "random":
            idxs = torch.randperm(n)[: self.max_feats]

        elif self.instance_strategy == "diversity":
            idxs = self._select_diverse_instances(feats, self.max_feats)

        else:
            raise ValueError(f"Unknown instance_strategy: {self.instance_strategy}")

        return idxs.long()

    def _select_instances_with_mask(self, feats, pid):
        """
        返回：
            feats_out: [N', D]
            mask:      [N']，True 表示真实实例，False 表示 padding
        """
        if feats.ndim != 2:
            raise ValueError(f"Expected feats to be 2D [N, D], got shape={tuple(feats.shape)}")

        n, d = feats.shape
        if n == 0:
            raise ValueError(f"Patient {pid} has zero instance features in its .pt file.")

        # 不限制数量：直接返回全部实例
        if self.instance_strategy == "all" or self.max_feats is None or self.max_feats <= 0:
            mask = torch.ones(n, dtype=torch.bool)
            return feats, mask

        # 如果实例数不足，就 padding 到 max_feats，并生成 mask
        if n < self.max_feats:
            padded = torch.zeros(self.max_feats, d, dtype=feats.dtype)
            padded[:n] = feats

            mask = torch.zeros(self.max_feats, dtype=torch.bool)
            mask[:n] = True
            return padded, mask

        # 否则按照策略采样 max_feats 个实例
        idxs = self._sample_indices(feats, pid)
        feats = feats[idxs]
        mask = torch.ones(feats.shape[0], dtype=torch.bool)
        return feats, mask

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        path = os.path.join(self.feat_dir, f"{pid}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")

        data = torch.load(path, map_location="cpu", weights_only=True)
        if "feats" not in data:
            raise KeyError(f"'feats' not found in feature file: {path}")

        feats = data["feats"].float()
        feats, mask = self._select_instances_with_mask(feats, pid)

        return feats, label, pid, mask