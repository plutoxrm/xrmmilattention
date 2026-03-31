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
        random_seed: int = 42
    ):
        super().__init__()
        self.feat_dir = feat_dir
        self.df = labels_df
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
        N = feats.shape[0]
        
        norms = torch.norm(feats, dim=1)
        selected_indices = [norms.argmax().item()]
        
        for _ in range(min(max_feats - 1, N - 1)):  # Can't select more than N
            best_diversity = -1
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
                    
                    # Handle zero-variance features
                    norm_a = feat_a_centered.norm()
                    norm_b = feat_b_centered.norm()
                    
                    if norm_a < 1e-6 or norm_b < 1e-6:
                        # Features with zero variance - treat as uncorrelated
                        diversity = 1.0
                    else:
                        corr = (feat_a_centered * feat_b_centered).sum() / (norm_a * norm_b)
                        corr = torch.clamp(corr, -1.0, 1.0)  # Numerical stability
                        diversity = 1 - abs(corr.item())
                    
                    min_diversity = min(min_diversity, diversity)
                
                if min_diversity > best_diversity:
                    best_diversity = min_diversity
                    best_idx = idx
            
            # Fallback: if no valid idx found, pick any unselected
            if best_idx is None:
                for idx in range(N):
                    if idx not in selected_indices:
                        best_idx = idx
                        break
            
            if best_idx is not None:
                selected_indices.append(best_idx)
            else:
                break  # No more indices to select
        
        return torch.tensor(selected_indices)
    
    def _select_instances(self, feats, pid):
        N = feats.shape[0]
        
        if self.instance_strategy == "all" or self.max_feats is None or self.max_feats <= 0:
            return feats
        
        if N < self.max_feats:
            padding = torch.zeros(self.max_feats - N, feats.shape[1])
            return torch.cat([feats, padding], dim=0)
        
        if self.instance_strategy == "top_norm":
            norms = torch.norm(feats, dim=1)
            idxs = torch.topk(norms, self.max_feats)[1]
        
        elif self.instance_strategy == "fixed_random":
            seed = int(pid) % (2**32)
            g = torch.Generator()
            g.manual_seed(seed)
            idxs = torch.randperm(N, generator=g)[:self.max_feats]
        
        elif self.instance_strategy == "random":
            idxs = torch.randperm(N)[:self.max_feats]
        
        elif self.instance_strategy == "diversity":
            idxs = self._select_diverse_instances(feats, self.max_feats)
        
        else:
            raise ValueError(f"Unknown instance_strategy: {self.instance_strategy}")
        
        return feats[idxs]
    
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = torch.tensor(self.labels[idx])
        
        path = os.path.join(self.feat_dir, f"{pid}.pt")
        data = torch.load(path, weights_only=True)
        feats = data["feats"]
        
        feats = self._select_instances(feats, pid)
        
        return feats, label, pid