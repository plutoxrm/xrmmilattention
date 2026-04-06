import torch
import torch.nn as nn


class AttentionMILPool(nn.Module):
    """
    Standard Attention MIL with optional padding mask.

    Input:
        feats: [B, N, D]
        mask:  [B, N], bool
               True  -> real instance
               False -> padding
    Output:
        pooled: [B, D]
        attn:   [B, N]
    """
    def __init__(self, in_dim: int, d_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, d_hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(d_hidden, 1)

    def forward(self, feats, mask=None):
        h = torch.tanh(self.fc1(feats))      # [B, N, H]
        h = self.dropout(h)
        logits = self.fc2(h).squeeze(-1)     # [B, N]

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()

            # 避免全 padding 的非法情况
            if (~mask).all(dim=1).any():
                raise ValueError("Found a bag with all positions masked out.")

            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits, dim=1)  # [B, N]

        # 数值安全：把 padding 位清零并重新归一化
        if mask is not None:
            attn = attn * mask.float()
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-12)

        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)  # [B, D]
        return pooled, attn


class LinearClassifierHead(nn.Module):
    """
    Attention + Linear
    """
    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.net = nn.Linear(in_dim, n_labels)

    def forward(self, x):
        return self.net(x)


class MLPClassifierHead(nn.Module):
    """
    Attention + MLP
    """
    def __init__(self, in_dim: int, n_labels: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, x):
        return self.net(x)


class MLPLNDropoutClassifierHead(nn.Module):
    """
    Attention + MLP + LayerNorm + Dropout
    """
    def __init__(
        self,
        in_dim: int,
        n_labels: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, x):
        return self.net(x)


class PatientMILFeatures(nn.Module):
    """
    只保留 Attention MIL，并支持 3 种分类头：
        1) linear
        2) mlp
        3) mlp_ln_dropout
    """
    def __init__(
        self,
        in_dim: int = 768,
        n_labels: int = 1,
        d_hidden_attn: int = 128,
        architecture: str = "attention",
        classifier_type: str = "linear",
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.3,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        if architecture != "attention":
            raise ValueError(
                f"Only 'attention' is supported now, but got architecture={architecture}"
            )

        self.pool = AttentionMILPool(
            in_dim=in_dim,
            d_hidden=d_hidden_attn,
            dropout=attn_dropout,
        )

        if classifier_type == "linear":
            self.classifier = LinearClassifierHead(in_dim=in_dim, n_labels=n_labels)
        elif classifier_type == "mlp":
            self.classifier = MLPClassifierHead(
                in_dim=in_dim,
                n_labels=n_labels,
                hidden_dim=classifier_hidden_dim,
            )
        elif classifier_type == "mlp_ln_dropout":
            self.classifier = MLPLNDropoutClassifierHead(
                in_dim=in_dim,
                n_labels=n_labels,
                hidden_dim=classifier_hidden_dim,
                dropout=classifier_dropout,
            )
        else:
            raise ValueError(
                f"Unknown classifier_type: {classifier_type}. "
                f"Choose from ['linear', 'mlp', 'mlp_ln_dropout']"
            )

    def forward(self, feats, mask=None):
        pooled, attn = self.pool(feats, mask=mask)
        logits = self.classifier(pooled)
        return logits, pooled, attn