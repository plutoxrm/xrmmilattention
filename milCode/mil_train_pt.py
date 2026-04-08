import torch
import torch.nn as nn
import torch.nn.functional as F


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

            if (~mask).all(dim=1).any():
                raise ValueError("Found a bag with all positions masked out.")

            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits, dim=1)  # [B, N]

        if mask is not None:
            attn = attn * mask.float()
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-12)

        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)  # [B, D]
        return pooled, attn


class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.net = nn.Linear(in_dim, n_labels)

    def forward(self, x):
        return self.net(x)


class MLPClassifierHead(nn.Module):
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
    Attention MIL + prototype auxiliary branch

    主干:
        feats -> attention pooling -> pooled -> classifier -> logits

    prototype 分支:
        feats -> proto_proj -> instance embedding z
             -> 与正/负 prototypes 求 cosine similarity
             -> 返回 s_pos / s_neg / margin
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
        use_prototype: bool = True,
        proto_dim: int = 128,
        num_pos_prototypes: int = 4,
        num_neg_prototypes: int = 4,
        proto_temperature: float = 0.1,
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

        self.use_prototype = use_prototype
        self.proto_temperature = proto_temperature

        if self.use_prototype:
            self.proto_proj = nn.Sequential(
                nn.Linear(in_dim, proto_dim),
                nn.ReLU(),
                nn.LayerNorm(proto_dim),
            )

            self.pos_prototypes = nn.Parameter(
                torch.randn(num_pos_prototypes, proto_dim)
            )
            self.neg_prototypes = nn.Parameter(
                torch.randn(num_neg_prototypes, proto_dim)
            )

            self._init_prototype_branch()

    def _init_prototype_branch(self):
        for m in self.proto_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_prototypes, mean=0.0, std=0.02)
        nn.init.normal_(self.neg_prototypes, mean=0.0, std=0.02)

    def _compute_prototype_scores(self, feats, mask=None):
        """
        feats: [B, N, D]
        return:
            z:       [B, N, P]
            s_pos:   [B, N]
            s_neg:   [B, N]
            margin:  [B, N] = s_pos - s_neg
        """
        z = self.proto_proj(feats)                       # [B, N, P]
        z_norm = F.normalize(z, p=2, dim=-1)

        pos_proto = F.normalize(self.pos_prototypes, p=2, dim=-1)  # [Kp, P]
        neg_proto = F.normalize(self.neg_prototypes, p=2, dim=-1)  # [Kn, P]

        sim_pos = torch.einsum("bnp,kp->bnk", z_norm, pos_proto) / self.proto_temperature
        sim_neg = torch.einsum("bnp,kp->bnk", z_norm, neg_proto) / self.proto_temperature

        s_pos, _ = sim_pos.max(dim=-1)   # [B, N]
        s_neg, _ = sim_neg.max(dim=-1)   # [B, N]
        margin = s_pos - s_neg           # [B, N]

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            s_pos = s_pos.masked_fill(~mask, 0.0)
            s_neg = s_neg.masked_fill(~mask, 0.0)
            margin = margin.masked_fill(~mask, 0.0)
            z = z.masked_fill(~mask.unsqueeze(-1), 0.0)

        return z, s_pos, s_neg, margin

    def forward(self, feats, mask=None):
        pooled, attn = self.pool(feats, mask=mask)
        logits = self.classifier(pooled)

        out = {
            "logits": logits,
            "pooled": pooled,
            "attn": attn,
            "inst_embed": None,
            "s_pos": None,
            "s_neg": None,
            "margin": None,
        }

        if self.use_prototype:
            z, s_pos, s_neg, margin = self._compute_prototype_scores(feats, mask=mask)
            out["inst_embed"] = z
            out["s_pos"] = s_pos
            out["s_neg"] = s_neg
            out["margin"] = margin

        return out