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
        h = torch.tanh(self.fc1(feats))
        h = self.dropout(h)
        logits = self.fc2(h).squeeze(-1)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if (~mask).all(dim=1).any():
                raise ValueError("Found a bag with all positions masked out.")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits, dim=1)

        if mask is not None:
            attn = attn * mask.float()
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-12)

        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)
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


def _apply_mask(x: torch.Tensor, mask: torch.Tensor | None):
    if mask is None:
        return x
    return x * mask.unsqueeze(-1).to(dtype=x.dtype)


def build_similarity_graph(
    feats: torch.Tensor,
    mask: torch.Tensor | None = None,
    topk: int = 8,
    similarity: str = "cosine",
    sigma: float = 1.0,
    eps: float = 1e-8,
):
    """
    Build a dense similarity graph for each bag.

    Args:
        feats: [B, N, D]
        mask:  [B, N], bool
        topk:  keep top-k neighbors per node; <=0 means full graph
        similarity: 'cosine' or 'rbf'
        sigma: used when similarity == 'rbf'

    Returns:
        adj_norm: [B, N, N] normalized adjacency for GCN
        adj_raw:  [B, N, N] weighted adjacency before normalization
    """
    if feats.ndim != 3:
        raise ValueError(f"Expected feats to be [B, N, D], got shape={tuple(feats.shape)}")

    b, n, _ = feats.shape
    device = feats.device

    if mask is None:
        mask = torch.ones(b, n, dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)

    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)

    if similarity == "cosine":
        feats_norm = F.normalize(feats, p=2, dim=-1, eps=eps)
        sim = torch.bmm(feats_norm, feats_norm.transpose(1, 2))
        sim = (sim + 1.0) * 0.5
    elif similarity == "rbf":
        dist = torch.cdist(feats, feats, p=2)
        sim = torch.exp(-(dist * dist) / max(sigma, eps))
    else:
        raise ValueError(f"Unknown similarity: {similarity}")

    sim = sim * valid_pair.float()

    eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
    sim_wo_self = sim.masked_fill(eye, 0.0)

    if topk is not None and topk > 0 and n > 1:
        k = min(int(topk), n - 1)
        candidate = sim_wo_self.masked_fill(~valid_pair, -1.0)
        values, indices = torch.topk(candidate, k=k, dim=-1)
        knn_mask = torch.zeros_like(sim_wo_self, dtype=torch.bool)
        knn_mask.scatter_(2, indices, values > 0)
        adj = sim_wo_self * knn_mask.float()
    else:
        adj = sim_wo_self

    adj = 0.5 * (adj + adj.transpose(1, 2))

    valid_diag = eye.float() * (mask.unsqueeze(1).float() * mask.unsqueeze(2).float())
    adj = adj + valid_diag
    adj = adj * valid_pair.float()

    degree = adj.sum(dim=-1)
    deg_inv_sqrt = degree.clamp_min(eps).pow(-0.5)
    adj_norm = adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
    adj_norm = adj_norm * valid_pair.float()

    return adj_norm, adj


class DenseGCNLayer(nn.Module):
    """
    Dense GCN layer:
        H' = LN( A_hat H W + residual ) -> ReLU -> Dropout
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_residual = residual
        if residual:
            self.residual_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x, adj_norm, mask=None):
        out = torch.bmm(adj_norm, self.linear(x))
        if self.use_residual:
            out = out + self.residual_proj(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = _apply_mask(out, mask)
        return out


class SimilarityGCNEncoder(nn.Module):
    """
    Construct graph from instance similarity inside each patient bag,
    then update node features with several dense GCN layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        topk: int = 8,
        similarity: str = "cosine",
        sigma: float = 1.0,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        hidden_dim = in_dim if hidden_dim is None or hidden_dim <= 0 else hidden_dim
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.similarity = similarity
        self.sigma = sigma

        self.input_proj = nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [DenseGCNLayer(hidden_dim, hidden_dim, dropout=dropout, residual=residual) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.last_adj = None
        self.last_adj_norm = None

    def forward(self, feats, mask=None):
        x = self.input_proj(feats)
        x = _apply_mask(x, mask)

        adj_norm, adj = build_similarity_graph(
            feats=feats,
            mask=mask,
            topk=self.topk,
            similarity=self.similarity,
            sigma=self.sigma,
        )
        self.last_adj = adj
        self.last_adj_norm = adj_norm

        for layer in self.layers:
            x = layer(x, adj_norm, mask=mask)

        x = self.output_norm(x)
        x = _apply_mask(x, mask)
        return x


class PatientMILFeatures(nn.Module):
    """
    Supports:
        1) attention         : vanilla Attention MIL
        2) graph_attention   : similarity graph -> GCN -> Attention MIL
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
        graph_hidden_dim: int = 0,
        graph_num_layers: int = 2,
        graph_topk: int = 8,
        graph_similarity: str = "cosine",
        graph_sigma: float = 1.0,
        graph_dropout: float = 0.1,
        graph_residual: bool = True,
    ):
        super().__init__()

        self.architecture = architecture
        feat_dim = in_dim

        if architecture == "attention":
            self.graph_encoder = None
        elif architecture == "graph_attention":
            feat_dim = in_dim if graph_hidden_dim is None or graph_hidden_dim <= 0 else graph_hidden_dim
            self.graph_encoder = SimilarityGCNEncoder(
                in_dim=in_dim,
                hidden_dim=feat_dim,
                num_layers=graph_num_layers,
                topk=graph_topk,
                similarity=graph_similarity,
                sigma=graph_sigma,
                dropout=graph_dropout,
                residual=graph_residual,
            )
        else:
            raise ValueError(
                "Unknown architecture: "
                f"{architecture}. Choose from ['attention', 'graph_attention']"
            )

        self.pool = AttentionMILPool(
            in_dim=feat_dim,
            d_hidden=d_hidden_attn,
            dropout=attn_dropout,
        )

        if classifier_type == "linear":
            self.classifier = LinearClassifierHead(in_dim=feat_dim, n_labels=n_labels)
        elif classifier_type == "mlp":
            self.classifier = MLPClassifierHead(
                in_dim=feat_dim,
                n_labels=n_labels,
                hidden_dim=classifier_hidden_dim,
            )
        elif classifier_type == "mlp_ln_dropout":
            self.classifier = MLPLNDropoutClassifierHead(
                in_dim=feat_dim,
                n_labels=n_labels,
                hidden_dim=classifier_hidden_dim,
                dropout=classifier_dropout,
            )
        else:
            raise ValueError(
                f"Unknown classifier_type: {classifier_type}. "
                "Choose from ['linear', 'mlp', 'mlp_ln_dropout']"
            )

    def forward(self, feats, mask=None):
        if self.graph_encoder is not None:
            feats = self.graph_encoder(feats, mask=mask)
        pooled, attn = self.pool(feats, mask=mask)
        logits = self.classifier(pooled)
        return logits, pooled, attn
