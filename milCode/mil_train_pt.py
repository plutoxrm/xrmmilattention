import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMILPool(nn.Module):
    def __init__(self, in_dim: int, d_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, 1)

    def forward(self, feats):
        h = torch.tanh(self.fc1(feats))
        logits = self.fc2(h).squeeze(-1)
        attn = torch.softmax(logits, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)
        return pooled, attn


class GatedAttentionMILPool(nn.Module):
    def __init__(self, in_dim: int, d_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(d_hidden, 1)

    def forward(self, feats):
        A_V = self.attention_V(feats)
        A_U = self.attention_U(feats)
        A = self.attention_weights(A_V * A_U).squeeze(-1)
        attn = torch.softmax(A, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)
        return pooled, attn


class DSMILPool(nn.Module):
    def __init__(self, in_dim: int, d_hidden: int = 256, dropout: float = 0.2):
        super().__init__()

        self.instance_fc = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.instance_attention = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.Tanh(),
            nn.Linear(d_hidden // 2, 1)
        )

        self.bag_fc = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, feats):
        h_instance = self.instance_fc(feats)
        attn = self.instance_attention(h_instance)
        attn = torch.softmax(attn, dim=1)
        z_instance = (h_instance * attn).sum(dim=1)

        h_bag = self.bag_fc(feats)
        z_bag = h_bag.max(dim=1)[0]

        pooled = torch.cat([z_instance, z_bag], dim=1)
        return pooled, attn


class TransLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class PPEG(nn.Module):
    """
    Positional encoding via depth-wise conv on 2D reshaped tokens.
    输入:  [B, 1 + N, C]，其中第 0 个 token 是 cls token
    输出:  [B, 1 + N_pad, C]
    """
    def __init__(self, dim=512):
        super().__init__()
        self.proj_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.proj_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.proj_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token = x[:, 0:1, :]
        feat_token = x[:, 1:, :]

        feat_token = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        feat_token = feat_token + self.proj_7(feat_token) + self.proj_5(feat_token) + self.proj_3(feat_token)
        feat_token = feat_token.flatten(2).transpose(1, 2).contiguous()

        x = torch.cat((cls_token, feat_token), dim=1)
        return x


class TransMILPool(nn.Module):
    """
    输入 feats: [B, N, D]
    输出 pooled: [B, 512]
    """
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self._fc1 = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layer1 = TransLayer(dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.pos_layer = PPEG(dim=embed_dim)
        self.layer2 = TransLayer(dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, feats):
        """
        feats: [B, N, D]
        """
        B, N, _ = feats.shape
        h = self._fc1(feats)  # [B, N, 512]

        # ---- pad 到近似方阵，便于 2D 位置编码 ----
        H = int(math.ceil(math.sqrt(N)))
        W = H
        add_length = H * W - N
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N_pad, 512]

        # ---- 拼接 cls token ----
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)  # [B, 1 + N_pad, 512]

        # ---- transformer block 1 ----
        h, _ = self.layer1(h)

        # ---- PPEG ----
        h = self.pos_layer(h, H, W)

        # ---- transformer block 2 ----
        h, attn_weights = self.layer2(h)

        # ---- 取 cls token ----
        h = self.norm(h)
        pooled = h[:, 0]  # [B, 512]

        # 给一个“伪实例注意力”方便和你现有接口兼容
        # attn_weights: [B, heads, tgt_len, src_len]
        # 取 cls token 对所有 patch token 的注意力，再对 heads 求平均
        attn = None
        if attn_weights is not None:
            cls_attn = attn_weights[:, :, 0, 1:]       # [B, heads, N_pad]
            attn = cls_attn.mean(dim=1)                # [B, N_pad]
            attn = attn[:, :N]                         # 截回原始实例数
            attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)

        return pooled, attn


class PatientMILFeatures(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        n_labels: int = 1,
        d_hidden_attn: int = 128,
        dropout: float = 0.3,
        architecture: str = 'attention'
    ):
        super().__init__()

        if architecture == 'attention':
            self.pool = AttentionMILPool(in_dim, d_hidden_attn, dropout=0.2)
            classifier_in_dim = in_dim

        elif architecture == 'gated':
            self.pool = GatedAttentionMILPool(in_dim, d_hidden_attn, dropout=0.2)
            classifier_in_dim = in_dim

        elif architecture == 'dsmil':
            self.pool = DSMILPool(in_dim, d_hidden=256, dropout=0.2)
            classifier_in_dim = 512

        elif architecture == 'transmil':
            self.pool = TransMILPool(
                in_dim=in_dim,
                embed_dim=512,
                num_heads=8,
                dropout=0.1
            )
            classifier_in_dim = 512

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.classifier = nn.Linear(classifier_in_dim, n_labels)

    def forward(self, feats):
        pooled, attn = self.pool(feats)
        logits = self.classifier(pooled)
        return logits, pooled, attn