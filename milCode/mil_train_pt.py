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
        #h = self.dropout(h)
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


class TransformerMILPool(nn.Module):
    def __init__(self, in_dim: int, d_hidden: int = 256, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, feats):
        x = self.feat_proj(feats)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return pooled, None


class PatientMILFeatures(nn.Module):
    def __init__(self, in_dim: int = 768, n_labels: int = 1, d_hidden_attn: int = 128, dropout: float = 0.3,
                 architecture: str = 'attention'):
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
            self.pool = TransformerMILPool(in_dim, d_hidden=256, n_heads=4, n_layers=2, dropout=0.2)
            classifier_in_dim = 256
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        '''
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, n_labels)
        )
        '''
        
        self.classifier = nn.Linear(classifier_in_dim, n_labels)

    
    def forward(self, feats):
        pooled, attn = self.pool(feats)
        logits = self.classifier(pooled)
        return logits, pooled, attn




