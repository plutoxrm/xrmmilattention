# mil_train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionMILPool(nn.Module):
    """
    简单的 Attention-MIL 聚合层：
    输入: feats [B, N, D]
    输出:
      pooled [B, D]  -- 加权汇总后的患者级特征
      attn   [B, N]  -- 每一帧的注意力权重
    """
    def __init__(self, in_dim: int, d_hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)

    def forward(self, feats: torch.Tensor):
        # feats: [B, N, D]
        h = torch.tanh(self.fc1(feats))      # [B, N, d_hidden]
        logits = self.fc2(h).squeeze(-1)     # [B, N]
        attn = torch.softmax(logits, dim=1)  # [B, N]
        pooled = torch.bmm(attn.unsqueeze(1), feats).squeeze(1)  # [B, D]
        return pooled, attn


class EncoderBackbone(nn.Module):
    """
    编码器：EfficientNet / 其他 timm 模型
    支持离线加载预训练权重；你下载的 efficientnet_b0_rwightman-*.pth 就在这里加载。
    """
    def __init__(self,
                 name: str = "efficientnet_b0",
                 pretrained: bool = False,
                 weights_path: str | None = None,
                 num_classes: int = 0,
                 global_pool: str = "avg"):
        super().__init__()
        # 如果是 DINOv2 / ViT，默认用 token pooling
        if ("dinov2" in name or name.startswith("vit_")) and global_pool == "avg":
            global_pool = "token"
        print(f"Encoder: {name}, global_pool={global_pool}")
        # 不从网上下权重，pretrained=False
        self.net = timm.create_model(
            name,
            pretrained=False,
            num_classes=num_classes,
            global_pool=global_pool
        )
        # 输出维度 D
        self.out_dim = getattr(self.net, "num_features", None)
        if self.out_dim is None:
            # 部分 timm 模型用这个接口
            self.out_dim = self.net.get_classifier().in_features

        # 如果你传了本地权重，就手动加载
        if pretrained and weights_path is not None and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            print(f"✅ Loaded offline pretrained weights: {weights_path}")
            if missing:
                print(f"  (info) missing keys: {len(missing)}")
            if unexpected:
                print(f"  (info) unexpected keys: {len(unexpected)}")
        else:
            if pretrained and weights_path is not None:
                print(f"⚠️ weights_path 不存在：{weights_path}，将使用随机初始化编码器。")
            else:
                print("ℹ️ 未使用预训练（随机初始化编码器）。")

    def forward(self, x: torch.Tensor):
        return self.net(x)


class PatientMILMultiLabel(nn.Module):
    """
    患者级多标签 MIL 模型：
    - EncoderBackbone: 图像特征提取（你下载的EfficientNet权重在这里用）
    - AttentionMILPool: 对 N 张图进行注意力加权
    - cls: 多标签线性分类头
    """
    def __init__(self,
                 n_labels: int,
                 encoder_name: str = "efficientnet_b0",
                 d_hidden_attn: int = 512,
                 dropout: float = 0.2,
                 pretrained: bool = False,
                 weights_path: str | None = None,
                 freeze_encoder: bool = True):
        super().__init__()
        self.encoder = EncoderBackbone(
            name=encoder_name,
            pretrained=pretrained,
            weights_path=weights_path,
            num_classes=0,
            global_pool="avg"
        )
        D = self.encoder.out_dim

        # Attention MIL 聚合
        self.pool = AttentionMILPool(D, d_hidden_attn)

        hidden1 = 1024
        hidden2 = 512
        hidden3 = 256
        self.cls = nn.Sequential(
            nn.Linear(D, hidden1),
            nn.ReLU(inplace=True),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),

            nn.Linear(hidden3, n_labels)
        )


        # ✅ 冻结 encoder 参数：不参加训练，不更新你离线下载的权重
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("✅ Encoder 参数已冻结（只训练 Attention + 分类头）。")
        else:
            print("ℹ️ Encoder 参数未冻结，将一起训练。")

    def forward(self, bags: torch.Tensor):
        """
        bags: [B, N, 3, H, W]
        返回:
          logits: [B, L]   多标签 logit
          pooled: [B, D]   患者级特征
          attn:   [B, N]   帧级注意力
        """
        B, N, C, H, W = bags.shape
        # print(f"模型预定义，正在打印bags的形状: {bags.shape}") #torch.Size([1, 39, 3, 518, 518])
        x = bags.view(B * N, C, H, W)          # [B*N, 3, H, W]
        # print(f"正在打印x的形状: {x.shape}") #torch.Size([39, 3, 518, 518])
        feats = self.encoder(x)                # [B*N, D]
        # print(f"正在打印feats的形状: {feats.shape}") #torch.Size([39, 768])
        D = feats.shape[-1]
        # print(f"正在打印D的形状: {D}") #768
        feats = feats.view(B, N, D)            # [B, N, D]
        # print(f"正在打印feats的形状: {feats.shape}") # torch.Size([1, 39, 768])
        pooled, attn = self.pool(feats)        # [B, D], [B, N]
        # print(f"正在打印pooled的形状: {pooled.shape}") #torch.Size([1, 768])
        # print(f"正在打印attn的形状: {attn.shape}") #torch.Size([1, 39])
        logits = self.cls(pooled)              # [B, L]
        # print(f"正在打印logits的形状: {logits.shape}") #torch.Size([1, 4])
        return logits, pooled, attn

        
