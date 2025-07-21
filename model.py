import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 正弦位置编码函数，动态生成
def get_sinusoidal_encoding(seq_len, dim, device):
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-np.log(10000.0) / dim))  # (dim/2,)
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, dim)

class ProteinTransformer(nn.Module):
    def __init__(self, input_dim=3, emb_dim=256, num_heads=16, num_layers=8, num_classes=20, dropout=0.05, max_len=2048):
        super(ProteinTransformer, self).__init__()

        # 坐标嵌入 + LayerNorm
        self.coord_embedding = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_transformer_norm = nn.LayerNorm(emb_dim)

        # MLP 层 + 强化 Dropout
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.global_dropout = nn.Dropout(dropout)

        # 输出层：预测氨基酸
        self.fc = nn.Linear(emb_dim, num_classes)

        # 最大序列长度的动态位置编码
        self.max_len = max_len

    def forward(self, coords, padding_mask=None):
        """
        coords: (B, L, 3) - 输入坐标
        padding_mask: (B, L) - True 表示需要被 mask（即 padding）
        """
        B, L, _ = coords.shape

        # 坐标嵌入
        x = self.coord_embedding(coords)  # (B, L, emb_dim)

        # 动态生成位置编码
        pos_encoding = get_sinusoidal_encoding(L, x.size(-1), x.device)  # (L, emb_dim)
        x = x + pos_encoding.unsqueeze(0)  # 广播到 (B, L, emb_dim)

        # Transformer 编码器
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.post_transformer_norm(x)

        # MLP 层处理
        x = self.mlp(x)
        x = self.global_dropout(x)

        # 输出层
        logits = self.fc(x)  # (B, L, num_classes)

        return logits
