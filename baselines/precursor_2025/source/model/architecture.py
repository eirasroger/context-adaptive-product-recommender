import torch
import torch.nn as nn

class SetTransformerBlock(nn.Module):
    """self-attention block with pre-norm, residuals, and FFN."""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        # x: (B, S, D)
        # Self-attention with pre-norm
        x2 = self.norm1(x)
        attn_mask = ~mask          # True where padding
        attn_out, _ = self.attn(x2, x2, x2,
                                key_padding_mask=attn_mask)
        x = x + attn_out           # residual

        # Feed-forward with pre-norm
        x2 = self.norm2(x)
        x = x + self.ffn(x2)       # residual
        return x

class SetRanker(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 scenario_dim: int, 
                 hidden_dims: list[int],
                 dropout: float,
                 num_heads: int = 2,
                 num_attn_blocks: int = 2,
                 ffn_factor: float = 4.0):
        """
        feat_dim: Total input dimension (Item features + Scenario vector)
        scenario_dim: (Unused in Early Fusion, kept for compatibility with the call args)
        hidden_dims: e.g. [512, 256, 128]
        """
        super().__init__()
        assert isinstance(hidden_dims, list) and len(hidden_dims) >= 1
        
        # We do NOT split the input. We feed (Item + Scenario) together.
        # The input dimension is the full feat_dim passed from main.py
        
        layers_feat = []
        in_dim = feat_dim  # Takes the full concatenated vector [Features, Scenario]
        
        target_dim = hidden_dims[-1]
        
        for h in hidden_dims[:-1]:
            layers_feat += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        
        # Final projection to embedding size
        layers_feat += [nn.Linear(in_dim, target_dim)]
        

        self.input_encoder = nn.Sequential(*layers_feat)

        # --- Transformer Blocks ---
        final_dim = target_dim
        ffn_dim = int(final_dim * ffn_factor)
        
        self.attn_blocks = nn.ModuleList([
            SetTransformerBlock(embed_dim=final_dim,
                                num_heads=num_heads,
                                ffn_dim=ffn_dim,
                                dropout=dropout)
            for _ in range(num_attn_blocks)
        ])

        # --- Scoring Head ---
        self.scorer = nn.Sequential(
            nn.Linear(2 * final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, 1)
        )

    def forward(self, X, mask):
        """
        X: (B, S, F) - Full input vector containing features AND scenario one-hot
        """
        B, S, F = X.shape
        
        # Flatten to (B*S, F) for the MLP
        flat_X = X.reshape(B * S, -1)
        
        # The MLP now sees Density AND Scenario flags simultaneously.
        # It can learn: e.g. if (Scenario_Thermal is 1) -> flip sign of Density weight.
        h = self.input_encoder(flat_X)
        h = h.view(B, S, -1) # (B, S, final_dim)

        # --- Apply Transformer Blocks ---
        for block in self.attn_blocks:
            h = block(h, mask)

        # --- Zero out padded positions ---
        h = h * mask.unsqueeze(-1)

        # --- Global Context Aggregation (Set Context) ---
        denom = mask.sum(1, keepdim=True).clamp(min=1).to(h.dtype)
        context = h.sum(1) / denom
        context = context.unsqueeze(1).expand(-1, S, -1) 

        # --- Scoring ---
        cat = torch.cat([h, context], dim=-1)
        scores = self.scorer(cat).squeeze(-1)
        
        scores = torch.sigmoid(scores)
        scores = scores * mask.float()
        
        return scores
