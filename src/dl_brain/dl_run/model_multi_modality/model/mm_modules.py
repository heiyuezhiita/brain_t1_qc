# 2023.10.08, modules of multi-modalty model
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn as gnn

# self attention and cross attention ("cross_modal_attention" in ref)
# self modality as Q, and other modality as K and V
# ref: https://github.com/rsinghlab/MADDi/blob/main/training/train_all_modalities.py#L137
# default attention args as fallow:
# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
# torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, 
# add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
class CrossAttentionMM2(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True, **kwargs):
        super(CrossAttentionMM2, self).__init__()
        self.att1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          batch_first=batch_first, **kwargs)
        self.att2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          batch_first=batch_first, **kwargs)
    def forward(self, x, y):
        # Query embeddings of shape (N, L, E) when batch_first=True
        # so, expand dim=1 as L
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        # att
        a1, _ = self.att1(x, y, y)
        a2, _ = self.att1(y, x, x)
        # remove expanded dim
        a1 = a1[:, 0, :]
        a2 = a2[:, 0, :]
        
        return torch.cat((a1, a2), dim=-1)


# ref: https://github.com/rsinghlab/MADDi/blob/main/training/train_all_modalities.py#L137
class SelfAttentionMM(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True, **kwargs):
        super(SelfAttentionMM, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                         batch_first=batch_first, **kwargs)
    def forward(self, x):
        # Query embeddings of shape (N, L, E) when batch_first=True
        # so, expand dim=1 as L
        x = x.unsqueeze(1)
        # att
        a1, _ = self.att(x, x, x)
        # remove expanded dim
        a1 = a1[:, 0, :]
        
        return a1

