"""Pure-PyTorch stand-in for NATTEN's legacy neighborhood-attention ops (natten<0.15 API), so
All-In-One's DiNAT model runs on Windows without the CUDA extension. Semantics follow NATTEN:
each query attends to a K-wide window (per axis) in its dilation group, shifted inward at the
edges; the relative-position bias is indexed by the true offset (+K-1)."""
import torch


def _nbr(L: int, K: int, d: int, device):
    i = torch.arange(L, device=device)
    g, ig = i % d, i // d
    Lg = (L - g + d - 1) // d
    start = torch.clamp(ig - K // 2, min=0)
    start = torch.minimum(start, torch.clamp(Lg - K, min=0))
    k = torch.arange(K, device=device)
    pos = start[:, None] + k[None, :]                     # group coordinates
    idx = torch.clamp(pos * d + g[:, None], max=L - 1)
    bias = torch.clamp(pos - ig[:, None] + K - 1, 0, 2 * K - 2)
    return idx, bias


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    L = query.shape[-2]
    idx, bias = _nbr(L, kernel_size, dilation, query.device)
    kn = key[:, :, idx]                                    # B,H,L,K,D
    attn = (query.unsqueeze(-2) * kn).sum(-1)
    if rpb is not None:
        attn = attn + rpb[:, bias].unsqueeze(0)            # H,L,K
    return attn


def natten1dav(attn, value, kernel_size, dilation):
    L = value.shape[-2]
    idx, _ = _nbr(L, kernel_size, dilation, value.device)
    vn = value[:, :, idx]                                  # B,H,L,K,D
    return (attn.unsqueeze(-1) * vn).sum(-2)


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    X, Y = query.shape[-3], query.shape[-2]
    K = kernel_size
    ix, bx = _nbr(X, K, dilation, query.device)
    iy, by = _nbr(Y, K, dilation, query.device)
    kn = key[:, :, ix[:, None, :, None], iy[None, :, None, :]]     # B,H,X,Y,K,K,D
    attn = (query[:, :, :, :, None, None, :] * kn).sum(-1)
    if rpb is not None:
        b = rpb[:, bx[:, None, :, None], by[None, :, None, :]]     # H,X,Y,K,K
        attn = attn + b.unsqueeze(0)
    return attn.reshape(*attn.shape[:4], K * K)


def natten2dav(attn, value, kernel_size, dilation):
    X, Y = value.shape[-3], value.shape[-2]
    K = kernel_size
    ix, _ = _nbr(X, K, dilation, value.device)
    iy, _ = _nbr(Y, K, dilation, value.device)
    vn = value[:, :, ix[:, None, :, None], iy[None, :, None, :]]   # B,H,X,Y,K,K,D
    a = attn.reshape(*attn.shape[:4], K, K, 1)
    return (a * vn).sum((-3, -2))
