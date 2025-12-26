import torch
from typing import Tuple, Dict, Any, Optional

def pack_indices_flat(indices: torch.Tensor) -> torch.Tensor:
    """
    Simple flatten pack:
    - (B, N) -> (B, N)
    - (B, N, K) / (B, K, N) -> (B, N*K)
    - otherwise: flatten all dims after batch.
    """
    if indices.dim() < 2:
        raise ValueError(f"indices should have batch dim, got shape={tuple(indices.shape)}")
    return indices.reshape(indices.shape[0], -1)


def pack_indices_with_offset(
    indices: torch.Tensor,
    vocab_size: int,
    *,
    codebook_dim: int = 1,
) -> torch.Tensor:
    """
    Offset-pack UniTok indices with multiple codebooks into a single 1D sequence.

    UniTok default often returns (B, K, N) = (batch, num_codebooks, num_tokens).
    We convert it to (B, N*K) by:
        seq[n, k] = indices[k, n] + k * vocab_size

    Args:
        indices: LongTensor, shape (B, K, N) or (B, N, K)
        vocab_size: cfg.vocab_size
        codebook_dim:
            - if indices is (B, K, N): codebook_dim=1
            - if indices is (B, N, K): codebook_dim=2
    Returns:
        packed: LongTensor, shape (B, N*K)
    """
    if indices.dim() != 3:
        raise ValueError(f"offset-pack expects 3D tensor (B,K,N) or (B,N,K), got {tuple(indices.shape)}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if codebook_dim not in (1, 2):
        raise ValueError(f"codebook_dim must be 1 or 2, got {codebook_dim}")

    idx = indices.long()
    if codebook_dim == 2:
        # (B, N, K) -> (B, K, N)
        idx = idx.permute(0, 2, 1).contiguous()

    B, K, N = idx.shape
    offset = torch.arange(K, device=idx.device).view(1, K, 1) * int(vocab_size)
    packed = (idx + offset).permute(0, 2, 1).reshape(B, N * K)
    return packed


def unpack_indices_with_offset(
    packed: torch.Tensor,
    *,
    vocab_size: int,
    num_codebooks: int,
    num_tokens: int,
    out_layout: str = "BKN",
) -> torch.Tensor:
    """
    Inverse of pack_indices_with_offset.

    Args:
        packed: (B, N*K) long
        vocab_size: cfg.vocab_size
        num_codebooks: K
        num_tokens: N
        out_layout: "BKN" -> return (B,K,N), "BNK" -> return (B,N,K)
    """
    if packed.dim() != 2:
        raise ValueError(f"packed must be 2D (B, L), got {tuple(packed.shape)}")
    if out_layout not in ("BKN", "BNK"):
        raise ValueError(f"out_layout must be 'BKN' or 'BNK', got {out_layout}")

    B, L = packed.shape
    K = int(num_codebooks)
    N = int(num_tokens)
    if L != N * K:
        raise ValueError(f"packed length mismatch: L={L}, expected N*K={N*K}")

    x = packed.view(B, N, K)
    # de-offset and recover per-codebook indices
    offset = torch.arange(K, device=packed.device).view(1, 1, K) * int(vocab_size)
    idx = (x - offset).clamp_min(0).long()  # (B,N,K)
    if out_layout == "BKN":
        return idx.permute(0, 2, 1).contiguous()
    return idx


def pack_indices(
    indices: torch.Tensor,
    *,
    vocab_size: Optional[int] = None,
    mode: str = "auto",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Unified packing entry.

    mode:
      - "auto": if indices is (B,K,N) or (B,N,K) and vocab_size is provided -> use offset-pack.
                else -> flat pack.
      - "flat": always flatten
      - "offset": always offset-pack (requires vocab_size and 3D indices)

    Returns:
      packed, meta
    """
    meta: Dict[str, Any] = {
        "orig_shape": tuple(indices.shape),
        "mode": None,
    }

    if mode not in ("auto", "flat", "offset"):
        raise ValueError(f"mode must be one of auto/flat/offset, got {mode}")

    if mode == "flat":
        meta["mode"] = "flat"
        return pack_indices_flat(indices), meta

    if mode == "offset":
        if vocab_size is None:
            raise ValueError("vocab_size is required for offset mode")
        meta["mode"] = "offset"
        meta["vocab_size"] = int(vocab_size)
        meta["layout"] = "BKN"  # we always output packed in N*K order
        return pack_indices_with_offset(indices, int(vocab_size), codebook_dim=1), meta

    # auto
    if indices.dim() == 3 and vocab_size is not None:
        # we assume UniTok returns (B,K,N) in your case
        meta["mode"] = "offset"
        meta["vocab_size"] = int(vocab_size)
        meta["layout"] = "BKN"
        return pack_indices_with_offset(indices, int(vocab_size), codebook_dim=1), meta

    meta["mode"] = "flat"
    return pack_indices_flat(indices), meta


def unpack_indices(packed: torch.Tensor, shape_after_batch: Tuple[int, ...]) -> torch.Tensor:
    """
    Backward-compatible inverse of pack_indices_flat, reshape (B, L) back to (B, *shape_after_batch).
    """
    return packed.reshape(packed.shape[0], *shape_after_batch)
