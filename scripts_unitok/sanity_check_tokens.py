import argparse
from pathlib import Path
import torch
from PIL import Image

from vla_unitok.vision_tokenizer.unitok_wrapper import UniTokWrapper
from vla_unitok.vision_tokenizer.token_packing import pack_indices


def save_rec(rec: torch.Tensor, out_path: str):
    # rec: (B,3,H,W) in [-1,1]
    rec = rec[0].detach().cpu()
    rec = (rec + 1.0) * 0.5
    rec = (rec.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(rec).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--src_img", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="/datasets/v2p/current/lucas/tmp/unitok",
                    help="save outputs under this directory (recommended: your own path)")
    ap.add_argument("--prefix", type=str, default="sample", help="output file prefix")

    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--save_idx", action="store_true", help="save raw indices tensor")
    ap.add_argument("--save_seq", action="store_true", help="save packed sequence tensor (B,L)")

    # packing mode
    ap.add_argument("--pack_mode", type=str, default="auto", choices=["auto", "flat", "offset"],
                    help="auto: offset-pack for 3D indices when vocab_size available; else flat")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = UniTokWrapper(args.ckpt_path, device=None)

    idx = tok.encode(args.src_img)  # likely (1,K,N)
    print("indices dtype:", idx.dtype, "shape:", tuple(idx.shape), "min/max:", idx.min().item(), idx.max().item())

    # save idx
    if args.save_idx:
        idx_path = out_dir / f"{args.prefix}_idx.pt"
        torch.save({"indices": idx.cpu(), "shape": tuple(idx.shape)}, idx_path)
        print("saved idx to:", str(idx_path))

    # save packed seq
    if args.save_seq:
        packed, meta = pack_indices(idx, vocab_size=getattr(tok.cfg, "vocab_size", None), mode=args.pack_mode)
        seq_path = out_dir / f"{args.prefix}_seq.pt"
        torch.save({"unitok_seq": packed.cpu(), "meta": meta}, seq_path)
        print("saved seq to:", str(seq_path), "shape:", tuple(packed.shape),
              "min/max:", int(packed.min()), int(packed.max()))

    # decode / reconstruct image
    if args.decode:
        rec = tok.decode(idx)
        rec_path = out_dir / f"{args.prefix}_rec.png"
        save_rec(rec, str(rec_path))
        print("saved reconstructed image to:", str(rec_path))


if __name__ == "__main__":
    main()
