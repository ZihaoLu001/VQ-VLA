import argparse
from pathlib import Path
import torch

from vla_unitok.vision_tokenizer.unitok_wrapper import UniTokWrapper
from vla_unitok.vision_tokenizer.token_packing import pack_indices

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_rel_stem(img_root: Path, p: Path) -> str:
    """
    Encode relative path into a filename-safe stem to avoid overwriting:
      a/b/frame_0001.png -> a__b__frame_0001.png
    """
    rel = p.relative_to(img_root).as_posix()
    rel = rel.replace("/", "__")
    # remove extension
    if rel.lower().endswith(p.suffix.lower()):
        rel = rel[: -len(p.suffix)]
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True, help="root directory that contains images")
    ap.add_argument("--out_dir", type=str, required=True, help="directory to save .pt token files")

    ap.add_argument("--pack", action="store_true", help="save packed seq (B,L) instead of raw indices")
    ap.add_argument("--pack_mode", type=str, default="auto", choices=["auto", "flat", "offset"],
                    help="auto: offset-pack for 3D indices when vocab_size available; else flat")

    ap.add_argument("--limit", type=int, default=-1, help="optional limit number of images for quick test")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = UniTokWrapper(args.ckpt_path, device=None)

    imgs = sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if args.limit > 0:
        imgs = imgs[: args.limit]
    print("found images:", len(imgs))

    for p in imgs:
        idx = tok.encode(str(p))  # (1, ...)
        if args.pack:
            packed, meta_pack = pack_indices(idx, vocab_size=getattr(tok.cfg, "vocab_size", None), mode=args.pack_mode)
            to_save = packed.cpu()
            meta = {
                "packed": True,
                "pack_meta": meta_pack,
                "orig_shape": tuple(idx.shape),
            }
        else:
            to_save = idx.cpu()
            meta = {
                "packed": False,
                "orig_shape": tuple(idx.shape),
            }

        stem = safe_rel_stem(img_dir, p)
        out_path = out_dir / (stem + ".pt")
        torch.save({"indices": to_save, "meta": meta, "src": str(p)}, out_path)

    print("done. saved to:", str(out_dir))


if __name__ == "__main__":
    main()
