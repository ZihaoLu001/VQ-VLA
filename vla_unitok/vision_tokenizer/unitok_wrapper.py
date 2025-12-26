import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from PIL import Image
from torchvision.transforms import transforms

# UniTok submodule is exposed via PYTHONPATH=/.../externals/UniTok
from utils.config import Args
from models.unitok import UniTok
from utils.data import normalize_01_into_pm1


@dataclass
class UniTokOutput:
    indices: torch.Tensor                 # LongTensor, (B,K,N) typically
    rec_img: Optional[torch.Tensor] = None  # (B,3,H,W) in [-1,1] if decode=True


class UniTokWrapper:
    """
    Thin wrapper around UniTok tokenizer checkpoint.

    - encode: PIL/paths/tensors -> code indices
    - decode: indices -> reconstructed image
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.ckpt_path = ckpt_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"UniTok checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = Args()
        cfg.load_state_dict(ckpt["args"])

        model = UniTok(cfg)
        model.load_state_dict(ckpt["trainer"]["unitok"])
        model.to(self.device)
        model.eval()

        self.cfg = cfg
        self.model = model

        # follow externals/UniTok/inference.py
        self.preprocess = transforms.Compose([
            transforms.Resize(int(cfg.img_size * cfg.resize_ratio)),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ])

    def _to_tensor(self, x: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, str):
            img = Image.open(x).convert("RGB")
            return self.preprocess(img).unsqueeze(0)  # (1,3,H,W) in [-1,1]
        if isinstance(x, Image.Image):
            return self.preprocess(x.convert("RGB")).unsqueeze(0)
        if torch.is_tensor(x):
            # expect (B,3,H,W) or (3,H,W)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.dim() != 4:
                raise ValueError(f"Tensor input must be (B,3,H,W) or (3,H,W), got {tuple(x.shape)}")
            return x
        raise TypeError(f"Unsupported input type: {type(x)}")

    @torch.no_grad()
    def encode(self, x: Union[str, Image.Image, torch.Tensor], *, to_cpu: bool = False) -> torch.Tensor:
        img = self._to_tensor(x).to(self.device)
        idx = self.model.img_to_idx(img)
        return idx.cpu() if to_cpu else idx

    @torch.no_grad()
    def decode(self, indices: torch.Tensor, *, to_cpu: bool = False) -> torch.Tensor:
        indices = indices.to(self.device)
        img = self.model.idx_to_img(indices)  # [-1,1]
        return img.cpu() if to_cpu else img

    @torch.no_grad()
    def encode_decode(self, x: Union[str, Image.Image, torch.Tensor], *, to_cpu: bool = False) -> UniTokOutput:
        idx = self.encode(x, to_cpu=False)
        rec = self.decode(idx, to_cpu=False)
        if to_cpu:
            return UniTokOutput(indices=idx.cpu(), rec_img=rec.cpu())
        return UniTokOutput(indices=idx, rec_img=rec)
