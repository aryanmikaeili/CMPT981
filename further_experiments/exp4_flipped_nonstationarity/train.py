"""Experiment 4 -- flipped non-stationarity.

In the baseline ``circles4`` dataset, every image is 10 random circles
on a fixed gray background.  The HIGH-FREQUENCY content (specific
circle placements/colors) varies image-to-image; the LOW-FREQUENCY
content (uniform gray background) is essentially fixed.  Under
continual training, the high-frequency branch absorbs the per-image
variance, and resetting it (`continual_high`) restored plasticity.

This experiment FLIPS that:
* The 10 circles are FIXED across all 50 images (same positions,
  radii, and colors -- shared HIGH-FREQUENCY content).
* The background is a smooth color GRADIENT that varies per image
  (different angle and RGB endpoints -- per-image LOW-FREQUENCY
  content).

Architecture and loss are kept IDENTICAL to the baseline (uses
``model1.FCNet`` and ``loss = MSE(out, GT) + MSE(low_pred, GT)``).
The only thing that changes vs. the baseline is the data distribution.

Predictions
-----------
* If LoP localizes by NON-STATIONARITY (whichever branch absorbs the
  task-specific residual loses plasticity, regardless of frequency
  band):
      Now the LOW branch is the one absorbing per-image variance, so
      ``continual_low`` should beat ``continual_high`` -- the OPPOSITE
      of the baseline ordering on circles4.

* If LoP localizes by FREQUENCY (the high-freq branch loses plasticity
  intrinsically, regardless of what content varies):
      ``continual_high`` should still beat ``continual_low``, just
      like the baseline.

This is the cleanest test of non-stationarity vs. frequency.

Dataset is auto-generated on first run into ``data/circles_fixed_grad_varying/``
with seed ``-data_seed`` (default 42), so the dataset is deterministic
and shareable; pass ``-data_dir`` to override the path or
``-data_seed`` to regenerate a different draw.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

EXP_DIR = Path(__file__).resolve().parent
EXP_ROOT = EXP_DIR.parent           # further_experiments/
PROJECT_ROOT = EXP_DIR.parents[1]   # repository root

for p in (EXP_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import utils  # noqa: E402
from dataset import ImageDataset  # noqa: E402
from model1 import FCNet  # noqa: E402

from common.viz import save_pair_with_text  # noqa: E402
from common.wandb_utils import init_wandb  # noqa: E402


# --------------------------------------------------------- dataset generation

DEFAULT_DATA_DIR = EXP_DIR / 'data' / 'circles_fixed_grad_varying'
GENERATION_RES = 256          # match circles4 generation resolution
NUM_IMAGES = 50
NUM_CIRCLES = 10
DATA_SEED = 42                # determines fixed circles + per-image gradients


def _generate_one_image(rng: np.random.Generator,
                        image_res: int,
                        fixed_circles: list[tuple[int, int, int, tuple[int, int, int]]]
                        ) -> np.ndarray:
    """Smooth color gradient + fixed circles -> uint8 HxWx3 array."""
    angle = rng.uniform(0.0, 2.0 * np.pi)
    c1 = rng.integers(0, 256, size=3).astype(np.float32)
    c2 = rng.integers(0, 256, size=3).astype(np.float32)

    yy, xx = np.meshgrid(
        np.arange(image_res), np.arange(image_res), indexing='ij'
    )
    yy = yy.astype(np.float32) / max(image_res - 1, 1)
    xx = xx.astype(np.float32) / max(image_res - 1, 1)
    t = np.cos(angle) * xx + np.sin(angle) * yy
    span = max(t.max() - t.min(), 1e-9)
    t = (t - t.min()) / span

    bg = (c1[None, None, :] * (1.0 - t[..., None])
          + c2[None, None, :] * t[..., None])
    image = bg.astype(np.uint8).copy()

    for (x, y, r, col) in fixed_circles:
        cv2.circle(image, (int(x), int(y)), int(r),
                   tuple(int(c) for c in col), -1)
    return image


def generate_dataset(data_dir: str,
                     num_images: int = NUM_IMAGES,
                     image_res: int = GENERATION_RES,
                     num_circles: int = NUM_CIRCLES,
                     seed: int = DATA_SEED) -> None:
    """Materialize the exp4 dataset on disk (idempotent per file)."""
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Sample the SHARED circles ONCE (identical for every image).
    fixed_circles: list[tuple[int, int, int, tuple[int, int, int]]] = []
    for _ in range(num_circles):
        x = int(rng.integers(0, image_res))
        y = int(rng.integers(0, image_res))
        r = int(rng.integers(20, 50))
        col = tuple(int(c) for c in rng.integers(0, 256, size=3))
        fixed_circles.append((x, y, r, col))

    # Save the circle spec as a sanity-check sidecar.
    spec_path = os.path.join(data_dir, 'fixed_circles.txt')
    if not os.path.isfile(spec_path):
        with open(spec_path, 'w') as f:
            f.write(f'# data_seed={seed}, image_res={image_res}, '
                    f'num_circles={num_circles}\n')
            f.write('# x y radius r g b\n')
            for (x, y, r, (cr, cg, cb)) in fixed_circles:
                f.write(f'{x} {y} {r} {cr} {cg} {cb}\n')

    for i in range(num_images):
        out_path = os.path.join(data_dir, f'flipped_{i:03d}.png')
        if os.path.isfile(out_path):
            continue
        img = _generate_one_image(rng, image_res, fixed_circles)
        Image.fromarray(img).save(out_path)


def ensure_dataset(data_dir: str,
                   num_images: int = NUM_IMAGES,
                   image_res: int = GENERATION_RES,
                   num_circles: int = NUM_CIRCLES,
                   seed: int = DATA_SEED) -> None:
    needed = {f'flipped_{i:03d}.png' for i in range(num_images)}
    have = set(os.listdir(data_dir)) if os.path.isdir(data_dir) else set()
    if needed.issubset(have):
        return
    print(f'Generating exp4 dataset in {data_dir} '
          f'({num_images} images @ {image_res}x{image_res}, seed={seed})...')
    generate_dataset(data_dir, num_images, image_res, num_circles, seed)


# --------------------------------------------------------------- trainer

class Trainer:
    """Mirrors baseline ``train1.Trainer`` -- only the dataset has changed."""

    def __init__(self,
                 image_path: str,
                 res: int,
                 model: FCNet,
                 device: torch.device,
                 lr: float,
                 nepochs: int,
                 out_dir: str,
                 viz_every: int):
        self.dataset = ImageDataset(image_path, res, device)
        self.dataloader = DataLoader(self.dataset, batch_size=res * res,
                                     shuffle=True)
        self.res = res
        self.nepochs = nepochs
        self.viz_every = viz_every
        self.model = model

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.5
        )
        self.criterion = nn.MSELoss()

        stem = os.path.splitext(os.path.basename(image_path))[0]
        self.out_dir = os.path.join(out_dir, stem)
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        pbar = tqdm(range(self.nepochs), desc='Epochs',
                    leave=False, position=1)
        psnr = psnr_low = psnr_high = float('nan')
        for epoch in pbar:
            self.model.train()
            for coords, rgb in self.dataloader:
                self.optimizer.zero_grad()
                out, low, _high = self.model(coords)
                loss = self.criterion(out, rgb) + self.criterion(low, rgb)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            if (epoch + 1) % self.viz_every == 0 or epoch == 0:
                psnr, psnr_low, psnr_high = self._eval_and_save(epoch)
                pbar.set_description(
                    f'epoch {epoch}, PSNR {psnr:.2f} '
                    f'(low {psnr_low:.2f}, high {psnr_high:.2f})'
                )
        return self.model, psnr, psnr_low, psnr_high

    @torch.no_grad()
    def _eval_and_save(self, epoch: int):
        self.model.eval()
        coords = self.dataset.coords
        pred, low, high = self.model(coords)
        gt = self.dataset.rgb_vals
        psnr = utils.get_psnr(pred, gt).item()
        psnr_low = utils.get_psnr(low, gt).item()
        psnr_high = utils.get_psnr(high, gt).item()

        w, h = self.dataset.image.size

        def to_uint8(t: torch.Tensor) -> np.ndarray:
            arr = t.cpu().numpy().reshape(h, w, 3)
            return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

        gt_img = to_uint8(gt)
        save_pair_with_text(
            np.hstack([gt_img, to_uint8(pred)]),
            f'PSNR: {psnr:.2f}',
            os.path.join(self.out_dir, f'output_{epoch}.png'),
        )
        save_pair_with_text(
            np.hstack([gt_img, to_uint8(low)]),
            f'Low Freq PSNR: {psnr_low:.2f}',
            os.path.join(self.out_dir, f'output_low_freq_{epoch}.png'),
        )
        save_pair_with_text(
            np.hstack([gt_img, to_uint8(high)]),
            f'High Freq PSNR: {psnr_high:.2f}',
            os.path.join(self.out_dir, f'output_high_freq_{epoch}.png'),
        )
        return psnr, psnr_low, psnr_high


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Exp4: flipped non-stationarity '
                    '(circles fixed, smooth gradient varies per image).'
    )
    parser.add_argument('-project', type=str, required=True,
                        help='W&B project name (required).')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-image_size', type=int, default=128)
    parser.add_argument('-data_dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='Where to find/generate the flipped-stationarity dataset.')
    parser.add_argument('-data_seed', type=int, default=DATA_SEED,
                        help='Seed for the on-disk dataset (fixes both the '
                             'shared circles and per-image gradients).')
    parser.add_argument('-num_res', type=int, default=10)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-nepochs', type=int, default=500)
    parser.add_argument('-viz_every', type=int, default=10)
    parser.add_argument('-training_mode', choices=['scratch', 'continual'],
                        default='scratch')
    parser.add_argument('-reset', choices=['no', 'low', 'high', ''],
                        default='')
    parser.add_argument('-out_root', type=str,
                        default=str(EXP_DIR / 'outputs'))
    parser.add_argument('-tags', nargs='*', default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.training_mode == 'scratch':
        assert args.reset == '', 'reset must be empty for scratch mode'
    else:
        assert args.reset in {'no', 'low', 'high'}, (
            "continual mode requires -reset {no, low, high}"
        )

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ensure_dataset(args.data_dir, seed=args.data_seed)
    image_paths = sorted(p for p in os.listdir(args.data_dir)
                         if p.lower().endswith(('.png', '.jpg', '.jpeg')))
    if not image_paths:
        raise RuntimeError(f'No images in {args.data_dir}')

    reset_tag = args.reset if args.reset else 'none'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = (
        f'exp4_{args.image_size}_{args.seed}_adam_{args.training_mode}_'
        f'{reset_tag}_{timestamp}'
    )
    out_dir_root = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir_root, exist_ok=True)

    config = {
        **vars(args),
        'experiment': 'exp4_flipped_nonstationarity',
        'arch': 'model1.FCNet (baseline)',
        'num_images': len(image_paths),
    }
    init_wandb(
        project=args.project,
        run_name=run_name,
        config=config,
        tags=['exp4_flipped_nonstationarity', args.training_mode, reset_tag]
             + list(args.tags),
        group='exp4_flipped_nonstationarity',
    )
    wandb.config.update({'out_dir_root': out_dir_root}, allow_val_change=True)

    print(f'Run name : {run_name}')
    print(f'Out dir  : {out_dir_root}')
    print(f'Data dir : {args.data_dir} ({len(image_paths)} images)')
    print(f'Mode     : {args.training_mode} (reset={reset_tag})')

    model: FCNet | None = None
    outer = tqdm(enumerate(image_paths), total=len(image_paths),
                 desc='Images', position=0)

    for counter, name in outer:
        outer.set_postfix_str(name)

        if args.training_mode == 'scratch' or model is None:
            model = FCNet(use_pe=True, num_res=args.num_res,
                          num_layers=2, width=256).to(device)

        trainer = Trainer(
            image_path=os.path.join(args.data_dir, name),
            res=args.image_size,
            model=model,
            device=device,
            lr=args.lr,
            nepochs=args.nepochs,
            out_dir=out_dir_root,
            viz_every=args.viz_every,
        )
        model, psnr, psnr_low, psnr_high = trainer.run()

        wandb.log({
            'PSNR': psnr,
            'PSNR_low': psnr_low,
            'PSNR_high': psnr_high,
            'image_index': counter,
        }, step=counter)

        if args.training_mode == 'continual':
            if args.reset == 'low':
                model.reset_low_freq()
            elif args.reset == 'high':
                model.reset_high_freq()
            # 'no' keeps both branches; nothing to do.

    wandb.finish()


if __name__ == '__main__':
    main()
