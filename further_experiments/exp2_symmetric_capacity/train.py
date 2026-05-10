"""Experiment 2 -- symmetric-capacity ablation.

The original architecture (`model1.FCNet`) has asymmetric branches::

    low_freq_mlp  : raw (x,y), 256-wide x 3-layer  = 133,123 params  (~86%)
    high_freq_mlp : PE-only ,  128-wide x 2-layer  =  22,147 params  (~14%)

That asymmetry confounds the original four-mode result: maybe
`continual_high` wins not because PE-input or residual-role weights
lose plasticity, but simply because resetting only 14% of the network
is cheaper than resetting 86%.

This experiment makes BOTH branches the same width and depth, so
resetting either branch reinitializes the same number of parameters.
Input routing is left unchanged from the original (low_freq = raw xy,
high_freq = PE-only), so the only difference vs. the baseline is the
removal of the capacity asymmetry.

Interpretation guide
--------------------
* If the original ``continual_high`` win was driven by CAPACITY:
  with matched capacity, ``continual_high`` and ``continual_low``
  should perform similarly (and similarly to ``scratch``); the gap
  should collapse.

* If the original win was driven by INPUT TYPE or BRANCH ROLE
  (the residual-fitter loses plasticity, regardless of size):
  ``continual_high`` should still beat ``continual_low`` even with
  matched widths and depths.

Defaults match the LARGER original branch (width=256, layers=3) for
both branches, so the model is comfortably over-parameterized for
128x128 image fitting.  Use ``-symmetric_width`` and
``-symmetric_layers`` to sweep other sizes (e.g. ``-symmetric_width
128 -symmetric_layers 2`` matches the SMALLER original branch and is
a useful complementary control).

Usage
-----

    export WANDB_API_KEY=<your_key>
    bash run_all.sh -project cmpt981-plasticity                       # default 256/3
    bash run_all.sh -project cmpt981-plasticity \
        -symmetric_width 128 -symmetric_layers 2                      # match small
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import set_seed
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

from common.viz import save_pair_with_text  # noqa: E402
from common.wandb_utils import init_wandb  # noqa: E402


# --------------------------------------------------------------------- model

class PEOnly(nn.Module):
    """Pure Fourier features (no raw xy concatenated). Output dim = 4*num_res."""

    def __init__(self, num_res: int = 10):
        super().__init__()
        self.num_res = num_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for r in range(self.num_res):
            outs.append(torch.sin(x * (2.0 ** r)))
            outs.append(torch.cos(x * (2.0 ** r)))
        return torch.cat(outs, dim=-1)


def make_mlp(in_dim: int, out_dim: int, width: int, num_layers: int) -> nn.Sequential:
    """ReLU MLP matching the shape of `model1.MLP`.

    `num_layers` counts hidden ReLU stages, so the network has
    `num_layers + 1` Linear modules in total.
    """
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)


class SymmetricFCNet(nn.Module):
    """`model1.FCNet` with both branches forced to the same width and depth.

    Input routing is preserved from the baseline:
        low_freq_mlp  : raw (x, y)            -> 3
        high_freq_mlp : Fourier features only -> 3
    Final RGB output is ``sigmoid(low_logits + high_logits)``, with the
    auxiliary loss ``MSE(low_freq_pred, GT)`` applied in the trainer.
    """

    def __init__(self, num_res: int = 10, width: int = 256, layers: int = 3):
        super().__init__()
        self.num_res = num_res
        self.pe = PEOnly(num_res=num_res)

        self.low_in = 2
        self.high_in = 4 * num_res
        self.width = width
        self.layers = layers

        self.low_freq_mlp = make_mlp(self.low_in, 3, width, layers)
        self.high_freq_mlp = make_mlp(self.high_in, 3, width, layers)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor):
        pe_feats = self.pe(x)
        low_logits = self.low_freq_mlp(x)
        high_logits = self.high_freq_mlp(pe_feats)
        out = torch.sigmoid(low_logits + high_logits)
        return out, torch.sigmoid(low_logits), torch.sigmoid(high_logits)

    def reset_low_freq(self) -> None:
        self.low_freq_mlp = make_mlp(self.low_in, 3,
                                     self.width, self.layers).to(self._device())

    def reset_high_freq(self) -> None:
        self.high_freq_mlp = make_mlp(self.high_in, 3,
                                      self.width, self.layers).to(self._device())


# ------------------------------------------------------------------- trainer

class Trainer:
    """One image, full-batch Adam, fresh optimizer per image (matches train1.py)."""

    def __init__(self,
                 image_path: str,
                 res: int,
                 model: SymmetricFCNet,
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
            for coords, rgb_vals in self.dataloader:
                self.optimizer.zero_grad()
                out, low, _high = self.model(coords)
                loss_full = self.criterion(out, rgb_vals)
                loss_low = self.criterion(low, rgb_vals)
                loss = loss_full + loss_low
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

        def to_uint8_img(t: torch.Tensor) -> np.ndarray:
            return (t.cpu().numpy().reshape(h, w, 3) * 255).astype(np.uint8)

        gt_img = to_uint8_img(gt)
        pred_pair = np.hstack([gt_img, to_uint8_img(pred)])
        low_pair = np.hstack([gt_img, to_uint8_img(low)])
        high_pair = np.hstack([gt_img, to_uint8_img(high)])

        save_pair_with_text(
            pred_pair, f'PSNR: {psnr:.2f}',
            os.path.join(self.out_dir, f'output_{epoch}.png'),
        )
        save_pair_with_text(
            low_pair, f'Low Freq PSNR: {psnr_low:.2f}',
            os.path.join(self.out_dir, f'output_low_freq_{epoch}.png'),
        )
        save_pair_with_text(
            high_pair, f'High Freq PSNR: {psnr_high:.2f}',
            os.path.join(self.out_dir, f'output_high_freq_{epoch}.png'),
        )
        return psnr, psnr_low, psnr_high


# ---------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Exp2: symmetric-capacity ablation '
                    '(same widths/depths in both branches).'
    )
    parser.add_argument('-project', type=str, required=True,
                        help='W&B project name (required).')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-image_size', type=int, default=128)
    parser.add_argument('-image_dir', type=str, default='circles4',
                        help='Path to image folder, relative to repo root if not absolute.')
    parser.add_argument('-num_res', type=int, default=10,
                        help='Fourier feature octaves used by the high branch.')
    parser.add_argument('-symmetric_width', type=int, default=256,
                        help='Width of BOTH branches (default 256, matches '
                             'the original LARGE branch).')
    parser.add_argument('-symmetric_layers', type=int, default=3,
                        help='Hidden layers in BOTH branches (default 3, '
                             'matches the original LARGE branch).')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-nepochs', type=int, default=500)
    parser.add_argument('-viz_every', type=int, default=10)
    parser.add_argument('-training_mode', choices=['scratch', 'continual'],
                        default='scratch')
    parser.add_argument('-reset', choices=['no', 'low', 'high', ''],
                        default='',
                        help="In continual mode: 'no' = keep both branches, "
                             "'low'/'high' = reset that branch between images.")
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

    image_dir = args.image_dir
    if not os.path.isabs(image_dir):
        image_dir = str(PROJECT_ROOT / image_dir)
    image_paths = sorted(os.listdir(image_dir))
    if not image_paths:
        raise RuntimeError(f'No images found in {image_dir}')

    reset_tag = args.reset if args.reset else 'none'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = (
        f'exp2_w{args.symmetric_width}l{args.symmetric_layers}_'
        f'{args.image_size}_{args.seed}_adam_{args.training_mode}_'
        f'{reset_tag}_{timestamp}'
    )
    out_dir_root = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir_root, exist_ok=True)

    # Compute and report per-branch parameter counts as a sanity check.
    probe = SymmetricFCNet(num_res=args.num_res,
                           width=args.symmetric_width,
                           layers=args.symmetric_layers)
    n_low = sum(p.numel() for p in probe.low_freq_mlp.parameters())
    n_high = sum(p.numel() for p in probe.high_freq_mlp.parameters())
    n_total = sum(p.numel() for p in probe.parameters())
    del probe

    config = {
        **vars(args),
        'experiment': 'exp2_symmetric_capacity',
        'arch': f'SymmetricFCNet(width={args.symmetric_width}, '
                f'layers={args.symmetric_layers})',
        'image_dir_resolved': image_dir,
        'num_images': len(image_paths),
        'param_count_low': n_low,
        'param_count_high': n_high,
        'param_count_total': n_total,
        'param_ratio_low_over_high': n_low / max(n_high, 1),
    }
    init_wandb(
        project=args.project,
        run_name=run_name,
        config=config,
        tags=['exp2_symmetric_capacity', args.training_mode, reset_tag,
              f'w{args.symmetric_width}l{args.symmetric_layers}']
             + list(args.tags),
        group=f'exp2_symmetric_capacity_w{args.symmetric_width}l{args.symmetric_layers}',
    )
    wandb.config.update({'out_dir_root': out_dir_root}, allow_val_change=True)

    print(f'Run name : {run_name}')
    print(f'Out dir  : {out_dir_root}')
    print(f'Image dir: {image_dir} ({len(image_paths)} images)')
    print(f'Mode     : {args.training_mode} (reset={reset_tag})')
    print(f'Branches : low={n_low:,} params, high={n_high:,} params, '
          f'total={n_total:,} (ratio {n_low/max(n_high,1):.2f}x)')

    model: SymmetricFCNet | None = None
    outer = tqdm(enumerate(image_paths), total=len(image_paths),
                 desc='Images', position=0)

    for counter, image_name in outer:
        outer.set_postfix_str(image_name)

        if args.training_mode == 'scratch' or model is None:
            model = SymmetricFCNet(
                num_res=args.num_res,
                width=args.symmetric_width,
                layers=args.symmetric_layers,
            ).to(device)

        trainer = Trainer(
            image_path=os.path.join(image_dir, image_name),
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
