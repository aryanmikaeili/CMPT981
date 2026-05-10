"""Experiment 1 -- swap which branch receives the Fourier features.

Original architecture (`model1.FCNet` driven by `train1.py`)::

    LARGE branch  : raw (x, y)            , 256-wide x 3-layer ReLU MLP
                    auxiliary-supervised against full GT.   ('low_freq_mlp')

    SMALL branch  : Fourier features only , 128-wide x 2-layer ReLU MLP
                    only sees gradient through the sum.     ('high_freq_mlp')

In the original four-mode comparison, resetting the SMALL (PE-input)
branch between tasks fully restored plasticity (`continual_high`),
while resetting the LARGE (raw-input) branch did not (`continual_low`).

This experiment KEEPS the per-branch widths and depths identical to the
original (so that capacity asymmetry is unchanged) and only swaps which
branch receives which input::

    LARGE branch  : Fourier features only , 256-wide x 3-layer ReLU MLP
                    auxiliary-supervised against full GT.    ('primary')

    SMALL branch  : raw (x, y)            , 128-wide x 2-layer ReLU MLP
                    only sees gradient through the sum.      ('residual')

Interpretation guide
--------------------
* If LoP localization follows INPUT TYPE (Fourier features cause LoP):
  ``-reset residual`` (reset the small/raw branch) should NOT eliminate
  LoP, while ``-reset primary`` (reset the large/PE branch) should --
  i.e. the OPPOSITE ranking from the original.

* If LoP localization follows BRANCH ROLE (the residual-fitter is what
  loses plasticity) or CAPACITY (the small branch loses plasticity):
  ``-reset residual`` still wins, just like ``continual_high`` did in
  the original setup.

Note on parameter counts
------------------------
Because input dim differs between PE-only (4*num_res) and raw (2), the
per-branch parameter counts differ slightly from the original even
though the widths/depths match.  Both branches still satisfy
``primary_params >> residual_params``, so the qualitative capacity
asymmetry is preserved.

Usage
-----

    export WANDB_API_KEY=<your_key>
    bash run_all.sh -project cmpt981-plasticity   # all 4 modes

    # or one mode at a time:
    python train.py -project cmpt981-plasticity -training_mode scratch
    python train.py -project cmpt981-plasticity -training_mode continual -reset no
    python train.py -project cmpt981-plasticity -training_mode continual -reset primary
    python train.py -project cmpt981-plasticity -training_mode continual -reset residual
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

# Make `from common.* import ...` and project-root modules importable.
for p in (EXP_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import utils  # noqa: E402
from dataset import ImageDataset  # noqa: E402

from common.viz import save_pair_with_text  # noqa: E402
from common.wandb_utils import init_wandb  # noqa: E402


# --------------------------------------------------------------------- model

class PEOnly(nn.Module):
    """Pure Fourier feature encoding.

    Returns ``[sin(x*2^0), cos(x*2^0), ..., sin(x*2^(R-1)), cos(x*2^(R-1))]``,
    which is ``4*num_res``-dimensional for a 2D input.  Crucially, the raw
    ``x`` is NOT concatenated -- this branch must rely on PE alone.
    """

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
    `num_layers + 1` Linear modules total (one input, `num_layers - 1`
    hidden-to-hidden, and one output).
    """
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)


class SwappedFCNet(nn.Module):
    """Inputs-swapped twin of `model1.FCNet`.

    Branch shapes (widths/depths) match the original; only the inputs
    routed into each branch are swapped.
    """

    def __init__(self,
                 num_res: int = 10,
                 primary_width: int = 256,
                 primary_layers: int = 3,
                 residual_width: int = 128,
                 residual_layers: int = 2):
        super().__init__()
        self.num_res = num_res
        self.pe = PEOnly(num_res=num_res)

        self.primary_in = 4 * num_res
        self.primary_width = primary_width
        self.primary_layers = primary_layers
        self.residual_in = 2
        self.residual_width = residual_width
        self.residual_layers = residual_layers

        self.primary_mlp = make_mlp(self.primary_in, 3,
                                    self.primary_width, self.primary_layers)
        self.residual_mlp = make_mlp(self.residual_in, 3,
                                     self.residual_width, self.residual_layers)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor):
        pe = self.pe(x)
        primary_logits = self.primary_mlp(pe)
        residual_logits = self.residual_mlp(x)
        out = torch.sigmoid(primary_logits + residual_logits)
        return out, torch.sigmoid(primary_logits), torch.sigmoid(residual_logits)

    def reset_primary(self) -> None:
        self.primary_mlp = make_mlp(self.primary_in, 3,
                                    self.primary_width,
                                    self.primary_layers).to(self._device())

    def reset_residual(self) -> None:
        self.residual_mlp = make_mlp(self.residual_in, 3,
                                     self.residual_width,
                                     self.residual_layers).to(self._device())


# ------------------------------------------------------------------- trainer

class Trainer:
    """One image, full-batch Adam, fresh optimizer per image (matches train1.py)."""

    def __init__(self,
                 image_path: str,
                 res: int,
                 model: SwappedFCNet,
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
        psnr = psnr_primary = psnr_residual = float('nan')
        for epoch in pbar:
            self.model.train()
            for coords, rgb_vals in self.dataloader:
                self.optimizer.zero_grad()
                out, primary, _residual = self.model(coords)
                loss_full = self.criterion(out, rgb_vals)
                loss_primary = self.criterion(primary, rgb_vals)
                loss = loss_full + loss_primary
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            if (epoch + 1) % self.viz_every == 0 or epoch == 0:
                psnr, psnr_primary, psnr_residual = self._eval_and_save(epoch)
                pbar.set_description(
                    f'epoch {epoch}, PSNR {psnr:.2f} '
                    f'(primary {psnr_primary:.2f}, residual {psnr_residual:.2f})'
                )

        return self.model, psnr, psnr_primary, psnr_residual

    @torch.no_grad()
    def _eval_and_save(self, epoch: int):
        self.model.eval()
        coords = self.dataset.coords
        pred, primary, residual = self.model(coords)
        gt = self.dataset.rgb_vals

        psnr = utils.get_psnr(pred, gt).item()
        psnr_primary = utils.get_psnr(primary, gt).item()
        psnr_residual = utils.get_psnr(residual, gt).item()

        # PIL Image.size is (W, H) but our flat tensors are stored row-major
        # over the full image, so reshape with (H, W, 3).
        w, h = self.dataset.image.size

        def to_uint8_img(t: torch.Tensor) -> np.ndarray:
            return (t.cpu().numpy().reshape(h, w, 3) * 255).astype(np.uint8)

        gt_img = to_uint8_img(gt)
        pred_pair = np.hstack([gt_img, to_uint8_img(pred)])
        primary_pair = np.hstack([gt_img, to_uint8_img(primary)])
        residual_pair = np.hstack([gt_img, to_uint8_img(residual)])

        save_pair_with_text(
            pred_pair, f'PSNR: {psnr:.2f}',
            os.path.join(self.out_dir, f'output_{epoch}.png'),
        )
        save_pair_with_text(
            primary_pair, f'Primary PSNR: {psnr_primary:.2f}',
            os.path.join(self.out_dir, f'output_primary_{epoch}.png'),
        )
        save_pair_with_text(
            residual_pair, f'Residual PSNR: {psnr_residual:.2f}',
            os.path.join(self.out_dir, f'output_residual_{epoch}.png'),
        )
        return psnr, psnr_primary, psnr_residual


# ---------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Exp1: swap PE/raw between primary and residual branches.'
    )
    parser.add_argument('-project', type=str, required=True,
                        help='W&B project name (required).')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-image_size', type=int, default=128)
    parser.add_argument('-image_dir', type=str, default='circles4',
                        help='Path to image folder, relative to repo root if not absolute.')
    parser.add_argument('-num_res', type=int, default=10,
                        help='Fourier feature octaves used by the PE encoding.')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-nepochs', type=int, default=500)
    parser.add_argument('-viz_every', type=int, default=10,
                        help='Eval + save reconstructions every N epochs.')
    parser.add_argument('-training_mode', choices=['scratch', 'continual'],
                        default='scratch')
    parser.add_argument('-reset', choices=['no', 'primary', 'residual', ''],
                        default='',
                        help="In continual mode: 'no' = keep both branches, "
                             "'primary'/'residual' = reset that branch between images.")
    parser.add_argument('-out_root', type=str,
                        default=str(EXP_DIR / 'outputs'),
                        help='Parent directory for image dumps from this experiment.')
    parser.add_argument('-tags', nargs='*', default=[],
                        help='Extra W&B tags for this run.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.training_mode == 'scratch':
        assert args.reset == '', 'reset must be empty for scratch mode'
    else:
        assert args.reset in {'no', 'primary', 'residual'}, (
            "continual mode requires -reset {no, primary, residual}"
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
        f'exp1_{args.image_size}_{args.seed}_adam_{args.training_mode}_'
        f'{reset_tag}_{timestamp}'
    )
    out_dir_root = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir_root, exist_ok=True)

    config = {
        **vars(args),
        'experiment': 'exp1_swap_pe',
        'arch': 'SwappedFCNet (primary=PE-only, residual=raw-xy)',
        'image_dir_resolved': image_dir,
        'num_images': len(image_paths),
    }
    init_wandb(
        project=args.project,
        run_name=run_name,
        config=config,
        tags=['exp1_swap_pe', args.training_mode, reset_tag] + list(args.tags),
        group='exp1_swap_pe',
    )
    wandb.config.update({'out_dir_root': out_dir_root}, allow_val_change=True)

    print(f'Run name : {run_name}')
    print(f'Out dir  : {out_dir_root}')
    print(f'Image dir: {image_dir} ({len(image_paths)} images)')
    print(f'Mode     : {args.training_mode} (reset={reset_tag})')

    model: SwappedFCNet | None = None
    outer = tqdm(enumerate(image_paths), total=len(image_paths),
                 desc='Images', position=0)

    for counter, image_name in outer:
        outer.set_postfix_str(image_name)

        # In scratch mode, OR on the very first iteration of continual mode,
        # build a brand-new model.
        if args.training_mode == 'scratch' or model is None:
            model = SwappedFCNet(num_res=args.num_res).to(device)

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
        model, psnr, psnr_primary, psnr_residual = trainer.run()

        wandb.log({
            'PSNR': psnr,
            'PSNR_primary': psnr_primary,
            'PSNR_residual': psnr_residual,
            'image_index': counter,
        }, step=counter)

        if args.training_mode == 'continual':
            if args.reset == 'primary':
                model.reset_primary()
            elif args.reset == 'residual':
                model.reset_residual()
            # 'no' keeps both branches; nothing to do.

    wandb.finish()


if __name__ == '__main__':
    main()
