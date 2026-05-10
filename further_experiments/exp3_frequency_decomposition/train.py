"""Experiment 3 -- frequency-decomposed targets.

The original auxiliary loss
    loss_low = MSE(low_freq_pred, GT)
forces the low branch to fit the FULL ground-truth image, so 'low_freq_mlp'
ends up reconstructing image-specific structure (blurry circles in the
right places) -- not a pure low-frequency component.  The split between
the two branches is therefore better described as 'primary reconstructor'
vs. 'residual sharpener', not as 'low frequency' vs. 'high frequency'.

This experiment forces a LITERAL frequency decomposition by changing
what each branch is supervised against::

    GT_low  := gaussian_blur(GT, sigma)         (pure low-pass)
    GT_high := GT - GT_low                      (pure high-pass residual)

    loss = MSE(combined, GT)                    (full reconstruction)
         + MSE(low_pred,  GT_low)               (low branch -> low band)
         + MSE(high_pred, GT_high)              (high branch -> high band)

Architectural side-effect
-------------------------
Because GT_high is signed (in [-1, 1]) we drop the output sigmoids on
both branches and on the combined output.  The combined output is just
``low_pred + high_pred`` in value space; the loss keeps it close to
[0, 1] because GT lives there.  Per-branch widths and depths match
the original baseline (low: 256x3, high: 128x2) so capacity asymmetry
is preserved and the comparison to the baseline four-mode result is as
direct as possible given the loss change.

Interpretation guide
--------------------
* If the original ``continual_high`` win was about FREQUENCY
  (high-frequency-fitting weights are what lose plasticity):
      ``continual_high`` should still win cleanly here -- because
      the high branch is now LITERALLY the high-frequency fitter.

* If the original win was about RESIDUAL-FITTING (whichever branch
  absorbs the task-specific signal loses plasticity, regardless of
  what frequency band that signal is in):
      The win may now move to whichever branch fits the more
      task-variable component.  In this dataset the high-frequency
      residual still varies a lot per image, so I expect
      ``continual_high`` to keep winning -- but exp4 (flipped
      non-stationarity) is the cleaner test of the residual hypothesis.

Usage
-----

    export WANDB_API_KEY=<your_key>
    bash run_all.sh -project cmpt981-plasticity                # default sigma=4
    bash run_all.sh -project cmpt981-plasticity -blur_sigma 8  # sweep blur kernel
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
import torchvision.transforms.functional as TF
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

EXP_DIR = Path(__file__).resolve().parent
EXP_ROOT = EXP_DIR.parent           # further_experiments/
PROJECT_ROOT = EXP_DIR.parents[1]   # repository root

for p in (EXP_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import utils  # noqa: E402

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
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)


class FreqDecomposedFCNet(nn.Module):
    """Two-branch INR with NO output sigmoids.

    Both branches output raw values; the combined image prediction is
    their sum in value space::

        low_pred  ~ blur(GT)        (target in [0, 1])
        high_pred ~ GT - blur(GT)   (target in [-1, 1])
        combined  =  low_pred + high_pred  ~ GT  (target in [0, 1])

    Per-branch shapes match the original ``model1.FCNet`` (low: 256x3,
    high: 128x2) so capacity asymmetry is unchanged from the baseline.
    """

    def __init__(self,
                 num_res: int = 10,
                 low_width: int = 256, low_layers: int = 3,
                 high_width: int = 128, high_layers: int = 2):
        super().__init__()
        self.num_res = num_res
        self.pe = PEOnly(num_res=num_res)

        self.low_in = 2
        self.high_in = 4 * num_res
        self.low_width = low_width
        self.low_layers = low_layers
        self.high_width = high_width
        self.high_layers = high_layers

        self.low_freq_mlp = make_mlp(self.low_in, 3, low_width, low_layers)
        self.high_freq_mlp = make_mlp(self.high_in, 3, high_width, high_layers)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor):
        pe_feats = self.pe(x)
        low_pred = self.low_freq_mlp(x)
        high_pred = self.high_freq_mlp(pe_feats)
        combined = low_pred + high_pred
        return combined, low_pred, high_pred

    def reset_low_freq(self) -> None:
        self.low_freq_mlp = make_mlp(self.low_in, 3,
                                     self.low_width,
                                     self.low_layers).to(self._device())

    def reset_high_freq(self) -> None:
        self.high_freq_mlp = make_mlp(self.high_in, 3,
                                      self.high_width,
                                      self.high_layers).to(self._device())


# ------------------------------------------------------------------- dataset

class FreqDecomposedImageDataset(Dataset):
    """ImageDataset variant that also exposes blurred-GT and residual-GT.

    Stores ``rgb_vals`` (full GT), ``gt_low`` (blurred), ``gt_high``
    (= rgb_vals - gt_low), and ``coords`` -- all flattened to
    ``(H*W, *)``.  Mirrors the original ``dataset.ImageDataset`` for
    the parts that overlap.
    """

    def __init__(self, image_path: str, res: int, blur_sigma: float,
                 device: torch.device | str = 'cuda'):
        self.image = Image.open(image_path).convert('RGB')
        self.image = utils.crop_and_resize(self.image, res)

        gt_chw = TF.to_tensor(self.image)  # (3, H, W) in [0, 1]
        kernel_size = max(3, 2 * int(round(3 * blur_sigma)) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        gt_low_chw = TF.gaussian_blur(
            gt_chw.unsqueeze(0),
            kernel_size=[kernel_size, kernel_size],
            sigma=[float(blur_sigma), float(blur_sigma)],
        ).squeeze(0)
        gt_high_chw = gt_chw - gt_low_chw

        self.rgb_vals = gt_chw.reshape(3, -1).T.to(device)
        self.gt_low = gt_low_chw.reshape(3, -1).T.to(device)
        self.gt_high = gt_high_chw.reshape(3, -1).T.to(device)
        self.coords = utils.get_coords(res, normalize=True).to(device).reshape(-1, 2)

        self.blur_sigma = blur_sigma
        self.kernel_size = kernel_size

    def __len__(self) -> int:
        return len(self.rgb_vals)

    def __getitem__(self, idx):
        return (self.coords[idx],
                self.rgb_vals[idx],
                self.gt_low[idx],
                self.gt_high[idx])


# ------------------------------------------------------------------- helpers

def psnr_with_peak(pred: torch.Tensor, gt: torch.Tensor, peak: float = 1.0) -> torch.Tensor:
    """Generalized PSNR for arbitrary peak (utils.get_psnr assumes peak=1)."""
    mse = torch.mean((pred - gt) ** 2)
    return 20.0 * torch.log10(torch.tensor(peak, device=pred.device) /
                              torch.sqrt(mse))


def to_uint8(t: torch.Tensor, h: int, w: int, *,
             clip_lo: float = 0.0, clip_hi: float = 1.0) -> np.ndarray:
    """Reshape and clip a flat (H*W, 3) tensor into a uint8 HxWx3 image."""
    arr = t.detach().cpu().numpy().reshape(h, w, 3)
    arr = np.clip(arr, clip_lo, clip_hi)
    arr = (arr - clip_lo) / max(clip_hi - clip_lo, 1e-9)
    return (arr * 255).astype(np.uint8)


# ------------------------------------------------------------------- trainer

class Trainer:
    """One image, full-batch Adam, fresh optimizer per image."""

    def __init__(self,
                 image_path: str,
                 res: int,
                 model: FreqDecomposedFCNet,
                 device: torch.device,
                 lr: float,
                 nepochs: int,
                 out_dir: str,
                 viz_every: int,
                 blur_sigma: float):
        self.dataset = FreqDecomposedImageDataset(
            image_path, res, blur_sigma, device
        )
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
        last = {}
        for epoch in pbar:
            self.model.train()
            for coords, rgb, gt_low, gt_high in self.dataloader:
                self.optimizer.zero_grad()
                combined, low, high = self.model(coords)
                loss_combined = self.criterion(combined, rgb)
                loss_low = self.criterion(low, gt_low)
                loss_high = self.criterion(high, gt_high)
                loss = loss_combined + loss_low + loss_high
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            if (epoch + 1) % self.viz_every == 0 or epoch == 0:
                last = self._eval_and_save(epoch)
                pbar.set_description(
                    f'epoch {epoch}, PSNR {last["PSNR"]:.2f} '
                    f'(low_blur {last["PSNR_low_vs_blur"]:.2f}, '
                    f'high_res {last["PSNR_high_vs_residual"]:.2f})'
                )

        return self.model, last

    @torch.no_grad()
    def _eval_and_save(self, epoch: int) -> dict:
        self.model.eval()
        coords = self.dataset.coords
        combined, low, high = self.model(coords)
        gt = self.dataset.rgb_vals
        gt_low = self.dataset.gt_low
        gt_high = self.dataset.gt_high

        # Clamped versions for cross-experiment-comparable PSNRs.
        combined_c = torch.clamp(combined, 0.0, 1.0)
        low_c = torch.clamp(low, 0.0, 1.0)
        high_clipped_to_unit = torch.clamp(high, 0.0, 1.0)

        psnr = psnr_with_peak(combined_c, gt, peak=1.0).item()
        psnr_low_vs_gt = psnr_with_peak(low_c, gt, peak=1.0).item()
        psnr_high_vs_gt = psnr_with_peak(high_clipped_to_unit, gt, peak=1.0).item()
        psnr_low_vs_blur = psnr_with_peak(low_c, gt_low, peak=1.0).item()
        # Residual target spans roughly [-1, 1], so peak-to-peak is 2.
        psnr_high_vs_residual = psnr_with_peak(high, gt_high, peak=2.0).item()
        mse_high_vs_residual = torch.mean((high - gt_high) ** 2).item()

        w, h = self.dataset.image.size

        gt_img = to_uint8(gt, h, w)
        combined_img = to_uint8(combined, h, w)
        low_img = to_uint8(low, h, w)
        # For the high-freq image, shift signed residuals into [0, 1] so
        # mid-gray = 0 residual, bright = positive, dark = negative.
        high_shifted = high * 0.5 + 0.5
        gt_high_shifted = gt_high * 0.5 + 0.5
        high_img = to_uint8(high_shifted, h, w)
        gt_high_img = to_uint8(gt_high_shifted, h, w)
        gt_low_img = to_uint8(gt_low, h, w)

        save_pair_with_text(
            np.hstack([gt_img, combined_img]),
            f'PSNR: {psnr:.2f}',
            os.path.join(self.out_dir, f'output_{epoch}.png'),
        )
        # Show low vs the blurred GT it was actually supervised against,
        # so the panel is self-consistent for this experiment.
        save_pair_with_text(
            np.hstack([gt_low_img, low_img]),
            f'Low vs blur(GT) PSNR: {psnr_low_vs_blur:.2f}',
            os.path.join(self.out_dir, f'output_low_freq_{epoch}.png'),
        )
        # Show high vs the high-pass residual it was supervised against
        # (both shifted into [0, 1] for display).
        save_pair_with_text(
            np.hstack([gt_high_img, high_img]),
            f'High vs (GT - blur) PSNR: {psnr_high_vs_residual:.2f}',
            os.path.join(self.out_dir, f'output_high_freq_{epoch}.png'),
        )

        return {
            'PSNR': psnr,
            'PSNR_low_vs_gt': psnr_low_vs_gt,
            'PSNR_high_vs_gt': psnr_high_vs_gt,
            'PSNR_low_vs_blur': psnr_low_vs_blur,
            'PSNR_high_vs_residual': psnr_high_vs_residual,
            'MSE_high_vs_residual': mse_high_vs_residual,
        }


# ---------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Exp3: literal frequency decomposition '
                    'via blurred-GT and residual-GT auxiliary losses.'
    )
    parser.add_argument('-project', type=str, required=True,
                        help='W&B project name (required).')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-image_size', type=int, default=128)
    parser.add_argument('-image_dir', type=str, default='circles4')
    parser.add_argument('-num_res', type=int, default=10)
    parser.add_argument('-blur_sigma', type=float, default=4.0,
                        help='Gaussian blur sigma in pixels for the '
                             'low-frequency target. Larger sigma => more '
                             'is offloaded to the high-freq residual.')
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

    image_dir = args.image_dir
    if not os.path.isabs(image_dir):
        image_dir = str(PROJECT_ROOT / image_dir)
    image_paths = sorted(os.listdir(image_dir))
    if not image_paths:
        raise RuntimeError(f'No images found in {image_dir}')

    reset_tag = args.reset if args.reset else 'none'
    sigma_tag = f's{args.blur_sigma:g}'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = (
        f'exp3_{sigma_tag}_{args.image_size}_{args.seed}_adam_'
        f'{args.training_mode}_{reset_tag}_{timestamp}'
    )
    out_dir_root = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir_root, exist_ok=True)

    config = {
        **vars(args),
        'experiment': 'exp3_frequency_decomposition',
        'arch': 'FreqDecomposedFCNet (no output sigmoid; combined = low + high)',
        'image_dir_resolved': image_dir,
        'num_images': len(image_paths),
    }
    init_wandb(
        project=args.project,
        run_name=run_name,
        config=config,
        tags=['exp3_frequency_decomposition', args.training_mode,
              reset_tag, sigma_tag] + list(args.tags),
        group=f'exp3_frequency_decomposition_{sigma_tag}',
    )
    wandb.config.update({'out_dir_root': out_dir_root}, allow_val_change=True)

    print(f'Run name : {run_name}')
    print(f'Out dir  : {out_dir_root}')
    print(f'Image dir: {image_dir} ({len(image_paths)} images)')
    print(f'Mode     : {args.training_mode} (reset={reset_tag})')
    print(f'Blur     : sigma={args.blur_sigma}')

    model: FreqDecomposedFCNet | None = None
    outer = tqdm(enumerate(image_paths), total=len(image_paths),
                 desc='Images', position=0)

    for counter, image_name in outer:
        outer.set_postfix_str(image_name)

        if args.training_mode == 'scratch' or model is None:
            model = FreqDecomposedFCNet(num_res=args.num_res).to(device)

        trainer = Trainer(
            image_path=os.path.join(image_dir, image_name),
            res=args.image_size,
            model=model,
            device=device,
            lr=args.lr,
            nepochs=args.nepochs,
            out_dir=out_dir_root,
            viz_every=args.viz_every,
            blur_sigma=args.blur_sigma,
        )
        model, metrics = trainer.run()

        wandb.log({
            **metrics,
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
