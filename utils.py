import numpy as np
from PIL import Image
import torch
import cv2


def crop_and_resize(image, res):
    # center crop the image
    w, h = image.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))
    image = image.resize((res, res))

    return image

def get_coords(res, normalize = False):
    x = y = torch.arange(res)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.stack((xx, yy), dim = -1)
    if normalize:
        coords = coords / (res - 1)

    return coords

def get_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr
