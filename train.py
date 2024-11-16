import os

import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import ImageDataset
from model import FCNet
import utils
from torch.utils.tensorboard import SummaryWriter

from accelerate.utils import set_seed
class Trainer:
    def __init__(self, image_path, res, use_pe=True, device='cuda', batch_size = 4096, nepochs = 200, model = None, out_dir = 'output'):
        self.dataset = ImageDataset(image_path, res, device)
        self.res = res
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        out_dir = os.path.join(out_dir, image_path.split('/')[-1].split('.')[0])
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        if model is not None:
            self.model = model
        else:
            self.model = FCNet(use_pe, num_res = 10, num_layers = 2, width=256).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = nepochs

    def run(self):
        pbar = tqdm(range(self.nepochs))
        for epoch in pbar:
            self.model.train()
            for coords, rgb_vals in self.dataloader:
                self.optimizer.zero_grad()
                out = self.model(coords)
                loss = self.criterion(out, rgb_vals)
                loss.backward()
                self.optimizer.step()


            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    coords = self.dataset.coords
                    pred = self.model(coords)
                    gt = self.dataset.rgb_vals
                    psnr = utils.get_psnr(pred, gt)
                pbar.set_description(f'Epoch:: {epoch}, PSNR: {psnr.item()}')
                pred = pred.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
                pred = (pred * 255).astype(np.uint8)
                gt = self.dataset.rgb_vals.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
                gt = (gt * 255).astype(np.uint8)
                save_image = np.hstack([gt, pred])
                self.visualize(np.array(save_image), text='PSNR: {:.2f}'.format(psnr), epoch = epoch)

        return self.model, psnr




    def visualize(self, image, text, epoch):
        save_image = np.ones((self.res + 50, 2 * self.res, 3), dtype=np.uint8) * 255
        img_start = 50
        save_image[img_start:img_start + self.res, :, :] = image
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        position = (100, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(save_image, text, position, font, scale, color, thickness)

        cv2.imwrite(os.path.join(self.out_dir, f'output_{epoch}.png'), save_image)



if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    writer = SummaryWriter(f'./runs_{seed}_scratch')
    image_dir = 'circles4'
    image_paths = sorted(os.listdir('circles4'))

    model = None
    for counter, image_path in enumerate(image_paths):
        print(image_path)
        trainer = Trainer(os.path.join(image_dir, image_path), 256, batch_size=256*256, nepochs=500, model = None, out_dir='output4')
        model, psnr = trainer.run()
        #trainer = Trainer(os.path.join(image_dir, image_path), 256, batch_size=256*256, nepochs=500, model = None, out_dir=f'output{5}')
        #model_scratch, psnr_scratch = trainer.run()
        writer.add_scalar('PSNR', psnr, counter)
        #writer.add_scalar('PSNR_scratch', psnr_scratch, counter)
        #writer.add_scalar('PSNR_diff', psnr - psnr_scratch, counter)