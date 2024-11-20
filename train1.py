import os

import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import ImageDataset
from model1 import FCNet
import utils
from torch.utils.tensorboard import SummaryWriter

from accelerate.utils import set_seed
from optim_adahessian import Adahessian
import argparse
from datetime import datetime
class Trainer:
    def __init__(self, image_path, res, use_pe=True, device='cuda', batch_size = 4096, 
                 nepochs = 200, model = None, out_dir = 'output', optimizer = 'adam', lr = 1e-3):
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
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr= lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        elif optimizer == 'adahessian':
            self.optimizer = Adahessian(self.model.parameters())
        elif optimizer == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = nepochs

    def run(self):
        pbar = tqdm(range(self.nepochs))
        for epoch in pbar:
            self.model.train()
            for coords, rgb_vals in self.dataloader:
                self.optimizer.zero_grad()
                out, low_freq, high_freq = self.model(coords)
                loss = self.criterion(out, rgb_vals)
                loss_low_freq = self.criterion(low_freq, rgb_vals)
                loss = loss + loss_low_freq
                loss.backward()
                self.optimizer.step()


            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.model.eval()
                with torch.no_grad():
                    coords = self.dataset.coords
                    pred, low_freq, high_freq = self.model(coords)
                    gt = self.dataset.rgb_vals
                    psnr = utils.get_psnr(pred, gt)
                pbar.set_description(f'Epoch:: {epoch}, PSNR: {psnr.item()}')
                pred = pred.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
                pred = (pred * 255).astype(np.uint8)
                gt = self.dataset.rgb_vals.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
                gt = (gt * 255).astype(np.uint8)
                save_image = np.hstack([gt, pred])
                self.visualize(np.array(save_image), text='PSNR: {:.2f}'.format(psnr), epoch = epoch, save_name='output_{}.png'.format(epoch))
                low_freq = low_freq.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
                low_freq = (low_freq * 255).astype(np.uint8)
                high_freq = high_freq.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
                high_freq = (high_freq * 255).astype(np.uint8)
                save_image_low_freq = np.hstack([gt, low_freq])
                save_image_high_freq = np.hstack([gt, high_freq])
                self.visualize(np.array(save_image_low_freq), text='Low Freq', epoch = epoch, save_name='output_low_freq_{}.png'.format(epoch))
                self.visualize(np.array(save_image_high_freq), text='High Freq', epoch = epoch, save_name='output_high_freq_{}.png'.format(epoch))

        return self.model, psnr




    def visualize(self, image, text, epoch, save_name):
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

        cv2.imwrite(os.path.join(self.out_dir, f'{save_name}'), save_image)




if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Continual Learning on the Circles')
    parser.add_argument('-optimizer',
                        choices=['adahessian', 'adam', 'lbfgs'],
                        help='optimizer for training the model',
                        default= 'adam') # TODO implement others
    
    parser.add_argument('-seed',  type = int , default= 42 , help = 'our random seed') 

    # parser.add_argument('-batch_size',  type = int , default= 4096 , help = 'batch_size') 

    parser.add_argument('-image_size',  type = int , default= 128 , help = 'input image size') 
    parser.add_argument('-lr',  type = float , default= 1e-3 , help = 'learning_rate') 
    parser.add_argument('-nepochs',  type = int , default= 500 , help = 'epochs') 
    parser.add_argument('-training_mode',  choices=['continual', 'scratch'],
                        help='training_mode',
                        default= 'scratch') 
    
    parser.add_argument('-reset',  choices=['no', 'high', 'low', ''], default='') 

    args = parser.parse_args()
    set_seed(args.seed)

    if args.training_mode == 'scratch':
        assert args.reset == '', 'reset should be empty for scratch mode'

    path = f'new_arch/{args.image_size}_{args.seed}_{args.optimizer}_{args.training_mode}_{args.reset}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(f'./runs10/{path}')
    image_dir = 'circles4'
    image_paths = sorted(os.listdir('circles4'))

    model = None
    print(f'Model is being trained in {args.training_mode} mode')
    print(f'The input image dir is {image_dir}')
    print(f'The optimizer is {args.optimizer}')
    print(f'The learning rate is {args.lr}')
    print(f'The number of epochs is {args.nepochs}')
    print(f'The image size is {args.image_size}')

    
    for counter, image_path in enumerate(image_paths):
        print(image_path)
       
        if args.training_mode == 'scratch':
            trainer = Trainer(os.path.join(image_dir, image_path), args.image_size, batch_size= args.image_size* args.image_size,
                           nepochs= args.nepochs, optimizer= args.optimizer , lr= args.lr,
                              model = None, out_dir=f'./output4/{path}')
            model, psnr = trainer.run()
        else:
            trainer = Trainer(os.path.join(image_dir, image_path), args.image_size, batch_size= args.image_size* args.image_size,
                           nepochs= args.nepochs, optimizer= args.optimizer , lr= args.lr,
                               model = model, out_dir=f'./output4/{path}')
            model, psnr = trainer.run()
            if args.reset == 'high':
                model.reset_high_freq()
            elif args.reset == 'low':
                model.reset_low_freq()
            elif args.reset == 'no':
                pass
            with torch.no_grad():
                model = model 
        #trainer = Trainer(os.path.join(image_dir, image_path), 256, batch_size=256*256, nepochs=500, model = None, out_dir=f'output{5}')
        #model_scratch, psnr_scratch = trainer.run()
        writer.add_scalar('PSNR', psnr, counter)
        #writer.add_scalar('PSNR_scratch', psnr_scratch, counter)
        #writer.add_scalar('PSNR_diff', psnr - psnr_scratch, counter)


# for training adahessian continually 
# python -optimizer adahessian -training_mode continual

# for training lbfgs continually 
# python3 -m -optimizer lbfgs -training_mode continual

# for training adam continually 
# python3 -m -training_mode continual

# for training adam from scratch
# python3 -m -training_mode scratch