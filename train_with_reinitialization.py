import os

import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import ImageDataset
from updated_models import FCNet
import utils
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from accelerate.utils import set_seed
from optim_adahessian import Adahessian
from seng import SENG
import argparse
from parser_utils import str2bool
from high_frequency_detection import detect_high_frequency

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
        elif optimizer == 'seng':
            self.optimizer =  SENG(self.model, 1.2, update_freq=200) # TODO not sure about the params
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
                if args.optimizer == 'seng' or args.optimizer == 'adahessian' or args.optimizer == 'lbfgs':
                    loss.backward(create_graph=True)
                else:
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

        #cv2.imwrite(os.path.join(self.out_dir, f'output_{epoch}.png'), save_image)




if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Continual Learning on the Circles with optional reinitialization')
    parser.add_argument('-optimizer',
                        choices=['adahessian', 'adam', 'lbfgs', 'seng'],
                        help='optimizer for training the model',
                        default= 'adam') # TODO implement others
    
    parser.add_argument('-seed',  type = int , default= 42 , help = 'our random seed') 

    # parser.add_argument('-batch_size',  type = int , default= 4096 , help = 'batch_size') 

    parser.add_argument('-image_size',  type = int , default= 256 , help = 'input image size') 
    parser.add_argument('-lr',  type = float , default= 1e-3 , help = 'learning_rate') 
    parser.add_argument('-nepochs',  type = int , default= 500 , help = 'epochs') 
    parser.add_argument('-training_mode',  choices=['continual', 'scratch'],
                        help='training_mode',
                        default= 'scratch') 
    parser.add_argument('-reinitialize', type=str2bool, nargs='?', 
                        const=True, default= True, help='whether to reinitialize the high frequency nuerons')
    
    parser.add_argument('-re_inputs', type=str2bool, nargs='?', 
                        const=True, default= True, help='if we are reinitializing, should we reinitializing inputs?')
    
    parser.add_argument('-re_outputs', type=str2bool, nargs='?', 
                        const=True, default= True, help='if we are reinitializing, should we reinitializing outputs?')
    
    parser.add_argument('-re_th',  type = float , default= None , help = 'threshold for finding active neurons') 

    parser.add_argument('-re_percentage',  type = float , default= None , help = 'percentage for finding active neurons in each layer') 

    parser.add_argument('-use_low', type=str2bool, nargs='?', 
                        const=True, default= False, help='wether to use low frequency samples for reinitializing neurons as well')

    
    

    args = parser.parse_args()
    set_seed(args.seed)
<<<<<<< HEAD
    writer = SummaryWriter(f'./activation_base_reinit/runs_{args.seed}_{args.optimizer}_{args.training_mode}_{args.reinitialize}_{args.re_inputs}_{args.re_outputs}_{args.re_th}_{args.re_percentage}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
=======
    writer = SummaryWriter(f'./runs_{args.seed}_{args.optimizer}_{args.training_mode}_{args.reinitialize}_{args.use_low}_{args.re_inputs}_{args.re_outputs}_{args.re_th}_{args.re_percentage}')
>>>>>>> e05ca4696ded760dc73db1dc584c4df407bb92d7
    #image_dir = f'circles4_reinitialization_{args.re_inputs}_{args.re_outputs}_{args.re_th}'

    image_dir = 'circles4'
    image_paths = sorted(os.listdir(f'circles4'))

    model = None
    print(f'Model is being trained in {args.training_mode} mode')
    print(f'The input image dir is {image_dir}')
    print(f'The optimizer is {args.optimizer}')
    print(f'The learning rate is {args.lr}')
    print(f'The number of epochs is {args.nepochs}')
    print(f'The image size is {args.image_size}')
    print(f'Reinitialization is  {args.reinitialize}')
    print(f're_inputs is {args.re_inputs}')
    print(f're_outputs is {args.re_outputs}')
    print(f'the threshold for reinitialization is {args.re_th}')
    print(f'using low frequency points as well : {args.use_low}')
    # Set device to 'cuda' if CUDA is available, otherwise default to 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    if device.type == "cuda":
        pass
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)


    
    for counter, image_path in enumerate(image_paths):
        print(image_path)
        if args.training_mode == 'scratch':
            trainer = Trainer(os.path.join(image_dir, image_path), args.image_size, batch_size= args.image_size* args.image_size,
                           nepochs= args.nepochs, optimizer= args.optimizer , lr= args.lr, device= device,
<<<<<<< HEAD
                              model = None, out_dir=f'./output4/reinitialization_{args.re_inputs}_{args.re_outputs}_{args.re_th}')
=======
                              model = None, out_dir=f'output4_reinitialization_{args.re_inputs}_{args.re_outputs}_{args.re_th}_{args.re_percentage}')
>>>>>>> e05ca4696ded760dc73db1dc584c4df407bb92d7
            model, psnr = trainer.run()
        else:
            trainer = Trainer(os.path.join(image_dir, image_path), args.image_size, batch_size= args.image_size* args.image_size,
                           nepochs= args.nepochs, optimizer= args.optimizer , lr= args.lr, device= device,
<<<<<<< HEAD
                               model = model, out_dir=f'./output4/reinitialization_{args.re_inputs}_{args.re_outputs}_{args.re_th}')
=======
                               model = model, out_dir=f'output4_reinitialization_{args.re_inputs}_{args.re_outputs}_{args.re_th}_{args.re_percentage}')
>>>>>>> e05ca4696ded760dc73db1dc584c4df407bb92d7
            model, psnr = trainer.run()
        
        if args.reinitialize:
            print('reinitalizing the model on high frequencey inputs (edges)')
            
            print('getting the coordinates with high frequency:')
            high_frequency_coordinates, _, _, _ = detect_high_frequency(image_path= os.path.join(image_dir, image_path))

            print(f'number of coordinates with high frequency : {len(high_frequency_coordinates)}')

            high_frequency_coordinates_tensor = torch.tensor(high_frequency_coordinates)
            

            low_frequency_coordinates_tensor = None
            if args.use_low: # using the low frequency points as well

                # Generate coordinate arrays
                x = np.arange(args.image_size)
                y = np.arange(args.image_size)
                xx, yy = np.meshgrid(x, y)

                # Stack and reshape the coordinate arrays
                all_coordinates = np.stack((xx.ravel(), yy.ravel()), axis=-1).tolist()
                low_frequency_coordinates = [coordinate for coordinate in all_coordinates if coordinate not in high_frequency_coordinates]
                low_frequency_coordinates_tensor = torch.tensor(low_frequency_coordinates)


            model.reinitialize_neurons(X = high_frequency_coordinates_tensor, threshold= args.re_th, top_percentage= args.re_percentage,
                                       reinit_input= args.re_inputs, reinit_output= args.re_outputs, X_negative= low_frequency_coordinates_tensor)

        
        writer.add_scalar('PSNR', psnr, counter)
        
# for training adam continually with reinitiliazation with percentage (only reinitialize top 10% in each layer) and using low frequency points as well
# python train_with_reinitialization.py -optimizer adam -training_mode continual -re_percentage 0.1 -use_low True

# for training adam continually with reinitiliazation
# python train_with_reinitialization.py -optimizer adam -training_mode continual -re_th 0.8


# for training adam continually with reinitiliazation with percentage (only reinitialize top 10% in each layer)
# python train_with_reinitialization.py -optimizer adam -training_mode continual -re_percentage 0.1

#for training adam continually with reinitiliazation with percentage and threshold (only reinitialize top 10% in each layer that have activation greater than threshold)
# python train_with_reinitialization.py -optimizer adam -training_mode continual -re_percentage 0.1 -re_th 0.8

# for training adam continually with reinitiliazation only on input weights
# python train_with_reinitialization.py -optimizer adam -training_mode continual -re_th 0.8 -re_outputs False

# for training adam continually with reinitiliazation only on output weights
# python train_with_reinitialization.py -optimizer adam -training_mode continual -re_th 0.8 -re_inputs False