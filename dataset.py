import torch
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import utils



class ImageDataset(Dataset):
    def __init__(self, image_path, res, device = 'cuda'):
        self.image = Image.open(image_path).convert('RGB')
        self.image = utils.crop_and_resize(self.image, res)
        self.rgb_vals = TF.to_tensor(self.image).to(device).reshape(3, -1).T
        self.coords = utils.get_coords(res, normalize = True).to(device).reshape(-1, 2)

    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        return self.coords[idx], self.rgb_vals[idx]



if __name__ == '__main__':

    # image_path = 'mona.webp'
    # res = 512
    # device = 'cuda'
    # dataset = ImageDataset(image_path, res, device)

    image_res = 256
    image = np.ones((image_res, image_res, 3)) * 128
    num_circles = 10

    #divide image to equal parts and in each part draw a circle in the center with random color

    # number_per_row = 8
    # color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    # for i in range(number_per_row):
    #     for j in range(number_per_row):
    #         (x, y) = (image_res // number_per_row * (i + 0.5), image_res // number_per_row * (j + 0.5))
    #         x = int(x)
    #         y = int(y)
    #         radius = image_res // number_per_row // 2
    #
    #         cv2.circle(image, (x, y), radius, color, -1)
    #
    #         #save the image
    #         save_image = Image.fromarray(image.astype(np.uint8))
    #
    #         #save_the image as circle_000i.png
    #         save_image.save('circle_{:03}.png'.format(i * number_per_row + j))

    #draw a circle with random position, radius and color
    # for i in range(50):
    #     (x, y) = (np.random.randint(0, image_res), np.random.randint(0, image_res))
    #     radius = np.random.randint(20, 50)
    #     color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    #
    #     cv2.circle(image, (x, y), radius, color, -1)
    #
    #     #save the image
    #     save_image = Image.fromarray(image.astype(np.uint8))
    #
    #
    #     save_image.save('circle_{:03}.png'.format(i))

    #draw num_circles random circles with random colors and positions and create 50 such images

    for i in range(50):
        image = np.ones((image_res, image_res, 3)) * 128
        for j in range(num_circles):
            (x, y) = (np.random.randint(0, image_res), np.random.randint(0, image_res))
            radius = np.random.randint(20, 50)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

            cv2.circle(image, (x, y), radius, color, -1)

        #save the image
        save_image = Image.fromarray(image.astype(np.uint8))


        save_image.save('circle_{:03}.png'.format(i))
