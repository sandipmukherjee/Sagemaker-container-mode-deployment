import os

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import io as skio
from torch.cuda import device
from torchvision.transforms.functional import normalize

from models import ISNetDIS


class DiSegmentPredictor:
    def __init__(self, model_path: str):
        self.model = None
        self.model = self.load_model(model_path)
        self.input_size = [1024, 1024]

    def get_model(self):
        return self.model

    @staticmethod
    def load_image(img_path: str):
        image = skio.imread(img_path)
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
        return image

    def process_image(self, im: numpy.ndarray):
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), self.input_size, mode="bilinear").type(torch.uint8)
        image = torch.divide(im_tensor, 255.0)
        return normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    def predict(self, src_img_path: str):
        """
        it will load the image from url
        it will process the image as required by the model
        it will call the model and get the mask
        :param src_img_path: source image path
        :return: mask after segmentation
        """
        print("im_path: ", src_img_path)
        src_img = self.load_image(src_img_path)
        image_shape = src_img.shape[0:2]
        image = self.process_image(src_img)
        if torch.cuda.is_available():
            image = image.cuda()
        result = self.model(image)
        result = torch.squeeze(F.upsample(result[0][0], image_shape, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)

        if device == 'cuda':
            torch.cuda.empty_cache()
        mask = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)  # it is the mask we need
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
        return Image.fromarray(mask, 'L')

    @staticmethod
    def load_model(model_path):
        net = ISNetDIS()
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        return net


