import numpy as np
import torch
import torch.nn.functional as F
import random

import torchvision.utils as tvls

def save_tensor_images(images, filename, nrow = None, normalize=True):
    if not nrow:
        tvls.save_image(images, filename, normalize=True,padding=0)
    else:
        tvls.save_image(images, filename, nrow=nrow, normalize=True, padding=0)