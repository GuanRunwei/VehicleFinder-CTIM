import torch
import torch.nn as nn
from torchvision import transforms


image_transforms = transforms.Compose(
    [
        transforms.ColorJitter(),
        # transforms.ToTensor()
    ]
)