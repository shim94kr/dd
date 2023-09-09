from typing import Optional, Callable
from torchvision import transforms
from torchvision.datasets import MNIST as MNIST_torch

class MNIST(MNIST_torch):
    def __init__(
            self,
            config,
    ) -> None:
        root = './data'
        split = config.dataset.split
        mean = config.dataset.mean
        std = config.dataset.std
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Resize((32, 32)),
                                        transforms.Normalize(mean=mean, std=std)])

        if split == "train":
            super().__init__(root, train=True, transform=transform, download = True)
        elif split == "test":
            super().__init__(root, train=False, transform=transform, download = True)