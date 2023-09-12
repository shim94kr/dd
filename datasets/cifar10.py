from typing import Optional, Callable
from torchvision import transforms
from torchvision.datasets import CIFAR10 as CIFAR10_torch
from PIL import Image
class CIFAR10(CIFAR10_torch):

    def __init__(
            self,
            config,
    ) -> None:
        root = './data'
        split = config.dataset.split
        mean = config.dataset.mean
        std = config.dataset.std
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Resize((32, 32), antialias = True),
                                        transforms.Normalize(mean=mean, std=std)])

        if split == "train":
            super().__init__(root, train=True, transform=transform, download = True)
        elif split == "test":
            super().__init__(root, train=False, transform=transform, download = True)
    
    '''
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % 1
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    '''