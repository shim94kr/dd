from typing import Optional, Callable
from torchvision import transforms
from torchvision.datasets import CIFAR10 as CIFAR10_torch
from PIL import Image
class CIFAR10(CIFAR10_torch):

    def __init__(
            self,
            config,
            augmentation = False,
    ) -> None:
        root = './data'
        split = config.dataset.split
        mean = config.dataset.mean
        std = config.dataset.std

        if augmentation:
            transform = transforms.Compose([transforms.ToTensor(),                                             
                                            transforms.Resize((32, 32), antialias = True),
                                            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                            transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                            transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor(), 
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