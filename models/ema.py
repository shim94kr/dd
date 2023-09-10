import torch

class ImageEMA():
    def __init__(self, num_classes, ipc, num_channel, img_size, device, beta=0.9999):
        self.val = torch.randn(size=(num_classes, ipc, num_channel, img_size, img_size), dtype=torch.float, device=device)
        self.beta = beta

    def update(self, out, label, label_2nd = None):
        self.val[label, label_2nd] = self.beta * self.val[label, label_2nd] + (1 - self.beta) * out