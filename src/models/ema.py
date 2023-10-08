import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class FeatEMA():
    def __init__(self, num_classes, ipc, num_channel, device, beta=0.999):
        self.num_classes = num_classes
        self.ipc = ipc
        self.val = torch.randn(size=(num_classes, ipc, num_channel), dtype=torch.float, device=device)
        self.beta = beta

    def update(self, out, label, label_2nd = None):
        label_2nd = torch.zeros_like(label) if label_2nd is None else label_2nd
        label_total = label + self.ipc * label_2nd
        label_matrix = F.one_hot(label_total, self.num_classes * self.ipc)
        label_matrix = label_matrix / (label_matrix.sum(dim=0, keepdim=True) + 1e-5)

        out = torch.einsum('bl, bc -> lc', label_matrix, out)
        out = self.beta * self.val.flatten(0,1) + (1 - self.beta) * out
        out = torch.where(label_matrix.sum(dim=0)[:,None] > 0, out, self.val.flatten(0,1))

        self.val = out.reshape([self.num_classes, self.ipc] + list(self.val.shape[2:]))

class ImageEMA():
    def __init__(self, num_classes, ipc, num_channel, img_size, device, beta=0.0):
        self.num_classes = num_classes
        self.ipc = ipc
        self.val = torch.randn(size=(num_classes, ipc, num_channel, img_size, img_size), dtype=torch.float, device=device)
        self.beta = beta

    def update(self, out, label, label_2nd = None):
        label_2nd = torch.zeros_like(label) if label_2nd is None else label_2nd
        label_total = label + self.ipc * label_2nd
        label_matrix = F.one_hot(label_total, self.num_classes * self.ipc)
        label_matrix = label_matrix / (label_matrix.sum(dim=0, keepdim=True) + 1e-5)

        out = torch.einsum('bl, bchw -> lchw', label_matrix, out)
        out = self.beta * self.val.flatten(0, 1) + (1 - self.beta) * out
        out = torch.where(label_matrix.sum(dim=0)[:,None,None,None] > 0, out, self.val.flatten(0,1))

        self.val = out.reshape([self.num_classes, self.ipc] + list(self.val.shape[2:]))

class ModelEMA(torch.nn.Module):
    def __init__(self, init_module, mu):
        super(ModelEMA, self).__init__()

        self.module = init_module
        self.mu = mu

    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)
    
    def encode(self, x, *args, **kwargs):
        return self.module.encode(x, *args, **kwargs)
    def decode(self, x, *args, **kwargs):
        return self.module.decode(x, *args, **kwargs)

    def update(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            # see : https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/ExponentialMovingAverage?hl=PL
            mu = min(self.mu, (1.0 + step) / (10.0 + step))

        state_dict = {}
        with torch.no_grad():
            for (name, m1), (name2, m2) in zip(self.module.state_dict().items(), module.state_dict().items()):
                if name != name2:
                    logger.warning("[ExpoentialMovingAverage] not matched keys %s, %s", name, name2)

                if step is not None and step < 0:
                    state_dict[name] = m2.clone().detach()
                else:
                    state_dict[name] = ((mu * m1) + ((1.0 - mu) * m2)).clone().detach()

        self.module.load_state_dict(state_dict)