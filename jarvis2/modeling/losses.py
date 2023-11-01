import torch
from torch.nn import functional as F


def cos_loss(y, sim):
    return 2 * torch.mean(torch.abs(y - sim))


def contrastive_loss(y, sim, margin=0.5):
    """
    Contrastive loss from Hadsell-et-al.'06
    https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    dist = 1 - sim
    return torch.mean(y * 2 * dist + (1 - y) * 2 * F.relu(margin - dist))
