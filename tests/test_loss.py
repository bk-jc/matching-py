import torch
from torch import nn

from src.losses import contrastive_loss

FP_TOLERANCE = 1e-6


def test_loss_similar_vectors():
    vector = torch.ones(10).view(1, -1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-12)
    cos_dist = cos(vector, vector)

    # Check cosine distance is 1
    assert torch.isclose(cos_dist, torch.tensor(1.), atol=FP_TOLERANCE)

    # Check that loss is low positive pair
    loss = contrastive_loss(cos_dist, sim=torch.tensor(1))
    assert torch.isclose(loss, torch.tensor(0.), atol=FP_TOLERANCE)

    # Check that loss is high for negative pair
    loss = contrastive_loss(cos_dist, sim=torch.tensor(0))
    assert torch.isclose(loss, torch.tensor(2.), atol=FP_TOLERANCE)


def test_loss_dissimilar_vectors():
    vector_one = torch.ones(10).view(1, -1)
    vector_two = -torch.ones(10).view(1, -1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-12)
    cos_dist = cos(vector_one, vector_two)

    # Check cosine distance is -1
    assert torch.isclose(cos_dist, torch.tensor(-1.), atol=FP_TOLERANCE)

    # Check that loss is high for positive pair
    loss = contrastive_loss(cos_dist, sim=torch.tensor(1))
    assert torch.isclose(loss, torch.tensor(2.), atol=FP_TOLERANCE)

    # Check that loss is low for negative pair
    loss = contrastive_loss(cos_dist, sim=torch.tensor(0))
    assert torch.isclose(loss, torch.tensor(-2.), atol=FP_TOLERANCE)
