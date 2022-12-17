import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        squarred_distance_1 = (anchor - positive).pow(2).sum(1)

        squarred_distance_2 = (anchor - negative).pow(2).sum(1)

        triplet_loss = F.relu(
            self.margin + squarred_distance_1 - squarred_distance_2
        ).mean()

        return triplet_loss
