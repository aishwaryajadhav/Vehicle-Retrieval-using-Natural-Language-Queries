from torch import nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, average = True):
        distance_positive = nn.CosineSimilarity()(anchor, positive)
        distance_negative = nn.CosineSimilarity()(anchor, negative)

        losses = F.relu(- distance_positive + distance_negative + self.margin)

        if(average):
            return losses.mean() 
        else:
            return losses.sum()