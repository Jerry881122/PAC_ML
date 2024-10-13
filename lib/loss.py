import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,weight=None,gamma=2):
        super().__init__()

        self.gamma = gamma
        self.pos_weight = weight[0] if weight is not None else 1.0
        self.neg_weight = weight[1] if weight is not None else 1.0

    def forward(self,output,label):

        output = torch.clamp(output, min=(1e-12), max=(1 - 1e-12))  # 避免 log(0)

        loss = -(label * self.pos_weight * torch.pow((1-output), self.gamma) * torch.log(output) + (1-label) * self.neg_weight * torch.pow(output, self.gamma) * torch.log(1-output) )
        
        return loss.mean()



class weighted_BCE_loss(nn.Module):

    def __init__(self,weight=None):
        super().__init__()
        self.pos_weight = weight[0] if weight is not None else 1.0
        self.neg_weight = weight[1] if weight is not None else 1.0

    def forward(self,output,label):

        output = torch.clamp(output, min=(1e-8), max=(1 - 1e-8))  # 避免 log(0)

        loss = -( self.pos_weight * label * torch.log(output) + self.neg_weight * (1-label) * torch.log(1-output) )
        loss = loss.mean()  # mean for mini-batch

        # loss = torch.clamp(loss,min=(1e-8), max=(1 - 1e-8)) # bounding

        return loss         
  
