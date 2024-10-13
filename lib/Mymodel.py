import torch
from torch import nn


class DNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack_model = nn.Sequential(
            nn.Linear(8,16),
            nn.PReLU(),
            nn.Linear(16,4),
            nn.PReLU(),
            nn.Linear(4,1),
            nn.Sigmoid(),

        )
        
        # self.init_weight()
    
    # def init_weight(self):
    #     print("Doing initialization")
    
    def forward(self,x):
        return self.stack_model(x)

class model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack_model = nn.Sequential(
            nn.Linear(8,8),
            nn.LeakyReLU(),
            nn.Linear(8,4),
            nn.LeakyReLU(),
            nn.Linear(4,1),
            nn.Sigmoid(),
        )
  
    def forward(self,x):
        return self.stack_model(x)


class model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack_model = nn.Sequential(
            nn.Linear(8,4),
            nn.LeakyReLU(),
            nn.Linear(4,2),
            nn.LeakyReLU(),
            nn.Linear(2,1),
            nn.Sigmoid(),
        )
        

    def forward(self,x):
        return self.stack_model(x)


class LSTM_model(nn.Module):
    def __init__(self,input_size,hidden_size,num_layer,output_size,device):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layer   = num_layer
        self.output_size = output_size
        self.device      = device

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layer,batch_first=True)

        # output layer
        self.fc   = nn.Linear(self.hidden_size,self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).requires_grad_().to(self.device)


        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])  # Selecting the last output
        out = self.sigmoid(out)
        return out



