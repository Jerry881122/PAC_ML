import pandas as pd
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# my own lib
from lib.MyDataset import MyDataset
from lib.SetCuda import SetCuda
from lib import Mymodel      # model function
from lib import train        # training function
from lib import test         # testing function
from lib.loss import weighted_BCE_loss
from lib.loss import FocalLoss


print("\nsimulated environment")
print('\tTorch',torch.__version__,'CUDA',torch.version.cuda)
print('\tcuDNN version: ' + str(torch.backends.cudnn.version()))
print('\tusing CUDA = ' + str(torch.cuda.is_available()))  # 檢查 CUDA 是否可用
print('\n')


# setting device
device = SetCuda()


# define parameter
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
print_LR = LR


# using pandas to read csv file             0 -> correct codeword , 1 -> error codeword 
df = pd.read_csv('resource/train/SNR2_L8/L=8_frame=100000_SNR=2.csv')
df_test = pd.read_csv('resource/test/SNR2_L8/L=8_frame=25000_SNR=2.csv')


# pandas -> numpy -> tensor , and first 8 data are feature & last data are label
feature = torch.tensor(df.iloc[:,0:-1].to_numpy(dtype=np.float32))
feature = feature.view(len(feature) , len(feature[0]) , 1)          # change shape to (10000,8,1)
label = torch.tensor(df.iloc[:,-1:].to_numpy(dtype=np.float32))

dataset = MyDataset(feature,label)
print(len(dataset))
train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

print(feature.shape)
print(label.shape)


# setting model parameter
input_size = 1
hidden_size = 50
num_layer = 2
output_size = 1

model = Mymodel.LSTM_model(input_size=input_size,hidden_size=hidden_size,
                            num_layer=num_layer,output_size=output_size).to(device)
print(model)



# # Loss and optimizer
# criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 100

# for epoch in range(num_epochs):
#     for inputs, targets in train_loader:
#         # Zero the gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, targets.unsqueeze(-1))
#         # loss.shape
        
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
    
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')