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

# begin timer
timer_begin = time.time()

name = input("tmp_model/NPM/")
path_name = "tmp_model/NPM/" + name + ".pth"
fig_name  = "tmp_model/NPM/" + name + ".jpg"
print(path_name)

# define parameter
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.01
print_LR = LR

# setting device
device = SetCuda()

# using pandas to read csv file
df = pd.read_csv('resource/train/SNR2_L8/NPM/L=8_frame=20000_train_1.csv')
df_test = pd.read_csv('resource/test/SNR2_L8/NPM/L=8_frame=10000_test_1.csv')


# pandas -> numpy -> tensor , and first 8 data are feature & last data are label
feature = torch.tensor(df.iloc[:,0:-1].to_numpy(dtype=np.float32))
label = torch.tensor(df.iloc[:,-1:].to_numpy(dtype=np.float32))
feature_test = torch.tensor(df_test.iloc[:,0:-1].to_numpy(dtype=np.float32))
label_test = torch.tensor(df_test.iloc[:,-1:].to_numpy(dtype=np.float32))

# using dataset and DataLoader from pytorch to store train data
dataset = MyDataset(feature,label)
dataset_test = MyDataset(feature_test,label_test)
train_loader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle = True)
test_loader = DataLoader(dataset_test,batch_size = BATCH_SIZE,shuffle = False)

# setting deep learning model
model = Mymodel.DNN_1().to(device)
print(model)



# setting loss function & gradient decent
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr=LR)

# training model
losses = []
for epoch in range(EPOCHS):
    # training data
    train.training_data(model,device,train_loader,optimizer,criterion,epoch,losses)

    if(epoch == EPOCHS-1):
        # testing the training data
        print('\ntesting training data')
        test.testing_data(model,device,train_loader,criterion)
        # testing the testing data
        print('\ntesting testing data')
        test.testing_data(model,device,test_loader,criterion)
        print('///////////////////////////////\n')

    # update the learning rate
    if epoch % 10 == 0 and epoch != 0  :
        LR = LR * 0.9
        print('LR update')

timer_end = time.time()
print("execution time =",timer_end-timer_begin,' second')        

# figure of loss v.s. weight update
train.loss_update_fig(losses,fig_name)


# print out model weight
# params = model.parameters()
# for param in params:
#     print(param)

# save the model weight 
torch.save(model.state_dict() , path_name)



print(model)
print('\nParameter\n\tEPOCHS =',EPOCHS)
print('\tBatch size =',BATCH_SIZE)
print('\tLearning rate =',print_LR,'\n')


