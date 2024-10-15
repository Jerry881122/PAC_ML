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

# print(np.__version__)

# begin timer
timer_begin = time.time()

# 輸入 model 的名字
name = input("tmp_model/")
path_name = "tmp_model/" + name + ".pth"
loss_name  = "tmp_model/loss_" + name + ".jpg"
learning_rate_name = "tmp_model/lr_" + name + ".jpg"
print(path_name)

# define parameter
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001
print_LR = LR

# setting device
device = SetCuda()

# using pandas to read csv file             0 -> correct codeword , 1 -> error codeword 
# df = pd.read_csv('resource/train/SNR2_L8/L=8_frame=100000_SNR=2.csv')
df_test = pd.read_csv('resource/test/SNR2_L8/L=8_frame=25000_SNR=2.csv')
df = pd.read_csv('resource/train/SNR2_L8/bounding/bounding_train_data.csv')
# df_test = pd.read_csv('resource/test/SNR2_L8/bounding/bounding_test_data.csv')


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
weight = torch.tensor([20,1])   # [pos_weight,neg_weight]

# criterion = nn.BCEWithLogitsLoss(pos_weight=weight[0])
# criterion = weighted_BCE_loss(weight=weight)
criterion = FocalLoss(weight=weight)
# criterion = nn.BCELoss()


optimizer = optim.SGD(model.parameters(),lr=LR)
# optimizer = optim.Adam(model.parameters() , lr=LR , betas=(0.9, 0.999) , eps=1e-08)

# learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.9 , last_epoch = -1)


# training model
losses = []     # storing loss 
learning_rates = []
for epoch in range(EPOCHS):
    # training data
    train.training_data(model,device,train_loader,optimizer,criterion,epoch,losses)

    # output results after all epochs
    if(epoch == EPOCHS-1):
        # testing the training data
        print('\ntesting training data')
        test.testing_data(model,device,train_loader,criterion)
        # testing the testing data
        print('\ntesting testing data')
        test.testing_data(model,device,test_loader,criterion)
        print('///////////////////////////////\n')

    learning_rates.append(optimizer.param_groups[0]['lr'])
    scheduler.step();
    

    # update the learning rate
    # if epoch % 10 == 0 and epoch != 0  :
    #     LR = LR * 0.5
    #     print('LR update')

timer_end = time.time()
print("execution time =",timer_end-timer_begin,' second')        

# figures
train.loss_update_fig(losses,loss_name)
train.learning_rate_updata_fig(learning_rates,learning_rate_name)

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


