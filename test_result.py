import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# my own lib
from lib.MyDataset import MyDataset
from lib.SetCuda import SetCuda
from lib import Mymodel      # model function
from lib import test         # testing function



# setting device
device = SetCuda()


############################################### need  to modify #################################################
# load model 
model = Mymodel.model_2().to(device)
model.load_state_dict(torch.load("tmp_model/1.pth"))


# load testing data
df_test_SNR_1 = pd.read_csv('resource/test/SNR1_L8/L=8_frame=200000_test_1.csv')
df_test_SNR_2 = pd.read_csv('resource/test/SNR2_L8/L=8_frame=200000_test_1.csv')
df_test_SNR_3 = pd.read_csv('resource/test/SNR3_L8/L=8_frame=200000_test_1.csv')
df = pd.read_csv('resource/train/SNR2_L8/L=8_frame=100000_train_1.csv')

# setting loss function
criterion = nn.BCELoss()
BATCH_SIZE = 32

############################################### need  to modify #################################################


feature_test_SNR_1 = torch.tensor(df_test_SNR_1.iloc[:,0:-1].to_numpy(dtype=np.float32))
label_test_SNR_1   = torch.tensor(df_test_SNR_1.iloc[:,-1:].to_numpy(dtype=np.float32))
feature_test_SNR_2 = torch.tensor(df_test_SNR_2.iloc[:,0:-1].to_numpy(dtype=np.float32))
label_test_SNR_2   = torch.tensor(df_test_SNR_2.iloc[:,-1:].to_numpy(dtype=np.float32))
feature_test_SNR_3 = torch.tensor(df_test_SNR_3.iloc[:,0:-1].to_numpy(dtype=np.float32))
label_test_SNR_3   = torch.tensor(df_test_SNR_3.iloc[:,-1:].to_numpy(dtype=np.float32))
feature = torch.tensor(df.iloc[:,0:-1].to_numpy(dtype=np.float32))
label = torch.tensor(df.iloc[:,-1:].to_numpy(dtype=np.float32))

dataset_test_SNR_1 = MyDataset(feature_test_SNR_1,label_test_SNR_1)
dataset_test_SNR_2 = MyDataset(feature_test_SNR_2,label_test_SNR_2)
dataset_test_SNR_3 = MyDataset(feature_test_SNR_3,label_test_SNR_3)
dataset = MyDataset(feature,label)

test_loader_SNR_1 = DataLoader(dataset_test_SNR_1,batch_size = BATCH_SIZE,shuffle = False)
test_loader_SNR_2 = DataLoader(dataset_test_SNR_2,batch_size = BATCH_SIZE,shuffle = False)
test_loader_SNR_3 = DataLoader(dataset_test_SNR_3,batch_size = BATCH_SIZE,shuffle = False)
train_loader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle = False)


print('training data')
test.testing_data(model,device,train_loader,criterion)
print('SNR = 1')
test.testing_data(model,device,test_loader_SNR_1,criterion)
print("SNR = 2")
test.testing_data(model,device,test_loader_SNR_2,criterion)
print("SNR = 3")
test.testing_data(model,device,test_loader_SNR_3,criterion)