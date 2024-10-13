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
import csv

# my own lib
from lib.MyDataset import MyDataset
from lib.SetCuda import SetCuda
from lib import Mymodel      # model function
from lib import train        # training function
from lib import test         # testing function
from lib.loss import weighted_BCE_loss
from lib.loss import FocalLoss


# begin timer
timer_begin = time.time()

print("\nsimulated environment")
print('\tTorch',torch.__version__,'CUDA',torch.version.cuda)
print('\tcuDNN version: ' + str(torch.backends.cudnn.version()))
print('\tusing CUDA = ' + str(torch.cuda.is_available()))  # 檢查 CUDA 是否可用
print('\n')


# setting device
device = SetCuda()


# define parameter
EPOCHS = 1
BATCH_SIZE = 32
LR = 0.001
print_LR = LR

# weight name
name = "LSTM_ROC"
path_name = "tmp_model/LSTM/" + name + ".pth"
loss_name = "tmp_model/LSTM/loss_" + name + ".jpg"
learning_rate_name = "tmp_model/LSTM/lr_" + name + ".jpg"
csv_name = "tmp_model/LSTM/" + name + ".csv"


# using pandas to read csv file             0 -> correct codeword , 1 -> error codeword 
df = pd.read_csv('resource/train/SNR2_L8/L=8_frame=100000_SNR=2.csv')
df_test = pd.read_csv('resource/test/SNR2_L8/L=8_frame=25000_SNR=2.csv')


# pandas -> numpy -> tensor , and first 8 data are feature & last data are label
feature = torch.tensor(df.iloc[:,0:-1].to_numpy(dtype=np.float32))
feature = feature.view(len(feature) , len(feature[0]) , 1)          # change shape to (10000,8,1)
label = torch.tensor(df.iloc[:,-1:].to_numpy(dtype=np.float32))
feature_test = torch.tensor(df_test.iloc[:,0:-1].to_numpy(dtype=np.float32))
feature_test = feature_test.view(len(feature_test) , len(feature_test[0]) , 1)
label_test = torch.tensor(df_test.iloc[:,-1:].to_numpy(dtype=np.float32))

# using dataset and DataLoader from pytorch to store train data
dataset = MyDataset(feature,label)
dataset_test = MyDataset(feature_test,label_test)
train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset_test,batch_size = BATCH_SIZE,shuffle = False)


# setting model parameter
input_size = 1
hidden_size = 8
num_layer = 1
output_size = 1

model = Mymodel.LSTM_model(input_size=input_size,hidden_size=hidden_size,
                            num_layer=num_layer,output_size=output_size,device=device).to(device)
print(model)



# # Loss and optimizer
criterion = nn.BCELoss()        # binary cross entropy
# criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
# optimizer = optim.SGD(model.parameters(),lr=LR)
optimizer = optim.Adam(model.parameters(), lr=LR)


losses = []

TPR = [0.0 , 0.0]
FPR = [0.0 , 0.0]
accurary = [0.0 , 0.0]

for epoch in tqdm(range(EPOCHS), ncols=50):
    # training data
    train.training_data(model,device,train_loader,optimizer,criterion,epoch,losses)

    # output results after all epochs
    if(epoch == EPOCHS-1):
        # testing the training data
        print('\ntesting training data')
        TPR[0],FPR[0],accurary[0] = test.testing_data(model,device,train_loader,criterion)
        # testing the testing data
        print('\ntesting testing data')
        TPR[1],FPR[1],accurary[1] = test.testing_data(model,device,test_loader,criterion)
        print('///////////////////////////////\n')


with open(csv_name , 'w' , newline = '') as Filecsv:
    writer = csv.writer(Filecsv)

    writer.writerow(['EPOCHS'] + [EPOCHS])
    writer.writerow(['BATCHS'] + [BATCH_SIZE])
    writer.writerow(['learning rate'] + [LR])
    writer.writerow([''])
    writer.writerow(['','train_result','test_result'])
    writer.writerow(['TPR'] + TPR)
    writer.writerow(['FPR'] + FPR)
    writer.writerow(['accurary'] + accurary)


train.loss_update_fig(losses,loss_name)



# print out model weight
# params = model.parameters()
# for param in params:
#     print(param)

# save the model weight 
torch.save(model.state_dict() , path_name)

# execution time ( represented as hour , minute , second)
timer_end = time.time()
hours, rem = divmod(timer_end-timer_begin, 3600)
minutes, seconds = divmod(rem, 60)
print("Execution Time: {:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds))