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
from lib.data_preprocess import data_preprocess # data pre-process , normalized , reshape ,etc.
from lib.data_preprocess import normalization
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


# 輸入 model 的名字
name = input("input model name = ")
path_name = "tmp_model/LSTM/" + name + ".pth"
loss_name = "tmp_model/LSTM/loss_" + name + ".jpg"
learning_rate_name = "tmp_model/LSTM/lr_" + name + ".jpg"
csv_name = "tmp_model/LSTM/" + name + ".csv"


# define parameter
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001



# using pandas to read csv file             0 -> correct codeword , 1 -> error codeword 
# df = pd.read_csv('resource/train/SNR2_L8/L=8_frame=100000_SNR=2.csv')
df_test = pd.read_csv('resource/test/SNR2_L8/L=8_frame=25000_SNR=2.csv')
df = pd.read_csv('resource/train/SNR2_L8/bounding/bounding_train_data.csv')     # bounding data
df_test_bounding = pd.read_csv('resource/test/SNR2_L8/bounding/bounding_test_data.csv')  # bounding data


# pandas -> numpy -> tensor , and first 8 data are feature & last data are label (data pre-process)
data_obj = data_preprocess()
feature = data_obj.train_process_LSTM(df,normalized=False)
feature_test = data_obj.test_process_LSTM(df_test,normalized=False)
feature_test_bounding = data_obj.test_process_LSTM(df_test_bounding,normalized=False)

label = torch.tensor(df.iloc[:,-1:].to_numpy(dtype=np.float32))
label_test = torch.tensor(df_test.iloc[:,-1:].to_numpy(dtype=np.float32))
label_test_bounding = torch.tensor(df_test_bounding.iloc[:,-1:].to_numpy(dtype=np.float32))


# using dataset and DataLoader from pytorch to store train data
dataset = MyDataset(feature,label)
dataset_test = MyDataset(feature_test,label_test)
dataset_test_bounding = MyDataset(feature_test_bounding,label_test_bounding)
train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset_test,batch_size = BATCH_SIZE,shuffle = False)
test_loader_bounding = DataLoader(dataset_test_bounding,batch_size = BATCH_SIZE,shuffle = False)


# setting model parameter
input_size = 1
lstm_size = 16
fc_size = 4
output_size = 1

model = Mymodel.LSTM_model(input_size=input_size,lstm_size=lstm_size,fc_size=fc_size,
                            output_size=output_size,device=device).to(device)
print(model)



# # Loss and optimizer
# weight = torch.tensor([5,1])   # [pos_weight,neg_weight]
criterion = nn.BCELoss()        # binary cross entropy
# criterion = weighted_BCE_loss(weight=weight)
# criterion = FocalLoss(weight=weight)
# criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
# optimizer = optim.SGD(model.parameters(),lr=LR)
optimizer = optim.Adam(model.parameters(), lr=LR)


losses = []

TPR = []
FPR = []
accurary = []
Recall = []
Precision = []
F1_score = []


for epoch in tqdm(range(EPOCHS), ncols=50):
    # training data
    train.training_data(model,device,train_loader,optimizer,criterion,epoch,losses)

    # output results after all epochs
    if(epoch == EPOCHS-1):
        # testing the training data
        print('\ntesting training data')
        tpr , fpr , acc , recall , precision , f1_score = test.testing_data(model,device,train_loader,criterion)
        TPR.append(tpr)
        FPR.append(fpr)
        accurary.append(acc)
        Recall.append(recall)
        Precision.append(precision)
        F1_score.append(f1_score)
        # testing the testing data 
        print('\ntesting testing data without bounding')
        tpr , fpr , acc , recall , precision , f1_score = test.testing_data(model,device,test_loader,criterion)
        TPR.append(tpr)
        FPR.append(fpr)
        accurary.append(acc)
        Recall.append(recall)
        Precision.append(precision)
        F1_score.append(f1_score)
        # testing the testing data 
        print('\ntesting testing data with bounding')
        tpr , fpr , acc , recall , precision , f1_score = test.testing_data(model,device,test_loader_bounding,criterion)
        TPR.append(tpr)
        FPR.append(fpr)
        accurary.append(acc)
        Recall.append(recall)
        Precision.append(precision)
        F1_score.append(f1_score)
        print('///////////////////////////////\n')


with open(csv_name , 'w' , newline = '') as Filecsv:
    writer = csv.writer(Filecsv)

    writer.writerow(['EPOCHS'] + [EPOCHS])
    writer.writerow(['BATCHS'] + [BATCH_SIZE])
    writer.writerow(['learning rate'] + [LR])
    writer.writerow([''])
    writer.writerow(['','train_result','test_result','test_bounding'])
    writer.writerow(['TPR'] + TPR)
    writer.writerow(['FPR'] + FPR)
    writer.writerow(['accurary'] + accurary)
    writer.writerow(['Recall'] + Recall)
    writer.writerow(['Precision'] + Precision)
    writer.writerow(['F1_score'] + F1_score)


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