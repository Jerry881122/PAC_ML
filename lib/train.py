import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt



def training_data(model , device , train_loader , optimizer , criterion , epoch , losses):

    model.train()

    # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for batch_index , (data , label) in enumerate(train_loader):
        # step 1. move the data to our device
        data , label = data.to(device) , label.to(device)
        # step 2. reset gradient value
        optimizer.zero_grad()
        # step 3. doing froward operation and get the result
        output = model(data)
        # step 4 . calculate loss base on loss function
        loss = criterion(output , label) 
        # print("output = ",output)
        # print("label = ",label)
        # print("loss = ",loss) 
        # step 5. doing back propagation and calculate the gradient value 
        loss.backward()
        # step 6. update the weight 
        optimizer.step()

        losses.append(loss.item())

        # print loss every 100 batch
        if batch_index % 100 == 0:    
            data_index = batch_index*len(data)
            total_data = len(train_loader.dataset)
            percentage = 100.*batch_index/len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,data_index,total_data,percentage,loss.item()))

    # params = model.parameters()
    # for param in params:
    #     print(param)

def loss_update_fig(losses,name,show=None):
    plt.figure(1)
    plt.plot(losses)
    plt.title('Loss vs Parameter Updates')
    plt.xlabel('Parameter Updates')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(name)
    if show == 1:
        plt.show()

def learning_rate_updata_fig(learning_rates,name,show=None):
    plt.figure(2)
    plt.plot(learning_rates)
    plt.title('learnging rate vs epoch')
    plt.xlabel('learning rate')
    plt.ylabel('epoch')
    plt.grid(True)
    plt.savefig(name)
    if show == 1:
        plt.show()
