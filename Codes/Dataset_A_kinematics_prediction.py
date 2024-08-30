import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy
import statistics 
from numpy import loadtxt
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statistics import stdev 
import math
import h5py
 
import numpy as np
import time

from scipy.signal import butter,filtfilt
import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
import pandas
import matplotlib.pyplot as plt

# from tsf.model import TransformerForecaster


# from tensorflow.keras.utils import np_utils
import itertools
###  Library for attention layers 
import pandas as pd
import os 
import numpy as np
#from tqdm import tqdm # Processing time measurement
from sklearn.model_selection import train_test_split 

import statistics
import gc
import torch.nn.init as init

############################################################################################################################################################################
############################################################################################################################################################################

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.utils.weight_norm as weight_norm


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.parameter import Parameter


import torch.optim as optim


from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler




# Data Loader

def data_loader(subject):
  with h5py.File('/home/sanzidpr/HICCS_submission/All_subjects_data_kinetics.h5', 'r') as hf:
    All_subjects = hf['All_subjects']
    Subject = All_subjects[subject]

    HOF=Subject['hof']
    IMU_KIN=Subject['IMU_Kin']

    treadmill_hof = HOF['Treadmill']
    levelground_hof = HOF['Levelground']
    slope_hof = HOF['Slope']
    stair_hof = HOF['Stair']
    round_hof = HOF['Round']
    obstacles_hof = HOF['Obstacles']

    treadmill_IMU_kin = IMU_KIN['Treadmill']
    levelground_IMU_kin = IMU_KIN['Levelground']
    slope_IMU_kin = IMU_KIN['Slope']
    stair_IMU_kin = IMU_KIN['Stair']
    round_IMU_kin= IMU_KIN['Round']
    obstacles_IMU_kin = IMU_KIN['Obstacles']

    
    hof_data=np.concatenate((treadmill_hof,levelground_hof,slope_hof,stair_hof,round_hof,obstacles_hof),axis=0)
    IMU_kin_data=np.concatenate((treadmill_IMU_kin,levelground_IMU_kin,slope_IMU_kin,stair_IMU_kin,round_IMU_kin,obstacles_IMU_kin),axis=0)

    return np.array(hof_data), np.array(IMU_kin_data)


subject_1_data_hof, subject_1_data_IMU_Kin=data_loader('Subject_1')
gc.collect()
subject_2_data_hof, subject_2_data_IMU_Kin=data_loader('Subject_2')
gc.collect()
subject_3_data_hof, subject_3_data_IMU_Kin=data_loader('Subject_3')
gc.collect()
subject_4_data_hof, subject_4_data_IMU_Kin=data_loader('Subject_4')
gc.collect()
subject_5_data_hof, subject_5_data_IMU_Kin=data_loader('Subject_5')
gc.collect()
subject_6_data_hof, subject_6_data_IMU_Kin=data_loader('Subject_6')
gc.collect()
subject_7_data_hof, subject_7_data_IMU_Kin=data_loader('Subject_7')
gc.collect()
subject_8_data_hof, subject_8_data_IMU_Kin=data_loader('Subject_8')
gc.collect()
subject_9_data_hof, subject_9_data_IMU_Kin=data_loader('Subject_9')
gc.collect()
subject_10_data_hof, subject_10_data_IMU_Kin=data_loader('Subject_10')
gc.collect()



#################################################################################################################################################################
#Subject Selection



main_dir = "/home/sanzidpr/HICCS_submission/Dataset A/Subject01"
#os.mkdir(main_dir) 
subject='Subject_01'
path='/home/sanzidpr/HICCS_submission/Dataset A/Subject01/'

train_data_hof=np.concatenate((subject_2_data_hof,subject_3_data_hof,subject_4_data_hof,subject_5_data_hof,subject_6_data_hof,
                               subject_7_data_hof,subject_8_data_hof,subject_9_data_hof,subject_10_data_hof),axis=0)

train_data_IMU_Kin=np.concatenate((subject_2_data_IMU_Kin,subject_3_data_IMU_Kin,subject_4_data_IMU_Kin,subject_5_data_IMU_Kin,subject_6_data_IMU_Kin,
                               subject_7_data_IMU_Kin,subject_8_data_IMU_Kin,subject_9_data_IMU_Kin,subject_10_data_IMU_Kin),axis=0)
                               

test_data_hof=subject_1_data_hof
test_data_IMU_Kin=subject_1_data_IMU_Kin



#################################################################################################################################################################



# Data processing and preparation

train_dataset_IMU=train_data_IMU_Kin[:,0:48]
train_dataset_hof=train_data_hof
train_dataset_target=np.concatenate((train_data_IMU_Kin[:,55:56],train_data_IMU_Kin[:,58:60],train_data_IMU_Kin[:,62:63],train_data_IMU_Kin[:,65:67]),axis=1)


test_dataset_IMU=test_data_IMU_Kin[:,0:48]
test_dataset_hof=test_data_hof
test_dataset_target=np.concatenate((test_data_IMU_Kin[:,55:56],test_data_IMU_Kin[:,58:60],test_data_IMU_Kin[:,62:63],test_data_IMU_Kin[:,65:67]),axis=1)

print(train_dataset_IMU.shape)
print(train_dataset_hof.shape)
print(train_dataset_target.shape)


gc.collect()
gc.collect()
gc.collect()


# # convert an array of values into a dataset matrix
def create_dataset_IMU(dataset_1, window=45):
  dataX= []
  k=0
  shift=9
  for i in range(int(len(dataset_1)/shift-8)):
    j=shift*k
    a = dataset_1[j:j+window,:]
    # print(a.shape)
    dataX.append(a)
    k=k+1
  return np.array(dataX)

# # convert an array of values into a dataset matrix
def create_dataset_hof(dataset_1, window=15):
  dataX= []
  k=0
  shift=3
  for i in range(int(len(dataset_1)/shift-8)):
    j=shift*k
    a = dataset_1[j:j+window,:]
    # print(a.shape)
    dataX.append(a)
    k=k+1
  return np.array(dataX)

# # convert an array of values into a dataset matrix
def create_dataset_Kinematics(dataset_1, window=45):
  dataX= []
  k=0
  shift=9
  for i in range(int(len(dataset_1)/shift-8)):
      j=shift*k
      a = dataset_1[j+window:j+window+shift,:]
      dataX.append(a)
      k=k+1
  return np.array(dataX)

window=45
window_hof=15

train_IMU=create_dataset_IMU(train_dataset_IMU)
train_hof=create_dataset_hof(train_dataset_hof)
train_target_future=create_dataset_Kinematics(train_dataset_target)
train_target_present=create_dataset_IMU(train_dataset_target)

gc.collect()
gc.collect()
gc.collect()
gc.collect()

test_IMU=create_dataset_IMU(test_dataset_IMU)
test_hof=create_dataset_hof(test_dataset_hof)
test_target_future=create_dataset_Kinematics(test_dataset_target)
test_target_present=create_dataset_IMU(test_dataset_target)


gc.collect()
gc.collect()
gc.collect()
gc.collect()

print(train_IMU.shape)
print(train_hof.shape)
print(train_target_present.shape)
print(train_target_future.shape)

train_X_IMU, X_validation_IMU,train_X_hof, X_validation_hof, train_y_5_present, Y_validation_present , train_y_5_future, Y_validation_future=train_test_split(train_IMU,train_hof,train_target_present, train_target_future,test_size=0.20, random_state=True)


### Data Processing

batch_size = 64

## all Modality Features

train_features = torch.Tensor(train_X_IMU)
train_features_hof = torch.Tensor(train_X_hof)
train_targets_present = torch.Tensor(train_y_5_present)
train_targets_future= torch.Tensor(train_y_5_future)


val_features = torch.Tensor(X_validation_IMU)
val_features_hof = torch.Tensor(X_validation_hof)
val_targets_present = torch.Tensor(Y_validation_present)
val_targets_future = torch.Tensor(Y_validation_future)


test_features = torch.Tensor(test_IMU)
test_features_hof = torch.Tensor(test_hof)
test_targets_present = torch.Tensor(test_target_present)
test_targets_future = torch.Tensor(test_target_future)


train_features_acc_8=torch.cat((train_features[:,:,0:3],train_features[:,:,6:9],train_features[:,:,12:15],train_features[:,:,18:21],train_features[:,:,24:27]\
                             ,train_features[:,:,30:33],train_features[:,:,36:39],train_features[:,:,42:45]),axis=-1)
test_features_acc_8=torch.cat((test_features[:,:,0:3],test_features[:,:,6:9],test_features[:,:,12:15],test_features[:,:,18:21],test_features[:,:,24:27]\
                             ,test_features[:,:,30:33],test_features[:,:,36:39],test_features[:,:,42:45]),axis=-1)
val_features_acc_8=torch.cat((val_features[:,:,0:3],val_features[:,:,6:9],val_features[:,:,12:15],val_features[:,:,18:21],val_features[:,:,24:27]\
                             ,val_features[:,:,30:33],val_features[:,:,36:39],val_features[:,:,42:45]),axis=-1)


train_features_gyr_8=torch.cat((train_features[:,:,3:6],train_features[:,:,9:12],train_features[:,:,15:18],train_features[:,:,21:24],train_features[:,:,27:30]\
                             ,train_features[:,:,33:36],train_features[:,:,39:42],train_features[:,:,45:48]),axis=-1)
test_features_gyr_8=torch.cat((test_features[:,:,3:6],test_features[:,:,9:12],test_features[:,:,15:18],test_features[:,:,21:24],test_features[:,:,27:30]\
                             ,test_features[:,:,33:36],test_features[:,:,39:42],test_features[:,:,45:48]),axis=-1)
val_features_gyr_8=torch.cat((val_features[:,:,3:6],val_features[:,:,9:12],val_features[:,:,15:18],val_features[:,:,21:24],val_features[:,:,27:30]\
                             ,val_features[:,:,33:36],val_features[:,:,39:42],val_features[:,:,45:48]),axis=-1)



train = TensorDataset(train_features, train_features_acc_8,train_features_gyr_8,train_features_hof, train_targets_present, train_targets_future)
val = TensorDataset(val_features, val_features_acc_8, val_features_gyr_8,val_features_hof,val_targets_present, val_targets_future)
test = TensorDataset(test_features, test_features_acc_8, test_features_gyr_8,test_features_hof, test_targets_present, test_targets_future)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)


gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()

# Important Function

import scipy.stats as stats

def RMSE_prediction(yhat_4,test_y):

  s1=yhat_4.shape[0]*yhat_4.shape[1]
 
  test_o=test_y.reshape((s1,6))
  yhat=yhat_4.reshape((s1,6))
  
  
  y_1_no=yhat[:,0]
  y_2_no=yhat[:,1]
  y_3_no=yhat[:,2]
  y_4_no=yhat[:,3]
  y_5_no=yhat[:,4]
  y_6_no=yhat[:,5]
  
  
  y_1=y_1_no
  y_2=y_2_no
  y_3=y_3_no
  y_4=y_4_no
  y_5=y_5_no
  y_6=y_6_no

  
  y_test_1=test_o[:,0]
  y_test_2=test_o[:,1]
  y_test_3=test_o[:,2]
  y_test_4=test_o[:,3]
  y_test_5=test_o[:,4]
  y_test_6=test_o[:,5]

  
  ###calculate RMSE
  
  rmse_1 =np.sqrt(mean_squared_error(y_test_1,y_1))
  rmse_2 =np.sqrt(mean_squared_error(y_test_2,y_2))
  rmse_3 =np.sqrt(mean_squared_error(y_test_3,y_3))
  rmse_4 =np.sqrt(mean_squared_error(y_test_4,y_4))
  rmse_5 =np.sqrt(mean_squared_error(y_test_5,y_5))
  rmse_6 =np.sqrt(mean_squared_error(y_test_6,y_6))

  
  print(rmse_1)
  print(rmse_2)
  print(rmse_3)
  print(rmse_4)
  print(rmse_5)
  print(rmse_6)

  
  p_1=np.corrcoef(y_1, y_test_1)[0, 1]
  p_2=np.corrcoef(y_2, y_test_2)[0, 1]
  p_3=np.corrcoef(y_3, y_test_3)[0, 1]
  p_4=np.corrcoef(y_4, y_test_4)[0, 1]
  p_5=np.corrcoef(y_5, y_test_5)[0, 1]
  p_6=np.corrcoef(y_6, y_test_6)[0, 1]

  
  
  print("\n") 
  print(p_1)
  print(p_2)
  print(p_3)
  print(p_4)
  print(p_5)
  print(p_6)

  
  
              ### Correlation ###
  p=np.array([(p_1+p_4)/2,(p_2+p_5)/2,(p_3+p_6)/2])  
  
  
      #### Mean and standard deviation ####
  
  rmse=np.array([(rmse_1+rmse_4)/2,(rmse_2+rmse_5)/2,(rmse_3+rmse_6)/2])
  
      #### Mean and standard deviation ####
  m=statistics.mean(rmse)
  SD=statistics.stdev(rmse)
  print('Mean: %.3f' % m,'+/- %.3f' %SD)
   
  m_c=statistics.mean(p)
  SD_c=statistics.stdev(p)
  print('Mean: %.3f' % m_c,'+/- %.3f' %SD_c)



  return rmse, p


# Define custom RMSE loss function
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        mse_loss = torch.nn.MSELoss()(pred, target)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch.nn as nn
import torch

class PearsonCorrCoefLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrCoefLoss, self).__init__()

    def forward(self, y_pred, y_true):
        x = y_pred - torch.mean(y_pred)
        y = y_true - torch.mean(y_true)
        loss = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
        return 1- loss


# Model with only prediction

# Model Training-- 8 IMUs+ Hof


## Training Function

def train_mm_m(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    # criterion =nn.MSELoss()
    criterion =RMSELoss()
    # criterion =PearsonCorrCoefLoss()


    # criterion=PearsonCorrLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    optimizer = torch.optim.Adam(model.parameters())

    
    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10
    

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(train_loader):
            optimizer.zero_grad()
            output= model(data_acc.to(device).float(),data_gyr.to(device).float())
            
            loss = criterion(output, target_future.to(device).float())
            loss.backward()
            optimizer.step()
          
            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)       
            
       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_hof, target_present, target_future in val_loader:
                output= model(data_acc.to(device).float(),data_gyr.to(device).float())
                val_loss += criterion(output, target_future.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break


            
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

  
    # # Save the trained model
    # torch.save(model.state_dict(), "model.pth")

    return model
    
    


## 1. LSTM

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=False, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(128, 64, bidirectional=False, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, (h_n, c_n) = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, (h_n, c_n)

class MM_LSTM(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_LSTM, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.fc = nn.Linear(3*64, 6*9)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc,(h_acc,c_acc)=self.encoder_acc(x_acc_2)
        x_gyr,(h_gyr,c_gyr)=self.encoder_gyr(x_gyr_2)
        x_hof,(h_hof,c_hof)=self.encoder_hof(x_hof_2)


        x=torch.cat((x_acc[:,-1,:],x_gyr[:,-1,:],x_hof[:,-1,:]),dim=-1).squeeze(0)

        out=self.fc(x).view(x.shape[0],9,6)


        return out
        

lr = 0.001
model = MM_LSTM(24,24,36)

mm_lstm = train_mm_m(train_loader, lr,15,model,path+subject+'_mm_lstm_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_lstm= MM_LSTM(24,24,36)
mm_lstm.load_state_dict(torch.load(path+subject+'_mm_lstm_IMU8_hof.pth'))
mm_lstm.to(device)

mm_lstm.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_lstm(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_1=np.hstack([rmse,p])

## 2. GRU

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=False, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(128, 64, bidirectional=False, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, h_n = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, h_n

class MM_GRU(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_GRU, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.fc = nn.Linear(3*64, 6*9)

        self.flatten=nn.Flatten()

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc,h_acc=self.encoder_acc(x_acc_2)
        x_gyr,h_gyr=self.encoder_gyr(x_gyr_2)
        x_hof,h_hof=self.encoder_hof(x_hof_2)

        x=torch.cat((x_acc[:,-1,:],x_gyr[:,-1,:],x_hof[:,-1,:]),dim=-1)
       
        out=self.fc(x).view(x.shape[0],9,6)


        return out

lr = 0.001
model = MM_GRU(24,24,36)

mm_gru = train_mm_m(train_loader, lr,15,model,path+subject+'_mm_gru_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_gru= MM_GRU(24,24,36)
mm_gru.load_state_dict(torch.load(path+subject+'_mm_gru_IMU8_hof.pth'))
mm_gru.to(device)

mm_gru.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_gru(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_2=np.hstack([rmse,p])

## 3. Bi-LSTM

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, (h_n, c_n) = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, (h_n, c_n)

class MM_Bi_LSTM(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_Bi_LSTM, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.fc = nn.Linear(3*128, 6*9)

        self.flatten=nn.Flatten()

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc,(h_acc,c_acc)=self.encoder_acc(x_acc_2)
        x_gyr,(h_gyr,c_gyr)=self.encoder_gyr(x_gyr_2)
        x_hof,(h_hof,c_hof)=self.encoder_hof(x_hof_2)

        x=torch.cat((x_acc[:,-1,:],x_gyr[:,-1,:],x_hof[:,-1,:]),dim=-1)
        
        out=self.fc(x).view(x.shape[0],9,6)


        return out

lr = 0.001
model = MM_Bi_LSTM(24,24,36)

mm_bi_lstm = train_mm_m(train_loader, lr,12,model,path+subject+'_mm_bi_lstm_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_bi_lstm= MM_Bi_LSTM(24,24,36)
mm_bi_lstm.load_state_dict(torch.load(path+subject+'_mm_bi_lstm_IMU8_hof.pth'))
mm_bi_lstm.to(device)

mm_bi_lstm.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_bi_lstm(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_3=np.hstack([rmse,p])

## 4. Bi-GRU

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, h_n = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, h_n

class MM_Bi_GRU(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_Bi_GRU, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.fc = nn.Linear(3*128, 6*9)

        self.flatten=nn.Flatten()

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc,h_acc=self.encoder_acc(x_acc_2)
        x_gyr,h_gyr=self.encoder_gyr(x_gyr_2)
        x_hof,h_hof=self.encoder_hof(x_hof_2)

        x=torch.cat((x_acc[:,-1,:],x_gyr[:,-1,:],x_hof[:,-1,:]),dim=-1)
       
        out=self.fc(x).view(x.shape[0],9,6)


        return out

lr = 0.001
model = MM_Bi_GRU(24,24,36)

mm_bi_gru = train_mm_m(train_loader, lr,10,model,path+subject+'_mm_bi_gru_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_bi_gru= MM_Bi_GRU(24,24,36)
mm_bi_gru.load_state_dict(torch.load(path+subject+'_mm_bi_gru_IMU8_hof.pth'))
mm_bi_gru.to(device)

mm_bi_gru.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_bi_gru(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_4=np.hstack([rmse,p])

## 5. LSTM --Encoder+decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=False, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(128, 64, bidirectional=False, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, (h_n, c_n) = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, (h_n, c_n)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, encoder_hidden, encoder_cell,  max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        input = torch.zeros(batch_size,1,6).to(device)

        
        for i in range(max_len):
            # Run one time step of LSTM
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
            
            # Use the predicted output as the next input
            input = output.unsqueeze(1)

            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

class MM_ED_LSTM(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_ED_LSTM, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.decoder=LSTMDecoder(6, 3*64, 6, 1, 0.0)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc,(h_acc,c_acc)=self.encoder_acc(x_acc_2)
        x_gyr,(h_gyr,c_gyr)=self.encoder_gyr(x_gyr_2)
        x_hof,(h_hof,c_hof)=self.encoder_hof(x_hof_2)


        h=torch.cat((h_acc,h_gyr,h_hof),dim=-1)
        c=torch.cat((c_acc,c_gyr,c_hof),dim=-1)
        
        out=self.decoder(h, c, 9)


        return out

lr = 0.001
model = MM_ED_LSTM(24,24,36)

mm_ed_lstm = train_mm_m(train_loader, lr,10,model,path+subject+'_mm_ed_lstm_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_lstm= MM_ED_LSTM(24,24,36)
mm_ed_lstm.load_state_dict(torch.load(path+subject+'_mm_ed_lstm_IMU8_hof.pth'))
mm_ed_lstm.to(device)

mm_ed_lstm.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_ed_lstm(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_5=np.hstack([rmse,p])

## 6. GRU --Encoder+decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=False, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(128, 64, bidirectional=False, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout=nn.Dropout(dropout)
        
        
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, h_n = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, h_n

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.dropout=nn.Dropout(dropout)

        
    def forward(self, encoder_hidden, max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        # cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        
        input = torch.zeros(batch_size,1,6).to(device)

        
        for i in range(max_len):


            # Run one time step of LSTM
            output, hidden = self.lstm(input, hidden)
            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
            
            # Use the predicted output as the next input
            input = output.unsqueeze(1)

            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
        

class MM_ED_GRU(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_ED_GRU, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.decoder=LSTMDecoder(6, 3*64, 6, 1, 0.0)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc, h_acc=self.encoder_acc(x_acc_2)
        x_gyr, h_gyr=self.encoder_gyr(x_gyr_2)
        x_hof, h_hof=self.encoder_hof(x_hof_2)


        h=torch.cat((h_acc,h_gyr,h_hof),dim=-1)

        out=self.decoder(h,9)

        return out

lr = 0.001
model = MM_ED_GRU(24,24,36)

mm_ed_gru= train_mm_m(train_loader, lr,10,model,path+subject+'_mm_ed_gru_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_gru= MM_ED_GRU(24,24,36)
mm_ed_gru.load_state_dict(torch.load(path+subject+'_mm_ed_gru_IMU8_hof.pth'))
mm_ed_gru.to(device)

mm_ed_gru.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_ed_gru(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_6=np.hstack([rmse,p])

## 7. Bi-LSTM --Encoder+decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, (h_n, c_n) = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, (h_n, c_n)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, encoder_hidden, encoder_cell,  max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        input = torch.zeros(batch_size,1,6).to(device)

        
        for i in range(max_len):
            # Run one time step of LSTM
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
            
            # Use the predicted output as the next input
            input = output.unsqueeze(1)

            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

class MM_ED_Bi_LSTM(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_ED_Bi_LSTM, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.decoder=LSTMDecoder(6, 3*64, 6, 1, 0.05)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc,(h_acc,c_acc)=self.encoder_acc(x_acc_2)
        x_gyr,(h_gyr,c_gyr)=self.encoder_gyr(x_gyr_2)
        x_hof,(h_hof,c_hof)=self.encoder_hof(x_hof_2)

        h=torch.cat((h_acc,h_gyr,h_hof),dim=-1)
        c=torch.cat((c_acc,c_gyr,c_hof),dim=-1)
        
        out=self.decoder(h, c, 9)


        return out

lr = 0.001
model = MM_ED_Bi_LSTM(24,24,36)

mm_ed_bi_lstm = train_mm_m(train_loader, lr,12,model,path+subject+'_mm_ed_bi_lstm_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_bi_lstm= MM_ED_Bi_LSTM(24,24,36)
mm_ed_bi_lstm.load_state_dict(torch.load(path+subject+'_mm_ed_bi_lstm_IMU8_hof.pth'))
mm_ed_bi_lstm.to(device)

mm_ed_bi_lstm.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_ed_bi_lstm(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_7=np.hstack([rmse,p])

## 8. Bi-GRU --Encoder+decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout=nn.Dropout(dropout)
        
        
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, h_n = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, h_n

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, output_size)

        
        self.dropout=nn.Dropout(dropout)
   
    def forward(self, encoder_hidden, max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        # cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        
        input = torch.zeros(batch_size,1,6).to(device)
        
        for i in range(max_len):

            # Run one time step of LSTM
            output, hidden = self.lstm(input, hidden)
            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
            
            # Use the predicted output as the next input
            input = output.unsqueeze(1)

            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
        

class MM_ED_Bi_GRU(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_ED_Bi_GRU, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.decoder=LSTMDecoder(6, 3*64, 6, 1, 0.05)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc, h_acc=self.encoder_acc(x_acc_2)
        x_gyr, h_gyr=self.encoder_gyr(x_gyr_2)
        x_hof, h_hof=self.encoder_hof(x_hof_2)


        h=torch.cat((h_acc,h_gyr,h_hof),dim=-1)

        out=self.decoder(h,9)

        return out

lr = 0.001
model = MM_ED_Bi_GRU(24,24,36)

mm_ed_Bi_GRU = train_mm_m(train_loader, lr,15,model,path+subject+'_mm_bi_ed_gru_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_Bi_GRU= MM_ED_Bi_GRU(24,24,36)
mm_ed_Bi_GRU.load_state_dict(torch.load(path+subject+'_mm_bi_ed_gru_IMU8_hof.pth'))
mm_ed_Bi_GRU.to(device)

mm_ed_Bi_GRU.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_ed_Bi_GRU(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_8=np.hstack([rmse,p])

# 9. Attention Without gating+ Bi-GRU --Encoder+decoder
# 9. Attention Without gating+ Bi-GRU --Encoder+decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout=nn.Dropout(dropout)
        
        
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, h_n = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, h_n

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.gating_net = nn.Sequential(
            nn.Linear(1542, 1542),
            nn.Sigmoid())
        
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, encoder_hidden, h_att, max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        # cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        # input = target_estimation[:, -1, :].unsqueeze(1)
        
        input_1 = torch.zeros(batch_size,1,6).to(device)
        input = torch.cat((input_1, h_att), dim=-1)

        # print(input.shape)

        
        for i in range(max_len):

            # gating_weight=self.gating_net(input)
            # input=input*gating_weight

            # Run one time step of LSTM
            output, hidden = self.lstm(input, hidden)
            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
          

            input = torch.cat((output.unsqueeze(1), input[:,:,6:390]), dim=-1)


            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
        

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()

        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        # Calculate attention weights.
        attn = self.V(torch.tanh(self.W(x)))
        attn = attn.squeeze(-1)
        attn = torch.softmax(attn, dim=1)

        # Calculate weighted average of hidden states.
        context = attn.unsqueeze(-1) * x
        context = context.sum(dim=1)

        return context

class MM_ED_Bi_GRU_attention_WFW(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_ED_Bi_GRU_attention_WFW, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2*window*128+window_hof*128, 54)
        # self.fc1 = nn.Linear(3*128, 54)

        self.attention_acc=nn.MultiheadAttention(24,4,batch_first=True)
        self.attention_gyr=nn.MultiheadAttention(24,4,batch_first=True)
        self.attention_hof=nn.MultiheadAttention(36,4,batch_first=True)

        self.temporal_attn_acc = TemporalAttention(128)
        self.temporal_attn_gyr = TemporalAttention(128)
        self.temporal_attn_hof = TemporalAttention(128)

        self.decoder=LSTMDecoder(390, 3*64, 6, 1,0.05)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc, h_acc=self.encoder_acc(x_acc_2)
        x_gyr, h_gyr=self.encoder_gyr(x_gyr_2)
        x_hof, h_hof=self.encoder_hof(x_hof_2)

        x_hof=x_hof.transpose(1,2)
        x_hof= F.interpolate(x_hof, size=(45))
        x_hof=x_hof.transpose(1,2)

        x=torch.cat((x_acc,x_gyr,x_hof),dim=-1)
        h=torch.cat((h_acc,h_gyr,h_hof),dim=-1)

        h_acc_att=self.temporal_attn_acc(x_acc)
        h_gyr_att=self.temporal_attn_gyr(x_gyr)
        h_hof_att=self.temporal_attn_hof(x_hof)

        h_att=torch.cat((h_acc_att,h_gyr_att,h_hof_att),dim=-1)

        #Do it separately
        h_att=h_att.unsqueeze(0)
        h_att=h_att.transpose(1,0)
        out=self.decoder(h,h_att, 9)


        return out

lr = 0.001
model= MM_ED_Bi_GRU_attention_WFW(24,24,36)

mm_ed_bi_gru_attention_wfw = train_mm_m(train_loader, lr,15,model,path+subject+'_mm_ed_bi_gru_attention_wfw_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_bi_gru_attention_wfw= MM_ED_Bi_GRU_attention_WFW(24,24,36)
mm_ed_bi_gru_attention_wfw.load_state_dict(torch.load(path+subject+'_mm_ed_bi_gru_attention_wfw_IMU8_hof.pth'))
mm_ed_bi_gru_attention_wfw.to(device)

mm_ed_bi_gru_attention_wfw.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_ed_bi_gru_attention_wfw(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_9=np.hstack([rmse,p])

## 10. Attention With gating+ Bi-GRU --Encoder+decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
        
        
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, h_n = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, h_n

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.gating_net = nn.Sequential(
            nn.Linear(390, 390),
            nn.Sigmoid())
        
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, encoder_hidden, h_att, max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        # cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        
        input_1 = torch.zeros(batch_size,1,6).to(device)
        input = torch.cat((input_1, h_att), dim=-1)


        for i in range(max_len):

            gating_weight=self.gating_net(input)
            input=input*gating_weight

            # Run one time step of LSTM
            output, hidden = self.lstm(input, hidden)
            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
          

            input = torch.cat((output.unsqueeze(1), input[:,:,6:390]), dim=-1)


            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
        

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()

        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        # Calculate attention weights.
        attn = self.V(torch.tanh(self.W(x)))
        attn = attn.squeeze(-1)
        attn = torch.softmax(attn, dim=1)

        # Calculate weighted average of hidden states.
        context = attn.unsqueeze(-1) * x
        context = context.sum(dim=1)

        return context

class MM_ED_Bi_GRU_attention_FW(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.0):
        super(MM_ED_Bi_GRU_attention_FW, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)


        self.temporal_attn_acc = TemporalAttention(128)
        self.temporal_attn_gyr = TemporalAttention(128)
        self.temporal_attn_hof = TemporalAttention(128)


        self.decoder=LSTMDecoder(390, 3*64, 6, 1, 0.05)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc, h_acc=self.encoder_acc(x_acc_2)
        x_gyr, h_gyr=self.encoder_gyr(x_gyr_2)
        x_hof, h_hof=self.encoder_hof(x_hof_2)

        x_hof=x_hof.transpose(1,2)
        x_hof= F.interpolate(x_hof, size=(45))
        x_hof=x_hof.transpose(1,2)


        x=torch.cat((x_acc,x_gyr,x_hof),dim=-1)
        h=torch.cat((h_acc,h_gyr,h_hof),dim=-1)


        h_acc_att=self.temporal_attn_acc(x_acc)
        h_gyr_att=self.temporal_attn_gyr(x_gyr)
        h_hof_att=self.temporal_attn_hof(x_hof)


        h_att=torch.cat((h_acc_att,h_gyr_att,h_hof_att),dim=-1)

        #Do it separately

        h_att=h_att.unsqueeze(0)
        h_att=h_att.transpose(1,0)
        out=self.decoder(h,h_att, 9)

        return out

lr = 0.001
model= MM_ED_Bi_GRU_attention_FW(24,24,36)

mm_ed_bi_gru_attention_fw = train_mm_m(train_loader, lr,15,model,path+subject+'_mm_ed_bi_gru_attention_fw_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_bi_gru_attention_fw= MM_ED_Bi_GRU_attention_FW(24,24,36)
mm_ed_bi_gru_attention_fw.load_state_dict(torch.load(path+subject+'_mm_ed_bi_gru_attention_fw_IMU8_hof.pth'))
mm_ed_bi_gru_attention_fw.to(device)

mm_ed_bi_gru_attention_fw.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):
        output = mm_ed_bi_gru_attention_fw(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_10=np.hstack([rmse,p])





# 11. Attention Without gating+ Bi-LSTM --Encoder+decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, (h_n, c_n) = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, (h_n, c_n)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.dropout=nn.Dropout(dropout)

        self.gating_net = nn.Sequential(nn.Linear(390, 390),nn.Sigmoid())
        
        
    def forward(self, encoder_hidden, encoder_cell, h_att,  max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        input_1 = torch.zeros(batch_size,1,6).to(device)
        input = torch.cat((input_1, h_att), dim=-1)


        
        for i in range(max_len):

            # gating_weight=self.gating_net(input)
            # input=input*gating_weight

            # Run one time step of LSTM
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
            
            # Use the predicted output as the next input
            input = torch.cat(( output.unsqueeze(1), input[:,:,6:390]), dim=-1)


            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()

        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        # Calculate attention weights.
        attn = self.V(torch.tanh(self.W(x)))
        attn = attn.squeeze(-1)
        attn = torch.softmax(attn, dim=1)

        # Calculate weighted average of hidden states.
        context = attn.unsqueeze(-1) * x
        context = context.sum(dim=1)

        return context

class MM_ED_Bi_LSTM_WFW(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.05):
        super(MM_ED_Bi_LSTM_WFW, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.temporal_attn_acc = TemporalAttention(128)
        self.temporal_attn_gyr = TemporalAttention(128)
        self.temporal_attn_hof = TemporalAttention(128)


        self.decoder=LSTMDecoder(3*128+6, 3*64, 6, 1, 0.05)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc,(h_acc,c_acc)=self.encoder_acc(x_acc_2)
        x_gyr,(h_gyr,c_gyr)=self.encoder_gyr(x_gyr_2)
        x_hof,(h_hof,c_hof)=self.encoder_hof(x_hof_2)

        h=torch.cat((h_acc,h_gyr,h_hof),dim=-1)
        c=torch.cat((c_acc,c_gyr,c_hof),dim=-1)

        h_acc_att=self.temporal_attn_acc(x_acc)
        h_gyr_att=self.temporal_attn_gyr(x_gyr)
        h_hof_att=self.temporal_attn_hof(x_hof)


        h_att=torch.cat((h_acc_att,h_gyr_att,h_hof_att),dim=-1)

        #Do it separately

        h_att=h_att.unsqueeze(0)
        h_att=h_att.transpose(1,0)
        
        out=self.decoder(h, c, h_att, 9)


        return out

lr = 0.001
model = MM_ED_Bi_LSTM_WFW(24,24,36)

mm_ed_bi_lstm_WFW = train_mm_m(train_loader, lr,12,model,path+subject+'_mm_ed_bi_lstm_WFW_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_bi_lstm_WFW= MM_ED_Bi_LSTM_WFW(24,24,36)
mm_ed_bi_lstm_WFW.load_state_dict(torch.load(path+subject+'_mm_ed_bi_lstm_WFW_IMU8_hof.pth'))
mm_ed_bi_lstm_WFW.to(device)

mm_ed_bi_lstm_WFW.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):                          
        output = mm_ed_bi_lstm_WFW(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_11=np.hstack([rmse,p])

## 12. Attention With gating+ Bi-LSTM --Encoder+decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, (h_n, c_n) = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, (h_n, c_n)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.dropout=nn.Dropout(dropout)

        self.gating_net = nn.Sequential(nn.Linear(390, 390),nn.Sigmoid())
        
        
    def forward(self, encoder_hidden, encoder_cell, h_att,  max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        input_1 = torch.zeros(batch_size,1,6).to(device)
        input = torch.cat((input_1, h_att), dim=-1)


        
        for i in range(max_len):

            gating_weight=self.gating_net(input)
            input=input*gating_weight

            # Run one time step of LSTM
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
            
            # Use the predicted output as the next input
            input = torch.cat(( output.unsqueeze(1), input[:,:,6:390]), dim=-1)


            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()

        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        # Calculate attention weights.
        attn = self.V(torch.tanh(self.W(x)))
        attn = attn.squeeze(-1)
        attn = torch.softmax(attn, dim=1)

        # Calculate weighted average of hidden states.
        context = attn.unsqueeze(-1) * x
        context = context.sum(dim=1)

        return context

class MM_ED_Bi_LSTM_WF(nn.Module):
    def __init__(self, input_acc, input_gyr,input_hof, drop_prob=0.05):
        super(MM_ED_Bi_LSTM_WF, self).__init__()   

        self.encoder_acc=Encoder(input_acc, drop_prob)   
        self.encoder_gyr=Encoder(input_gyr, drop_prob) 
        self.encoder_hof=Encoder(input_hof, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_hof= nn.BatchNorm1d(input_hof, affine=False)

        self.temporal_attn_acc = TemporalAttention(128)
        self.temporal_attn_gyr = TemporalAttention(128)
        self.temporal_attn_hof = TemporalAttention(128)


        self.decoder=LSTMDecoder(3*128+6, 3*64, 6, 1, 0.05)

    def forward(self, x_acc, x_gyr, x_hof):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_hof_1=x_hof.view(x_hof.size(0)*x_hof.size(1),x_hof.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_hof_1=self.BN_hof(x_hof_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
        x_hof_2=x_hof_1.view(-1, window_hof, x_hof_1.size(-1))

        x_acc,(h_acc,c_acc)=self.encoder_acc(x_acc_2)
        x_gyr,(h_gyr,c_gyr)=self.encoder_gyr(x_gyr_2)
        x_hof,(h_hof,c_hof)=self.encoder_hof(x_hof_2)

        h=torch.cat((h_acc,h_gyr,h_hof),dim=-1)
        c=torch.cat((c_acc,c_gyr,c_hof),dim=-1)

        h_acc_att=self.temporal_attn_acc(x_acc)
        h_gyr_att=self.temporal_attn_gyr(x_gyr)
        h_hof_att=self.temporal_attn_hof(x_hof)


        h_att=torch.cat((h_acc_att,h_gyr_att,h_hof_att),dim=-1)

        #Do it separately

        h_att=h_att.unsqueeze(0)
        h_att=h_att.transpose(1,0)
        
        out=self.decoder(h, c, h_att, 9)


        return out

lr = 0.001
model = MM_ED_Bi_LSTM_WF(24,24,36)

mm_ed_bi_lstm_WF = train_mm_m(train_loader, lr,12,model,path+subject+'_mm_ed_bi_lstm_WF_IMU8_hof.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_bi_lstm_WF= MM_ED_Bi_LSTM_WF(24,24,36)
mm_ed_bi_lstm_WF.load_state_dict(torch.load(path+subject+'_mm_ed_bi_lstm_WF_IMU8_hof.pth'))
mm_ed_bi_lstm_WF.to(device)

mm_ed_bi_lstm_WF.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):                          
        output = mm_ed_bi_lstm_WF(data_acc.to(device).float(),data_gyr.to(device).float(),data_hof.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_12=np.hstack([rmse,p])



## Training Function

def train_mm_m(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    # criterion =nn.MSELoss()
    criterion =RMSELoss()
    # criterion =PearsonCorrCoefLoss()


    # criterion=PearsonCorrLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    optimizer = torch.optim.Adam(model.parameters())

    
    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10
    

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(train_loader):
            optimizer.zero_grad()
            output= model(data_acc.to(device).float(),data_gyr.to(device).float())
            
            loss = criterion(output, target_future.to(device).float())
            loss.backward()
            optimizer.step()
          
            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)       
            
       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_hof, target_present, target_future in val_loader:
                output= model(data_acc.to(device).float(),data_gyr.to(device).float())
                val_loss += criterion(output, target_future.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break


            
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

  
    # # Save the trained model
    # torch.save(model.state_dict(), "model.pth")

    return model

### 13. Attention Without gating+ Bi-LSTM+ Bi-GRU --Encoder+decoder
#
#class Encoder_lstm(nn.Module):
#    def __init__(self, input_dim, dropout):
#        super(Encoder_lstm, self).__init__()
#        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
#        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
#        self.dropout=nn.Dropout(dropout)
#              
#    def forward(self, x):
#        out_1, _ = self.lstm_1(x)
#        out_1=self.dropout(out_1)
#        out_2, (h_n, c_n) = self.lstm_2(out_1)
#        out_2=self.dropout(out_2)
#        
#        return out_2, (h_n, c_n)
#
#class Encoder_gru(nn.Module):
#    def __init__(self, input_dim, dropout):
#        super(Encoder_gru, self).__init__()
#        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
#        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
#        self.dropout=nn.Dropout(dropout)
#        
#        
#    def forward(self, x):
#        out_1, _ = self.lstm_1(x)
#        out_1=self.dropout(out_1)
#        out_2, h_n = self.lstm_2(out_1)
#        out_2=self.dropout(out_2)
#        
#        return out_2, h_n
#
#class LSTMDecoder(nn.Module):
#    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
#        super(LSTMDecoder, self).__init__()
#        self.hidden_size = hidden_size
#        self.num_layers = num_layers
#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
#        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
#        self.fc = nn.Linear(2*2*hidden_size, output_size)
#        self.dropout=nn.Dropout(dropout)
#
#        
#        
#    def forward(self, encoder_hidden, encoder_cell, h_gru, h_att_lstm, h_att_gru,  max_len):
#        batch_size = encoder_hidden.shape[1]
#        hidden = encoder_hidden
#        hidden_gru=h_gru
#        cell = encoder_cell
#        outputs = []
#        
#        # Use the last time step of target as the initial input
#        input_1 = torch.zeros(batch_size,1,6).to(device)
#        input = torch.cat((input_1, h_att_lstm, h_att_gru), dim=-1)
#
#        
#        for i in range(max_len):
#
#            # Run one time step of LSTM
#            output_1, (hidden, cell) = self.lstm(input, (hidden, cell))
#            output_2, hidden_gru = self.gru(input, hidden_gru)
#
#            output=torch.cat((output_1,output_2),dim=-1)
#
#            output=self.dropout(output)
#            
#            # Use the output for prediction
#            output = self.fc(output.squeeze(1))
#            outputs.append(output.unsqueeze(1))
#            
#            # Use the predicted output as the next input
#            input = torch.cat(( output.unsqueeze(1), input[:,:,6:518]), dim=-1)
#
#
#            
#        # Concatenate all the outputs along the time dimension
#        outputs = torch.cat(outputs, dim=1)
#        
#        return outputs
#
#class TemporalAttention(nn.Module):
#    def __init__(self, hidden_size):
#        super(TemporalAttention, self).__init__()
#
#        self.W = nn.Linear(hidden_size, hidden_size)
#        self.V = nn.Linear(hidden_size, 1)
#
#    def forward(self, x):
#        # x: (batch_size, sequence_length, hidden_size)
#
#        # Calculate attention weights.
#        attn = self.V(torch.tanh(self.W(x)))
#        attn = attn.squeeze(-1)
#        attn = torch.softmax(attn, dim=1)
#
#        # Calculate weighted average of hidden states.
#        context = attn.unsqueeze(-1) * x
#        context = context.sum(dim=1)
#
#        return context
#
#class MM_ED_Bi_LSTM_GRU_WFW(nn.Module):
#    def __init__(self, input_acc, input_gyr, drop_prob=0.05):
#        super(MM_ED_Bi_LSTM_GRU_WFW, self).__init__()   
#
#        self.encoder_acc_lstm=Encoder_lstm(input_acc, drop_prob)   
#        self.encoder_gyr_lstm=Encoder_lstm(input_gyr, drop_prob) 
#
#        self.encoder_acc_gru=Encoder_gru(input_acc, drop_prob)   
#        self.encoder_gyr_gru=Encoder_gru(input_gyr, drop_prob) 
#
#        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
#        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
#
#        self.temporal_attn_acc = TemporalAttention(128)
#        self.temporal_attn_gyr = TemporalAttention(128)
#
#
#        self.decoder=LSTMDecoder(2*2*128+6, 2*64, 6, 1, 0.05)
#
#    def forward(self, x_acc, x_gyr):
#
#        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
#        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
#
#        x_acc_1=self.BN_acc(x_acc_1)
#        x_gyr_1=self.BN_gyr(x_gyr_1)
#
#        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
#        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))
#
#        x_acc_lstm,(h_acc_lstm, c_acc_lstm)=self.encoder_acc_lstm(x_acc_2)
#        x_gyr_lstm,(h_gyr_lstm, c_gyr_lstm)=self.encoder_gyr_lstm(x_gyr_2)
#
#
#        x_acc_gru,h_acc_gru=self.encoder_acc_gru(x_acc_2)
#        x_gyr_gru,h_gyr_gru=self.encoder_gyr_gru(x_gyr_2)
#
#
#        h_lstm=torch.cat((h_acc_lstm,h_gyr_lstm),dim=-1)
#        c_lstm=torch.cat((c_acc_lstm,c_gyr_lstm),dim=-1)
#
#        h_gru=torch.cat((h_acc_gru,h_gyr_gru),dim=-1)
#
#
#        h_acc_att_lstm=self.temporal_attn_acc(x_acc_lstm)
#        h_gyr_att_lstm=self.temporal_attn_gyr(x_gyr_lstm)
#
#        h_acc_att_gru=self.temporal_attn_acc(x_acc_gru)
#        h_gyr_att_gru=self.temporal_attn_gyr(x_gyr_gru)
#
#
#        h_att_lstm=torch.cat((h_acc_att_lstm,h_gyr_att_lstm),dim=-1)
#        h_att_gru=torch.cat((h_acc_att_gru,h_gyr_att_gru),dim=-1)
#
#        #Do it separately
#
#        h_att_lstm=h_att_lstm.unsqueeze(0)
#        h_att_lstm=h_att_lstm.transpose(1,0)
#
#        h_att_gru=h_att_gru.unsqueeze(0)
#        h_att_gru=h_att_gru.transpose(1,0)
#        
#        out=self.decoder(h_lstm, c_lstm, h_gru, h_att_lstm, h_att_gru, 9)
#
#
#        return out
#
#lr = 0.001
#model = MM_ED_Bi_LSTM_GRU_WFW(24,24)
#
#mm_ed_bi_lstm_gru_WFW = train_mm_m(train_loader, lr,12,model,path+subject+'_mm_ed_bi_lstm_gru_WFW_IMU8.pth')
#
#gc.collect()
#gc.collect()
#gc.collect()
#
#mm_ed_bi_lstm_gru_WFW= MM_ED_Bi_LSTM_GRU_WFW(24,24)
#mm_ed_bi_lstm_gru_WFW.load_state_dict(torch.load(path+subject+'_mm_ed_bi_lstm_gru_WFW_IMU8.pth'))
#mm_ed_bi_lstm_gru_WFW.to(device)
#
#mm_ed_bi_lstm_gru_WFW.eval()
#
## iterate through batches of test data
#with torch.no_grad():
#    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):                          
#        output = mm_ed_bi_lstm_gru_WFW(data_acc.to(device).float(),data_gyr.to(device).float())
#        if i==0:
#          yhat_5=output
#          test_target=target_future
#
#        yhat_5=torch.cat((yhat_5,output),dim=0)
#        test_target=torch.cat((test_target,target_future),dim=0)
#
#        # clear memory
#        del data, target_future,output
#        torch.cuda.empty_cache()
#
#
#
#yhat_4 = yhat_5.detach().cpu().numpy() 
#test_target = test_target.detach().cpu().numpy()
#print(yhat_4.shape)
#
#rmse, p=RMSE_prediction(yhat_4,test_target)  
#
#ablation_13=np.hstack([rmse,p])

## 14. Attention Without gating+ Bi-LSTM + Bi-GRU --Encoder+decoder

class Encoder_lstm(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_lstm, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
              
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, (h_n, c_n) = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, (h_n, c_n)

class Encoder_gru(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_gru, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout=nn.Dropout(dropout)
        
        
    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, h_n = self.lstm_2(out_1)
        out_2=self.dropout(out_2)
        
        return out_2, h_n

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*2*hidden_size, output_size)
        self.dropout=nn.Dropout(dropout)

        self.gating_net = nn.Sequential(nn.Linear(2*2*128+6, 2*2*128+6),nn.Sigmoid())
        
        
    def forward(self, encoder_hidden, encoder_cell, h_gru, h_att_lstm, h_att_gru,  max_len):
        batch_size = encoder_hidden.shape[1]
        hidden = encoder_hidden
        hidden_gru=h_gru
        cell = encoder_cell
        outputs = []
        
        # Use the last time step of target as the initial input
        input_1 = torch.zeros(batch_size,1,6).to(device)
        input = torch.cat((input_1, h_att_lstm, h_att_gru), dim=-1)

        
        for i in range(max_len):

            gating_weight=self.gating_net(input)
            input=input*gating_weight

            # Run one time step of LSTM
            output_1, (hidden, cell) = self.lstm(input, (hidden, cell))
            output_2, hidden_gru = self.gru(input, hidden_gru)

            output=torch.cat((output_1,output_2),dim=-1)

            output=self.dropout(output)
            
            # Use the output for prediction
            output = self.fc(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
            
            # Use the predicted output as the next input
            input = torch.cat(( output.unsqueeze(1), input[:,:,6:518]), dim=-1)


            
        # Concatenate all the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()

        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        # Calculate attention weights.
        attn = self.V(torch.tanh(self.W(x)))
        attn = attn.squeeze(-1)
        attn = torch.softmax(attn, dim=1)

        # Calculate weighted average of hidden states.
        context = attn.unsqueeze(-1) * x
        context = context.sum(dim=1)

        return context

class MM_ED_Bi_LSTM_GRU_WF(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.05):
        super(MM_ED_Bi_LSTM_GRU_WF, self).__init__()   

        self.encoder_acc_lstm=Encoder_lstm(input_acc, drop_prob)   
        self.encoder_gyr_lstm=Encoder_lstm(input_gyr, drop_prob) 

        self.encoder_acc_gru=Encoder_gru(input_acc, drop_prob)   
        self.encoder_gyr_gru=Encoder_gru(input_gyr, drop_prob) 

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.temporal_attn_acc = TemporalAttention(128)
        self.temporal_attn_gyr = TemporalAttention(128)


        self.decoder=LSTMDecoder(2*2*128+6, 2*64, 6, 1, 0.05)

    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, window, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, window, x_gyr_1.size(-1))

        x_acc_lstm,(h_acc_lstm, c_acc_lstm)=self.encoder_acc_lstm(x_acc_2)
        x_gyr_lstm,(h_gyr_lstm, c_gyr_lstm)=self.encoder_gyr_lstm(x_gyr_2)


        x_acc_gru,h_acc_gru=self.encoder_acc_gru(x_acc_2)
        x_gyr_gru,h_gyr_gru=self.encoder_gyr_gru(x_gyr_2)


        h_lstm=torch.cat((h_acc_lstm,h_gyr_lstm),dim=-1)
        c_lstm=torch.cat((c_acc_lstm,c_gyr_lstm),dim=-1)

        h_gru=torch.cat((h_acc_gru,h_gyr_gru),dim=-1)


        h_acc_att_lstm=self.temporal_attn_acc(x_acc_lstm)
        h_gyr_att_lstm=self.temporal_attn_gyr(x_gyr_lstm)


        h_acc_att_gru=self.temporal_attn_acc(x_acc_gru)
        h_gyr_att_gru=self.temporal_attn_gyr(x_gyr_gru)


        h_att_lstm=torch.cat((h_acc_att_lstm,h_gyr_att_lstm),dim=-1)
        h_att_gru=torch.cat((h_acc_att_gru,h_gyr_att_gru),dim=-1)

        #Do it separately

        h_att_lstm=h_att_lstm.unsqueeze(0)
        h_att_lstm=h_att_lstm.transpose(1,0)

        h_att_gru=h_att_gru.unsqueeze(0)
        h_att_gru=h_att_gru.transpose(1,0)


        
        out=self.decoder(h_lstm, c_lstm, h_gru, h_att_lstm, h_att_gru, 9)


        return out

lr = 0.001
model = MM_ED_Bi_LSTM_GRU_WF(24,24)

#mm_ed_bi_lstm_gru_WF = train_mm_m(train_loader, lr,12,model,path+subject+'_mm_ed_bi_lstm_gru_WF_IMU8.pth')

gc.collect()
gc.collect()
gc.collect()

mm_ed_bi_lstm_gru_WF= MM_ED_Bi_LSTM_GRU_WF(24,24)
mm_ed_bi_lstm_gru_WF.load_state_dict(torch.load(path+subject+'_mm_ed_bi_lstm_gru_WF_IMU8.pth'))
mm_ed_bi_lstm_gru_WF.to(device)

mm_ed_bi_lstm_gru_WF.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_hof, target_present, target_future) in enumerate(test_loader):                          
        output = mm_ed_bi_lstm_gru_WF(data_acc.to(device).float(),data_gyr.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target_future

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target_future),dim=0)

        # clear memory
        del data, target_future,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy() 
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p=RMSE_prediction(yhat_4,test_target)  

ablation_14=np.hstack([rmse,p])


s1=yhat_4.shape[0]*yhat_4.shape[1]

test_target=test_target.reshape((s1,6))
yhat=yhat_4.reshape((s1,6))




path_1='/home/sanzidpr/HICCS_submission/Dataset A/Results/'


#### Result Summary

result=np.hstack([test_target,yhat])


from numpy import savetxt
savetxt(path_1+'Dataset_A_'+subject+'_plot.csv', result, delimiter=',')
       
    
#################################################################################################################################################################################################




