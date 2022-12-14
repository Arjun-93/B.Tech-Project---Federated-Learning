"""
Client 1 Training

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        
    #sigmoid transformation of the input 
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def clients_training(X_train, y_train, lr):
    num_epochs = 500
    learning_rate = 0.0001 
    criterion = nn.BCELoss() # Binary cross Entropy loss                              
    optimizer = torch.optim.SGD(lr.parameters(), lr=learning_rate) 
    error_loss = []
    for epoch in range(num_epochs):
        train_loss = 0
        optimizer.zero_grad()
        y_pred = lr(X_train)
        loss = criterion(y_pred.reshape(1584), y_train)             
        loss.backward()
        optimizer.step()
        # if (epoch+1) % 20 == 0:                                          
        #     print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        train_loss += loss.item()*X_train.size(0)
        train_loss = train_loss/1584
        error_loss.append(train_loss)
    total_loss = sum(error_loss)/len(error_loss)
    return lr.state_dict(), total_loss

def get_weights(param_dict):
    weight = param_dict['linear.weight']
    bias = param_dict['linear.bias']

    weight = weight.tolist()
    bias = bias.tolist()

    parameter_list = weight[0]
    parameter_list.append(bias[0])
    # print(len(parameter_list))
    # print(parameter_list)
    return parameter_list

df = pd.read_csv("Project_file\\models\\feature_selected_voice_data.csv")

idx = int(len(df)*0.5)
client1_dataset = df[:idx]

# Client1 dataset -->
client1_X = client1_dataset.iloc[:,:-1]
client1_Y = client1_dataset["label"]
le = preprocessing.LabelEncoder()
client1_Y = le.fit_transform(client1_Y)
client1_X = client1_X.to_numpy()
# client1_Y = client1_Y.to_numpy()

X_train_1 = client1_X.astype('float32')
y_train_1 = client1_Y.astype('float32')

X_train_1 = torch.from_numpy(X_train_1)
y_train_1 =torch.from_numpy(y_train_1)

n_samples, n_features = X_train_1.shape
model = LogisticRegression(n_features)
param_dict, loss1 = clients_training(X_train_1, y_train_1, model)

param = get_weights(param_dict)
# print(param)
