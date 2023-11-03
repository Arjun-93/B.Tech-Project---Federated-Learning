"""
Client 1 Training

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

<<<<<<< HEAD

def function():
    df = pd.read_csv('Project_file\\models\\feature_selected_voice_data.csv')
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
    param =  (round(x,5) for x in param)
    # for x in param:
    #     print((str(abs(x))))
    str_parm = " ".join(str(y) for y in param)
    # print(str_parm)
    return str_parm
=======
import os
import torch
from io import BytesIO
import torch
from collections import OrderedDict
import os
import torch
from io import BytesIO
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
>>>>>>> c2141a36778e0b10fc3105f4142c413f5ec2f56c

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
def clients_training(X_train, y_train, lr):
    num_epochs = 10
    learning_rate = 0.1 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(lr.parameters(), lr=learning_rate) 
    error_loss = []
    
    correct = 0
    total = 0
    for epoch in range(num_epochs):
        train_loss = 0
        optimizer.zero_grad()
        y_pred = lr(X_train)
        
        # Add random noise to the prediction
        noise = torch.randn_like(y_pred)
        y_pred_with_noise = y_pred + noise
        
        loss = criterion(y_pred_with_noise.reshape(1584), y_train)             
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_train.size(0)
        train_loss = train_loss / 1584
        error_loss.append(train_loss)
        
        y_pred = torch.round(torch.sigmoid(y_pred_with_noise))
        correct += (y_pred.reshape(1584) == y_train).sum().item()
        total += y_train.size(0)
    
    total_loss = sum(error_loss) / len(error_loss)
    train_accuracy = correct / total
    return lr.state_dict(), total_loss, train_accuracy

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

<<<<<<< HEAD
strm = function()
print(strm)

=======

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def check(n_features):
    model = LogisticRegression(n_features)
    file_path = 'C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\average.txt'
    if os.path.getsize(file_path) == 0:
        return model
    else:
        data_string = read_file(file_path)
        data_list = data_string.strip().split()
        # Convert the list of values into a tensor
        tensor = torch.tensor([float(value) for value in data_list])
        truncated_tensor = tensor[:13].unsqueeze(0)
        state_dict = OrderedDict([('linear.weight', truncated_tensor), ('linear.bias', tensor.new_zeros(1))])
        model.load_state_dict(state_dict)
        return model
        

def function():
    df = pd.read_csv("C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\feature_selected_voice_data.csv")
    train_idx = int(len(df)*0.8)
    test_idx = len(df) - train_idx

    train_idx2 =  int(train_idx/2)
    client1_dataset = df[:train_idx2]
    client2_dataset = df[train_idx2:train_idx]
    
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
    model = check(n_features)
    param_dict, loss1 , training_accuracy = clients_training(X_train_1, y_train_1, model)

    param = get_weights(param_dict)
    rounded_param = [round(num, 3) for num in param]
    param_str = " ".join(str(num) for num in rounded_param)
    print(training_accuracy)
    with open('C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\server1.txt', 'w') as file:
            file.write(param_str)
    with open('C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\accuracy1.txt', 'w') as file:
            file.write(str(training_accuracy) + "\n")

    return param_str

print(function())


>>>>>>> c2141a36778e0b10fc3105f4142c413f5ec2f56c





