"""
Client 2 Training

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def check(n_features):
    model = LogisticRegression(n_features)
    file_path = "C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\average.txt"
    if is_file_empty(file_path):
        return model
    else:
        file_content = read_file(file_path)
        model.load_state_dict(torch.load(file_content))
        return model

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
    
    correct  =0
    total = 0
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
        
        y_pred = torch.round(y_pred)
        correct += (y_pred.reshape(1584) == y_train).sum().item()
        total += y_train.size(0)
    total_loss = sum(error_loss)/len(error_loss)
    train_accuracy = correct/total
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


def function():
    df = pd.read_csv("C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\feature_selected_voice_data.csv")

    idx = int(len(df)*0.5)
    client2_dataset = df[idx:]

    # Client2 Datatset -->
    client2_X = client2_dataset.iloc[:,:-1]
    client2_Y = client2_dataset["label"]
    le = preprocessing.LabelEncoder()
    client2_Y = le.fit_transform(client2_Y)
    client2_X = client2_X.to_numpy()
    # client2_Y = client2_Y.to_numpy()

    X_train_2 = client2_X.astype('float32')
    y_train_2 = client2_Y.astype('float32')

    X_train_2 = torch.from_numpy(X_train_2)
    y_train_2 =torch.from_numpy(y_train_2)

    n_samples, n_features = X_train_2.shape
    model = check(n_features)
    param_dict, loss1, training_accuracy = clients_training(X_train_2, y_train_2, model)

    param = get_weights(param_dict)
    rounded_param = [round(num, 3) for num in param]
    param_str = " ".join(str(num) for num in rounded_param)

    try:
        with open('C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\server2.txt', 'w') as file:
            file.write(param_str)
    except Exception as e:
        print("An error occurred while writing to the file:", e)
        
    with open('C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\accuracy2.txt', 'a') as file:
            file.write(training_accuracy)

print(function())
