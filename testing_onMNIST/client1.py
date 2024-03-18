import time

# Record start time
start_time = time.time()

import os
import json
import torch
import torch
import ast
# from dequantize import main_dequnatize
import dequantize as dq
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from format1 import main_quantize
from collections import OrderedDict
import re
import fileinput
# import plotting as pp
import numpy as np

np.random.seed(0)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(14 * 14, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 2)  

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, dataloader, num_epochs=10, learning_rate=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    accuracy_list = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = correct / total_samples
        accuracy_list.append(accuracy)
    accuracy = np.sum(accuracy_list)/len(accuracy_list)
    
data_transform = transforms.Compose([
    transforms.Resize((14, 14)),  
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),
])

dataset_01 = ImageFolder('train_mnist_01', transform=data_transform)
batch_size = 64
loader_01 = DataLoader(dataset_01, batch_size=batch_size, shuffle=True)
model = MLP()
file_path = "D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client1.txt"
if os.path.getsize(file_path) == 0:
    train_model(model, loader_01)
else:
    try :
        with open(file_path, "r") as file:
            data = file.read()

        data = data.replace('\x00', '')
        with open(file_path, "w") as file:  
            file.write(data)

        parameter = dq.main_dequnatize(file_path,1)
        # print(len(parameter))
        model.load_state_dict(parameter)
        train_model(model, loader_01)
    except:
        with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client1_output_model.txt','w') as outFile:
            with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client1_model.txt','r') as file:
                for line in file:
                    cleaned_line = re.sub(r'\s+', ' ', line)
                    outFile.write(cleaned_line)
        with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client1_output_model.txt', 'r') as file:
            model_data = file.read()

        model_data = model_data.replace(' ', '')
        model_data = model_data.replace('tensor', '')
        lines = model_data.strip().split('\n')
        model_dict = OrderedDict()
        keys=[]
        values=[]
        model_ls = model_data.split(':')
        keys.append(model_ls[0])
        for i in model_ls[1:-1]:
            value , key = i[1:].split(')')
            values.append(eval(value))
            keys.append(key)
            
        values.append(eval(model_ls[-1]))
        for i in range(len(keys)):
            model_dict[keys[i]] = torch.FloatTensor(values[i])
        parameter = model_dict
        model.load_state_dict(parameter)
        train_model(model, loader_01)
# print((model.state_dict()))
with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client1_model.txt', 'w') as f:
    for key, value in model.state_dict().items():
        f.write(f"{key}: {value}\n")
        
        
# testing model accuracy #####################################################
# def test_model(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     return accuracy

# test_dataset = ImageFolder('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\test_mnist_01', transform=data_transform)
# batch_size = 64
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# test_accuracy = test_model(model, test_loader)
# with open ('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\accuracy2.txt','a') as f:
#     f.write(str(test_accuracy)+'\n')
#################################################################################    


with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\previous_model_client1.txt', 'a') as f:
    for key, value in model.state_dict().items():
        f.write(f"{key}: {value}\n")    

main_quantize('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client1_model.txt',1)

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Write elapsed time to a text file
with open('execution_time_client1.txt', 'w') as file:
    file.write(f"Elapsed time: {elapsed_time} seconds")

    