import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from collections import OrderedDict
import re
import os

# Define the MLP model
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

def main_plotting(file_path):
    # Define data transformation
    data_transform = transforms.Compose([
        transforms.Resize((14, 14)), 
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
    ])

    # Load the test dataset
    test_dataset_01 = ImageFolder('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\test_mnist_01', transform=data_transform)
    batch_size = 64
    test_loader_01 = DataLoader(test_dataset_01, batch_size=batch_size, shuffle=True)
    model = MLP()
    
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\avg_tmp.txt', 'r') as file:
        if(len(file.read())==0):
            return 
        
    
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\avg_tmp.txt','w') as outFile:
        with open(file_path,'r') as file:
            for line in file:
                cleaned_line = re.sub(r'\s+', ' ', line)
                outFile.write(cleaned_line)
                    
        with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\avg_tmp.txt', 'r') as file:
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

    def test_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    # Calculate the testing accuracy
    testing_accuracy = test_model(model, test_loader_01)
    print(f"Testing Accuracy: {testing_accuracy:.2f}%")
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\accuracy.txt', 'a') as f:
        f.write(f"{testing_accuracy}\n")    

    while True:
        def read_data(file_path):
            data = []
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    data.append(float(line.strip()))
            return data

        def plot_data(data):
            x = list(range(1, len(data)+1))
            plt.plot(x, data)
            plt.xlabel('Rounds')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Plot for Each Client')
            plt.show()

        file_path1 = 'D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\accuracy1.txt'
        data = read_data(file_path1)
        plot_data(data)

        file_path2 = 'D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\accuracy2.txt'
        data = read_data(file_path2)
        plot_data(data)
        
main_plotting("D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\avg_param.txt")