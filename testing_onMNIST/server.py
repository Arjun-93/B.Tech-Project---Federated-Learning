import time
start_time = time.time()
from format import main_quantize
from dequantize import main_dequnatize
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from format import main_quantize
from collections import OrderedDict
from cleaner import remove_null_characters
import re

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

model_01 = MLP()
model_23 = MLP()
server_model = MLP()

pathfile = 'D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\avg_param.txt'
try:
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\client1_server.txt', 'r') as f:
        content = f.read()
        content = content.replace('\x00','')

    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\client1_server.txt', 'w') as f:
        f.write(content)
    param1 = main_dequnatize('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\client1_server.txt',1)
    model_01.load_state_dict(param1)


    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\client2_server.txt', 'r') as f:
        content = f.read()
        content = content.replace('\x00','')

    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\client2_server.txt', 'w') as f:
        f.write(content)
    param2 = main_dequnatize('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\client2_server.txt',2)
    model_23.load_state_dict(param2)
        
    model_01_param = model_01.state_dict()
    model_23_param = model_23.state_dict()

    avg_param = {}
    for key in model_01_param.keys():
        avg_param[key] = (0.6*model_01_param[key] + model_23_param[key]) / 2
        
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\avg_param.txt', 'w') as f:
        for key, value in avg_param.items():
            f.write(f'{key}: {value}\n')
    
    # convert avg_param into ordered dict
    avg_param = OrderedDict(avg_param)
    server_model.load_state_dict(avg_param)
            
    # copy the content of avg_param.txt to previous_param.txt
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\avg_param.txt', 'r') as f:
        content = f.read()
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\previous_param.txt', 'w') as f:
        f.write(content)
        
except:
    pathfile = 'D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\previous_param.txt'
    
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\test_temp.txt','w') as outFile:
        with open(pathfile,'r') as file:
            for line in file:
                cleaned_line = re.sub(r'\s+', ' ', line)
                outFile.write(cleaned_line)
                
    with open('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\test_temp.txt', 'r') as file:
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
    server_model.load_state_dict(parameter)

data_transform = transforms.Compose([
    transforms.Resize((14, 14)),  
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),
])

################################################################################## 
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

test_dataset = ImageFolder('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\test_mnist_23', transform=data_transform)
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_accuracy = test_model(server_model, test_loader)
with open ('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\accuracy.txt','a') as f:
    f.write(str(test_accuracy)+'\n')
#################################################################################
        
quantize_param = main_quantize(pathfile,3)

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Write elapsed time to a text file
with open('execution_time_server.txt', 'w') as file:
    file.write(f"Elapsed time: {elapsed_time} seconds")
