import os
import torch.nn as nn
import torch
from collections import OrderedDict
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        
    #sigmoid transformation of the input 
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def createModel(n_features):
    model = LogisticRegression(n_features)
    file_path = "C:\\Users\\intel\\Desktop\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Project_file\\models\\average.txt"
    if os.path.getsize(file_path) == 0:
        return "There is no parmerters to load"
    else:
        data_string = read_file(file_path)
        data_list = data_string.strip().split()
        tensor = torch.tensor([float(value) for value in data_list])
        truncated_tensor = tensor[:13].unsqueeze(0)
        state_dict = OrderedDict([('linear.weight', truncated_tensor), ('linear.bias', tensor.new_zeros(1))])
        model.load_state_dict(state_dict)
        return model


def createTest_dataset():
    df = pd.read_csv("C:\\Users\\intel\\Desktop\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Project_file\\models\\feature_selected_voice_data.csv")
    train_idx = int(len(df)*0.8)
    test_idx = len(df) - train_idx

    client2_dataset = df[test_idx:]

    # Client2 Datatset -->
    X_test = client2_dataset.iloc[:,:-1]
    y_test = client2_dataset["label"]
    
    le = preprocessing.LabelEncoder()
    y_test = le.fit_transform(y_test)
    X_test = X_test.to_numpy()
    # y_test = y_test.to_numpy()

    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    X_test = torch.from_numpy(X_test)
    y_test =torch.from_numpy(y_test)
    n_samples, n_features = X_test.shape
    return X_test, y_test, n_features
3
def testing(X_test, y_test, model):
    criterion = nn.BCELoss() 
    correct = 0
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = y_pred.view(-1)
        predicted = torch.round(y_pred)
        correct += (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = 100 * correct / total
        test_loss = criterion(y_pred, y_test.float())
    return test_loss, accuracy

X_test, y_test, n_features = createTest_dataset()
test_model = createModel(n_features)
# print(test_model)
# print(test_model.state_dict())
test_loss, correct = testing(X_test, y_test, test_model)
print(f"Test Loss: {test_loss.item():.4f}, accuracy: {correct}/{len(y_test)})")