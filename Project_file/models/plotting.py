import matplotlib.pyplot as plt
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
    plt.title('Accuracy Plot for Client1')
    plt.show()

file_path = 'C:\\Users\\intel\\Desktop\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Project_file\\models\\accuracy1.txt'
data = read_data(file_path)
plot_data(data)