import matplotlib.pyplot as plt

def function():
    with open('server1.txt', 'r') as file:
        lines = file.readlines()
        list1 = [float(x) for x in lines[0].split()]

    with open('server2.txt', 'r') as file1:
        lines1 = file1.readlines()
        # print(type(lines1))
        # print(lines1)
        list2 = [float(y) for y in lines1[0].split()]

    avg_str = ""
    for i in range(len(list1)):
        average = (list1[i] + list2[i]) / 2
        avg_str += str(round(average, 3)) + " "
        with open('average.txt', 'w') as file2:
            file2.write(avg_str)
    return avg_str

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

file_path = 'C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\accuracy1.txt'
data = read_data(file_path)
plot_data(data)

# print(function())









