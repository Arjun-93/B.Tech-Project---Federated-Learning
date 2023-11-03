<<<<<<< HEAD
def function():
    with open('Project_file\\models\\server1.txt', 'r') as file:
        lines = file.readlines()
        list1 = []
        for i in lines[0].split():
            try:
                list1.append(float(i))
            except:
                list1.append(0.0)
        #list1 = [float(x) for x in lines[0].split()]

    with open('Project_file\\models\\server1.txt', 'r') as file1:
        lines1 = file1.readlines()
        list2 = []
        for i in lines1[0].split():
            try:
                list2.append(float(i))
            except:
                list2.append(0.0)
        # print(type(lines1))
        # print(lines1)
        # list2 = [float(y) for y in lines1[0].split()]

    avg_str = ""
    
    if(len(list1) != len(list2)):
        if(len(list1) > len(list2)):
            for i in range(abs(len(list1)-len(list2))):
                list2.append(0.0)
        else:
            for i in range(abs(len(list1)-len(list2))):
                list1.append(0.0)
    for i in range(len(list1)):
        average = (list1[i] + list2[i]) / 2
        avg_str += str(round(average, 5)) + " "
        with open('Project_file\models\average.txt', 'w') as file2:
            file2.write(avg_str)
    return avg_str

print(function())
=======
import matplotlib.pyplot as plt

def function():
    with open('C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\server1.txt', 'r') as file:
        lines = file.readlines()
        list1 = [float(x) for x in lines[0].split()]

    with open('C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\server2.txt', 'r') as file1:
        lines1 = file1.readlines()
        # print(type(lines1))
        # print(lines1)
        list2 = [float(y) for y in lines1[0].split()]

    avg_str = ""
    for i in range(len(list1)):
        average = (list1[i] + list2[i]) / 2
        avg_str += str(round(average, 3)) + " "
        with open('C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\average.txt', 'w') as file2:
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

# file_path = 'C:\\Users\\arjun\\OneDrive\\Desktop\\BTP\\B.Tech-Project---Federated-Learning\\Project_file\\models\\accuracy1.txt'
# data = read_data(file_path)
# plot_data(data)

print(function())









>>>>>>> c2141a36778e0b10fc3105f4142c413f5ec2f56c
