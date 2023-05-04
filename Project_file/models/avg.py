
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

print(function())









