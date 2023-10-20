def function():
    with open('Final_Demo\\server1.txt', 'r') as file:
        lines = file.readlines()
        list1 = []
        for i in lines[0].split():
            try:
                list1.append(float(i))
            except:
                list1.append(0.0)
        #list1 = [float(x) for x in lines[0].split()]

    with open('Final_Demo\\server1.txt', 'r') as file1:
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
        with open('Final_Demo\\average.txt', 'w') as file2:
            file2.write(avg_str)
    return avg_str

print(function())