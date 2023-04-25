
def function():
    try:
        with open('server1.txt', 'r+') as file:
            lines = file.readlines()
            # print(lines)
        with open('server2.txt', 'r+') as file1:
            lines1 = file1.readlines()
            # print(lines1)
        with open('average.txt', 'w') as file2:
            file2.write('')
        for i in range(0,6):
            print(str((int(lines[i])+int(lines1[i])/2)))
            file2.write(str((int(lines[i])+int(lines1[i])/2)))
        return 0
    except:
        return 0

function()