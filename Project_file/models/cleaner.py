def remove_null_characters(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    cleaned_content = content.replace('\x00', '')

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned_content)
def is_valid_rows(rows):
    for row in rows:
        if not row.isdigit() or len(row) < 1:
            return False
    return True

def function(file_name):
    count = 0
    with open(file_name, 'r') as file:
        rows = [line.strip() for line in file]
        valid_rows = []
        for row in rows:
            if is_valid_rows([row]):
                count = count +1
                valid_rows.append(row)
        if count < 6:
            return False
        else:
            output = '\n'.join(valid_rows)
            print(output)
            with open('input.txt', 'w') as file:
                file.write(output)
        return True
<<<<<<< HEAD
remove_null_characters('C:\\Users\\intel\\Desktop\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Project_file\\models\\servertoclient2.txt', 'C:\\Users\\intel\\Desktop\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Project_file\\models\\servertoclient2.txt')
function('C:\\Users\\intel\\Desktop\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Project_file\\models\\servertoclient2.txt')
=======


def filter_numbers(noisy_string):
    filtered_string = ''.join(filter(str.isdigit, noisy_string))
    return filtered_string
>>>>>>> c2141a36778e0b10fc3105f4142c413f5ec2f56c
