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