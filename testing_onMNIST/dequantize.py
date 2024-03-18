import numpy as np
import torch
import struct

def binary_string(binary_strings):
    floating_numbers = []
    integer_numbers = []
    length = len(binary_strings)
    
    # Extract the first 12 floating-point numbers (31 bits each)
    # for i in range(12):
    #     start_index = i * 31
    #     end_index = start_index + 31
    #     floating_numbers.append(binary_strings[start_index:end_index])
    
    # Extract the integer numbers (16 bits each) after the floating-point numbers
    for i in range(0, length, 16):
        start_index = i
        end_index = start_index + 16
        integer_numbers.append(binary_strings[start_index:end_index])

    binary_string = floating_numbers + integer_numbers
    # reconstructed_numbers = []
    level_string = ""
    for binary_string in floating_numbers:
        binary = '0' + binary_string
        integer_value = int(binary, 2)
        packed = struct.pack('!I', integer_value)
        reconstructed_float = struct.unpack('f', packed)[0]
        # reconstructed_numbers.append(reconstructed_float)
        level_string += str(reconstructed_float) + " "

    for binary_string in integer_numbers:
        binary = '0' + binary_string
        integer_value = int(binary, 2)
        # reconstructed_numbers.append(integer_value)
        level_string += str(integer_value) + " "

    return level_string

def hex_to_binary(hexadecimal_string):
    hex_chars  = hexadecimal_string.split()
    binary_string = ""
    for hex_char in hex_chars:
        binary_string += bin(int(hex_char, 16))[2:].zfill(4)
    return binary_string

    # hex_str = "4114 40ff 4107 3ba2 421b 41f2 3bf1 40b8 4046 3f67"

# Function to dequantize a quantized matrix
def dequantize(quantized_matrix, max_val, min_val, num_bits):
    max_val = float(max_val)
    min_val = float(min_val)
    dequantized = ((quantized_matrix / (2**num_bits - 1)) * (max_val - min_val)) + min_val
    return torch.tensor(dequantized)  # Convert to a PyTorch tensor

def main_dequnatize(file_path, client_num):
    with open(file_path, "r") as file:
        hex_string = file.read()
    file_path_temp = "D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\temp_client1.txt"
    if(client_num == 2):
        file_path_temp = "D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\temp_client2.txt"
    elif(client_num == 3):
        file_path_temp = "D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\temp_server.txt"
    with open(file_path_temp, "r") as file_temp:
        bit_string_temp = file_temp.read()
    
    hex_string = hex_string.replace('\x00', '')
    hex_string = hex_string.rstrip()
    binary_str = hex_to_binary(hex_string[:-2])
    binary_str = binary_str.rstrip()
    level_string = binary_string(binary_str)
    binary_str = level_string.rstrip()
    # print(len(hex_string))
    if(len(hex_string) != 3386):
        raise ValueError("bad length")
        return

    parts = level_string.split()
    parts_temp = bit_string_temp.split()
    fc1_weight_size = (4, 196)
    fc1_bias_size = 4
    fc2_weight_size = (8, 4)
    fc2_bias_size = 8
    fc3_weight_size = (2, 8)
    fc3_bias_size = 2

    fc1_weight_max = float(parts_temp[0])
    fc1_weight_min = float(parts_temp[1])
    fc1_bias_max = float(parts_temp[2])
    fc1_bias_min = float(parts_temp[3])

    fc2_weight_max = float(parts_temp[4])
    fc2_weight_min = float(parts_temp[5])
    fc2_bias_max = float(parts_temp[6])
    fc2_bias_min = float(parts_temp[7])

    fc3_weight_max = float(parts_temp[8])
    fc3_weight_min = float(parts_temp[9])
    fc3_bias_max = float(parts_temp[10])
    fc3_bias_min = float(parts_temp[11])

    quantized_values = list(map(int, parts))

    index = 0
    fc1_weight_dequantized = dequantize(np.array(quantized_values[index:index + fc1_weight_size[0] * fc1_weight_size[1]]), fc1_weight_max, fc1_weight_min, 16)
    index += fc1_weight_size[0] * fc1_weight_size[1]

    fc1_bias_dequantized = dequantize(np.array(quantized_values[index:index + fc1_bias_size]), fc1_bias_max, fc1_bias_min, 16)
    index += fc1_bias_size

    fc2_weight_dequantized = dequantize(np.array(quantized_values[index:index + fc2_weight_size[0] * fc2_weight_size[1]]), fc2_weight_max, fc2_weight_min, 16)
    index += fc2_weight_size[0] * fc2_weight_size[1]

    fc2_bias_dequantized = dequantize(np.array(quantized_values[index:index + fc2_bias_size]), fc2_bias_max, fc2_bias_min, 16)
    index += fc2_bias_size

    fc3_weight_dequantized = dequantize(np.array(quantized_values[index:index + fc3_weight_size[0] * fc3_weight_size[1]]), fc3_weight_max, fc3_weight_min, 16)
    index += fc3_weight_size[0] * fc3_weight_size[1]

    fc3_bias_dequantized = dequantize(np.array(quantized_values[index:index + fc3_bias_size]), fc3_bias_max, fc3_bias_min, 16)
    fc1_weight_dequantized = fc1_weight_dequantized.reshape(fc1_weight_size)
    fc2_weight_dequantized = fc2_weight_dequantized.reshape(fc2_weight_size)
    fc3_weight_dequantized = fc3_weight_dequantized.reshape(fc3_weight_size)

    model_dict = {
        'fc1.weight': fc1_weight_dequantized,
        'fc1.bias': fc1_bias_dequantized,
        'fc2.weight': fc2_weight_dequantized,
        'fc2.bias': fc2_bias_dequantized,
        'fc3.weight': fc3_weight_dequantized,
        'fc3.bias': fc3_bias_dequantized,
    }
    # print(model_dict)
    return model_dict


    # # Now you have key-value pairs for the dequantized weight and bias matrices in a dictionary:
    # for key, value in model_dict.items():
    #     print(f"{key}: {value}")

# a = main_dequnatize("D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client1.txt",1)
# print(a)