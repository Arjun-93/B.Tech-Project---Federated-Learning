import numpy as np
import torch
import struct
# import dequantize as dq
binary_numbers = []
def floatingPoint_to_binary(input_string, num_bits=16):
    numbers = input_string.split()
    floating_string = ""
    binary_string = ""
    for num in numbers:
        if "." in num:
            float_num = float(num)
            packed = struct.pack('f', float_num)
            binary_string = bin(struct.unpack('!I', packed)[0])[2:].zfill(32)
            binary_string = binary_string[1:]
            binary_string = binary_string.zfill(31)
            binary_numbers.append(binary_string)
            # floating_string += num + " " + num + " "
        else:
            abs_num = abs(int(num))
            binary_num = bin(abs_num)[2:]
            binary_num = binary_num.zfill(num_bits)
            binary_numbers.append(binary_num)

    # while(len(floating_string)<200):
    #     floating_string+=" "
    # print((floating_string))
    # binary_string  += floating_string
    binary_string = "".join(binary_numbers)
    with open("D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\binary_string.txt", 'w') as file:
        file.write(binary_string)
    return binary_string

def hexadecimal_converter(binary_string):
    hexadecimal_string = ""
    for i in range(0, len(binary_string), 16):
        temp_binary = binary_string[i:i + 16]
        temp_hex = ""
        for j in range(0, len(temp_binary), 4):
            temp_hex += hex(int(temp_binary[j:j + 4], 2))[2:]
        hexadecimal_string += temp_hex
    return hexadecimal_string

def quantize(matrix, num_bits):
    max_val = np.max(matrix)
    min_val = np.min(matrix)
    quantized_matrix = np.round((matrix - min_val) / (max_val - min_val) * (2**num_bits - 1))
    return quantized_matrix, max_val, min_val

def main_quantize(file_path, client_num):
    with open(file_path, "r") as file:
        weights_text = file.read()

    
    # multiBit = [2,4,8,16,32]
    b = 16
    # Extract the weights and biases from the text
    fc1_weight_text = weights_text.split("fc1.weight: tensor(")[1].split(")")[0]
    fc2_weight_text = weights_text.split("fc2.weight: tensor(")[1].split(")")[0]
    fc3_weight_text = weights_text.split("fc3.weight: tensor(")[1].split(")")[0]
    fc1_bias_text = weights_text.split("fc1.bias: tensor(")[1].split(")")[0]
    fc2_bias_text = weights_text.split("fc2.bias: tensor(")[1].split(")")[0]
    fc3_bias_text = weights_text.split("fc3.bias: tensor(")[1].split(")")[0]

    # Convert the weights and biases to numpy arrays
    fc1_weight = np.array(eval(fc1_weight_text))
    fc2_weight = np.array(eval(fc2_weight_text))
    fc3_weight = np.array(eval(fc3_weight_text))
    fc1_bias = np.array(eval(fc1_bias_text))
    fc2_bias = np.array(eval(fc2_bias_text))
    fc3_bias = np.array(eval(fc3_bias_text))

    # Quantize the weights and biases
    fc1_weight_quantized, fc1_weight_max, fc1_weight_min = quantize(fc1_weight, 16)
    fc2_weight_quantized, fc2_weight_max, fc2_weight_min = quantize(fc2_weight, 16)
    fc3_weight_quantized, fc3_weight_max, fc3_weight_min = quantize(fc3_weight, 16)
    fc1_bias_quantized, fc1_bias_max, fc1_bias_min = quantize(fc1_bias, 16)
    fc2_bias_quantized, fc2_bias_max, fc2_bias_min = quantize(fc2_bias, 16)
    fc3_bias_quantized, fc3_bias_max, fc3_bias_min = quantize(fc3_bias, 16)

    bit_string = ""
    bit_string_temp = ""
    bit_string_temp += f"{fc1_weight_max} {fc1_weight_min} "  # Max and min values of fc1_weight
    bit_string_temp += f"{fc1_bias_max} {fc1_bias_min} "  # Max and min values of fc1_bias

    bit_string_temp += f"{fc2_weight_max} {fc2_weight_min} "  # Max and min values of fc2_weight
    bit_string_temp += f"{fc2_bias_max} {fc2_bias_min} "  # Max and min values of fc2_bias

    bit_string_temp += f"{fc3_weight_max} {fc3_weight_min} "  # Max and min values of fc3_weight
    bit_string_temp += f"{fc3_bias_max} {fc3_bias_min}"  # Max and min values of fc3_bias

    # Convert the quantized weights and biases to integers and append them to the bit string
    bit_string += " " + " ".join(map(str, fc1_weight_quantized.ravel().astype(int))) + " "  # FC1 weight quantized values
    bit_string += " ".join(map(str, fc1_bias_quantized.astype(int)) ) + " "  # FC1 bias quantized values

    bit_string += " ".join(map(str, fc2_weight_quantized.ravel().astype(int))) + " "  # FC2 weight quantized values
    bit_string += " ".join(map(str, fc2_bias_quantized.astype(int)) ) + " "  # FC2 bias quantized values

    bit_string += " ".join(map(str, fc3_weight_quantized.ravel().astype(int)) ) + " "  # FC3 weight quantized values
    bit_string += " ".join(map(str, fc3_bias_quantized.astype(int)) )  # FC3 bias quantized values

    # while len(bit_string)<12000:
    #     bit_string+=" "
    if(client_num == 1):
        with open("D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\temp_client1.txt",'w') as file:
            file.write(bit_string_temp)
    elif(client_num == 2):
        with open("D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\temp_client2.txt",'w') as file:
            file.write(bit_string_temp)  
    else:
        with open("D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\temp_server.txt",'w') as file:
            file.write(bit_string_temp)
    
    bit_string = floatingPoint_to_binary(bit_string, num_bits=16)
    hex_string = hexadecimal_converter(bit_string)
    hex_string = hex_string.rstrip()
    # with open("D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client2_server.txt",'w') as input_file:
    #     input_file.write(hex_string)
    print((hex_string))
    return (hex_string)

# model = main_quantize('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client2_model.txt',2)
# model_param = dq.main_dequnatize('D:\\Arjun Workspace\\B.Tech-Project---Federated-Learning\\Final_Demo\\testing_onMNIST\\Textfile\\client2_server.txt',2)
# print(model_param)



