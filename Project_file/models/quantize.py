import numpy as np
def quantize(a):
    array = a
    max_value = np.amax(array)
    min_value = np.amin(array)
    scaled = (array - min_value) / (max_value - min_value)
    bits= 5
    rounded = np.round(scaled * (2 ** bits - 1))
    bits = rounded.tobytes()
    return bits

a = [1,2,3,4,5,6,7,8,9,10]
quantized = quantize(a)
print(quantized)

