import numpy as np
def quantize():
    array = np.random.rand(20, 1)
    max_value = np.amax(array)
    min_value = np.amin(array)
    scaled = (array - min_value) / (max_value - min_value)
    rounded = np.round(scaled * (2 ** bits - 1))
    bits = rounded.tobytes()
    return bits


quantized = quantize()
print(quantized)