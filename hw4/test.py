import numpy as np

x = np.arange(30)
x2 = np.reshape(x, (2, 5,3))
print(x2)
# print(np.roll(x2, (2, 1), axis=(1, 0)))
# print(np.roll(x2, (2, 1), axis=(0, 1)))
print(np.roll(x2, 3, axis=1))