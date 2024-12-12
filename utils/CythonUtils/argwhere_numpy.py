import numpy as np

objmask = np.random.randint(0, 1024, size=(1024, 1024))
obc = 0

print(objmask, obc)
print((objmask == obc).shape)
yx = np.argwhere(objmask == obc)

print(yx)
print(yx.shape)