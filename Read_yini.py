import struct
import numpy as np

f = open("L75n64.yini","rb")
cosmo_yini = np.zeros(0)
for line in f.readlines():
    cosmo_yini = np.append(cosmo_yini, line)
print(cosmo_yini)