import struct
import math

import numpy as np
import matplotlib.pyplot as plt

# Input the file information
# ----------------------------------------------------------
filename = "L75n64.yicg" 


# ----------------------------------------------------------

f = filename

yicg = open(f, 'rb')
print('Opening,,, ')

shtsize = struct.calcsize('h') # 2 byte
intsize = struct.calcsize('i') # 4 byte
fltsize = struct.calcsize('f') # 4 byte
dblsize = struct.calcsize('d') # 8 byte
Headersize = 26


dummy = yicg.read(intsize)

# SimulationBox
dummy = yicg.read(dblsize*3)
SimulationBoxmin = struct.unpack('3d',dummy)
print('Simulation Box min', SimulationBoxmin)

dummy1 = yicg.read(dblsize*3)
SimulationBoxmax = struct.unpack(str(3)+'d',dummy1)
print('Simulation Box max', SimulationBoxmax)

#SimulationBoxActive
dummy = yicg.read(dblsize*3)
SimulationBoxActivemin = struct.unpack('3d',dummy)
print('SimulationBoxActive min', SimulationBoxActivemin)


dummy = yicg.read(dblsize*3)
SimulationBoxActivemax = struct.unpack('3d',dummy)
print('SimulationBoxActive max', SimulationBoxActivemax)

#DisplayingBox
dummy = yicg.read(dblsize*3)
DisplayingBoxmin = struct.unpack('3d',dummy)
print('DisplayingBox min', DisplayingBoxmin)


dummy = yicg.read(dblsize*3)
DisplayingBoxmax = struct.unpack('3d',dummy)
print('DisplayingBox max', DisplayingBoxmax)

#MeshUnitLength
dummy = yicg.read(dblsize)
MeshUnitLength = struct.unpack('d',dummy)

#MeshWInflow
dummy = yicg.read(dblsize*5)
MeshWInflow = struct.unpack('5d',dummy)
print('MeshWInflow =', MeshWInflow)

#Gamma
dummy = yicg.read(dblsize)
Gamma = struct.unpack('d',dummy)

#Resolution
dummy = yicg.read(dblsize)
Resolution = struct.unpack('d',dummy)
print('Resolution =', Resolution)

ParticleNum = 64**3
dummy = yicg.read(dblsize*5*ParticleNum)
Primitive_var = struct.unpack(str(5*ParticleNum)+'d',dummy)
Primitive_var = np.array(Primitive_var)

Primitive_var = Primitive_var.reshape((ParticleNum, 5))
rho = Primitive_var[:,0]
vx = Primitive_var[:,1]
vy = Primitive_var[:,2]
vz = Primitive_var[:,3]
T = Primitive_var[:,4]


print('unique velocity value of Vx',np.unique(vx))
print('len(np.unique(vx)) =', len(np.unique(vx)))