import struct
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.mplot3d import Axes3D

# Input the file information
# ----------------------------------------------------------
base = "./"
filename = "snapshot" # as GADGET format

yopname = "ics_gadget64_fullhydro.yop"
yicgname = "ics_gadget64_fullhydro.yicg"
# ----------------------------------------------------------

f = base + filename

# Calculate the unit scale
# ----------------------------------------------------------
Gamma = 5.0/3.0
mu = 1.0
X = 1.0 # mu = 4 / (6X + Y + 2) for a fully ionized gas
scale_m = 1.989e40 # 10^10 Msun in kilograms
scale_l = 3.086e22 # Mpc in meters

# z ~ 50
#igmrho = 5e-7 # H/cm^3
#igmtemp = 30 # Kelvins

igmrho = 1e-10
igmtemp = 1e10

cosmounit = 1

# ----------------------------------------------------------

scale_d = scale_m / scale_l / scale_l / scale_l # unit density -> kg/m^3
scale_d_cgs = scale_d * 1000.0 / 100.0 / 100.0 / 100.0 # unit density -> g/cm^3

G = 6.67408e-11 # in mks unit
scale_t = 1.0 / math.sqrt(G * scale_d) # unit time -> seconds (to make G = 1)
print('time scale = ', scale_t, ' s')
scale_t_inGyr = scale_t / 31536000 / 1e9 # unit time -> Gyrs
print('time scale = ', scale_t_inGyr, ' Gyrs')

scale_v = scale_l / scale_t # unit velocity -> m/s
scale_v_cgs = scale_v * 100.0 # unit velocity -> cm/s

R = 8314.4622 # in J kg^-1 K^-1 = (m/s)^2 K^-1
scale_temp = scale_v * scale_v / R * mu # unit temperature (T/mu) -> Kelvins (to make R = 1)
print('temperature scale = ', scale_temp)
scale_e = scale_v * scale_v / 1e3 / 1e3 # unit energy <(unit velocity)^2> -> (km/s)^2

igmrho = 1.67e-24 * igmrho / X # in g/cm^3
igmrho = igmrho / scale_d_cgs # as a simulation density unit

igmtemp = igmtemp / scale_temp # as a simulation temperature unit
igmpress = igmrho * igmtemp # as a simulation pressure unit

MeshWIGM = (igmrho, 0.0, 0.0, 0.0, igmpress)
print('IGM properties = ', MeshWIGM)
print('')

# Simulation setting
# ----------------------------------------------------------
SimulationBoxActiveMin = np.array([0.0,0.0,0.0]) # Only cube available
SimulationBoxActiveMax = np.array([100.0,100.0,100.0])
Resolution = 128
MeshUnitLength = 100.0/Resolution
# ----------------------------------------------------------

offset = 0
SimulationBoxMin = SimulationBoxActiveMin - offset * MeshUnitLength
SimulationBoxMax = SimulationBoxActiveMax + offset * MeshUnitLength
DisplayingBoxMin = SimulationBoxActiveMin * 1.0
DisplayingBoxMax = SimulationBoxActiveMax * 1.0
MeshWInflow = MeshWIGM

# Open the GADGET initial condition
GADGET = open(f, 'rb')
shtsize = Howtouse_struct.calcsize('h') # 2 byte
intsize = Howtouse_struct.calcsize('i') # 4 byte
fltsize = Howtouse_struct.calcsize('f') # 4 byte
dblsize = Howtouse_struct.calcsize('d') # 8 byte

dummy = GADGET.read(intsize)
dummy = GADGET.read(intsize * 6)
npart = Howtouse_struct.unpack('6i', dummy)
npart = list(npart)
dummy = GADGET.read(dblsize * 6)
massarr = Howtouse_struct.unpack('6d', dummy)
dummy = GADGET.read(dblsize)
time = Howtouse_struct.unpack('d', dummy)
dummy = GADGET.read(dblsize)
redshift = Howtouse_struct.unpack('d', dummy)
scale_factor = 1.0 / (1+redshift[0])

left = 256 - 6 * 4 - 6 * 8 - 8 - 8
dummy = GADGET.read(left)
dummy = GADGET.read(intsize)

print("Npart =", npart)
print("massarr =", massarr)
print("Time =", time)
print("Redshift =", redshift)
print("")

N = sum(npart)

# Read positions
dummy = GADGET.read(intsize)
dummy = GADGET.read(fltsize * 3 * N)
pos = Howtouse_struct.unpack(str(3 * N) + 'f', dummy)
pos = np.array(pos)
pos = pos.reshape((N,3)) 
dummy = GADGET.read(intsize)

x = pos[:,0]
y = pos[:,1]
z = pos[:,2]

# Read velocities
dummy = GADGET.read(intsize)
dummy = GADGET.read(fltsize * 3 * N)
vel = Howtouse_struct.unpack(str(3 * N) + 'f', dummy)
vel = np.array(vel)
vel = vel.reshape((N,3))
dummy = GADGET.read(intsize)

vx = vel[:,0]
vy = vel[:,1]
vz = vel[:,2]

# Read IDs
dummy = GADGET.read(intsize)
dummy = GADGET.read(intsize * N)
iden = Howtouse_struct.unpack(str(N) + 'i', dummy)
dummy = GADGET.read(intsize)

# Read masses
w = np.where((np.array(npart) > 0) & (np.array(massarr) == 0))
Nwithmass = np.array(npart)
Nwithmass = Nwithmass[w]
Nwithmass = sum(Nwithmass)

if Nwithmass > 0:
    dummy = GADGET.read(intsize)
    dummy = GADGET.read(fltsize * Nwithmass)
    mass = Howtouse_struct.unpack(str(Nwithmass) + 'f', dummy)
    dummy = GADGET.read(intsize)

# Read density & internal energies
NGas = npart[0]

if NGas > 0:
    dummy = GADGET.read(intsize)
    dummy = GADGET.read(fltsize * NGas)
    u = Howtouse_struct.unpack(str(NGas) + 'f', dummy)
    dummy = GADGET.read(intsize)
     
    dummy = GADGET.read(intsize)
    dummy = GADGET.read(fltsize * NGas)
    rho = Howtouse_struct.unpack(str(NGas) + 'f', dummy)
    dummy = GADGET.read(intsize)

GADGET.close()

E1 = sum(np.array(vx) ** 2 + np.array(vy) ** 2 + np.array(vz) ** 2)
print('E = ', E1)
print('N = ', N)
print('<V^2> = ', E1/N)
print("")

# Plot the GADGET initial condition
range1 = 0
range2 = npart[0]
range3 = npart[0]
range4 = N

## SPH + Background distribution
plt.figure(1)
plt.rc('font', family='Ubuntu')
plt.plot(x[range1:range2], y[range1:range2], 'k.', ms=0.4)
plt.axis([0,100,0,100])
plt.axes().set_aspect('equal')

for j in range(Resolution):
    xlist = (np.arange(Resolution+1) + 0.5) * MeshUnitLength
    ylist = np.zeros(Resolution+1) + (j + 0.5) * MeshUnitLength
    plt.plot(xlist, ylist, 'k.', ms=0.4)

plt.tight_layout()
plt.show()

# print(np.mean(rho))
# print(np.mean((Gamma-1.0) * (np.array(u) / scale_e)))

#--------------------------------particle--------------------------------

# Write the header information
yop = open(yopname, 'wb')

mHeaders = 0
dummy = Howtouse_struct.pack('i', mHeaders)
yop.write(dummy)

nparticle = 0
# Write positions
for i in range(npart[0], npart[0]+npart[1]):
    xtemp = x[i] * 3.086e22 / scale_l # Mpc -> unit length
    ytemp = y[i] * 3.086e22 / scale_l
    ztemp = z[i] * 3.086e22 / scale_l 
   
    if (xtemp == 0): xtemp += 1e-6
    if (ytemp == 0): ytemp += 1e-6
    if (ztemp == 0): ztemp += 1e-6

    dummy = Howtouse_struct.pack('3d', xtemp, ytemp, ztemp)
    yop.write(dummy)
    nparticle = nparticle + 1

# Write velocities
for i in range(npart[0], npart[0]+npart[1]):
    if (cosmounit == 1):
        vxtemp = math.sqrt(scale_factor) * scale_factor * vx[i] * 1e3 / scale_v # (km/s) -> (m/s) -> unit velocity
        vytemp = math.sqrt(scale_factor) * scale_factor * vy[i] * 1e3 / scale_v
        vztemp = math.sqrt(scale_factor) * scale_factor * vz[i] * 1e3 / scale_v
    else:
        vxtemp = vx[i] * 1e3 / scale_v # (km/s) -> (m/s) -> unit velocity
        vytemp = vy[i] * 1e3 / scale_v
        vztemp = vz[i] * 1e3 / scale_v
    dummy = Howtouse_struct.pack('3d', vxtemp, vytemp, vztemp)
    yop.write(dummy)

ke = (((math.sqrt(scale_factor) *vx[npart[0]:]*1e3/scale_v)**2 + (math.sqrt(scale_factor) *vy[npart[0]:]*1e3/scale_v)**2 + (math.sqrt(scale_factor) *vz[npart[0]:]*1e3/scale_v)**2) * 0.5)
print(np.mean(ke))

# Write masses
for i in range(npart[0], npart[0]+npart[1]): # in the zoom-in region
    mtemp = massarr[1] * 1.989e40 / scale_m # 10^10 Msun -> unit mass
    dummy = Howtouse_struct.pack('d', mtemp)
    yop.write(dummy)

yop.close()

#----------------------------------gas-----------------------------------

# Write the header information
yicg = open(yicgname, 'wb')

mHeaders = 26
dummy = Howtouse_struct.pack('i', mHeaders)
yicg.write(dummy)

dummy = Howtouse_struct.pack('3d', *SimulationBoxMin)
yicg.write(dummy)
dummy = Howtouse_struct.pack('3d', *SimulationBoxMax)
yicg.write(dummy)
dummy = Howtouse_struct.pack('3d', *SimulationBoxActiveMin)
yicg.write(dummy)
dummy = Howtouse_struct.pack('3d', *SimulationBoxActiveMax)
yicg.write(dummy)
dummy = Howtouse_struct.pack('3d', *DisplayingBoxMin)
yicg.write(dummy)
dummy = Howtouse_struct.pack('3d', *DisplayingBoxMax)
yicg.write(dummy)
dummy = Howtouse_struct.pack('d', MeshUnitLength)
yicg.write(dummy)
dummy = Howtouse_struct.pack('5d', *MeshWInflow)
yicg.write(dummy)
dummy = Howtouse_struct.pack('d', Gamma)
yicg.write(dummy)
dummy = Howtouse_struct.pack('d', Resolution)
yicg.write(dummy)

# Calculate pressures
tlist = (Gamma-1.0) * np.array(u) / scale_e
plist = np.array(rho) * (1.989e40 / scale_m) / ((3.086e22 / scale_l) ** 3) * tlist

ke = ((math.sqrt(scale_factor) *np.array(vx[:npart[0]]))**2 + (math.sqrt(scale_factor) *np.array(vy[:npart[0]]))**2 + (math.sqrt(scale_factor) *np.array(vz[:npart[0]]))**2) * 0.5
ke2 = ((math.sqrt(scale_factor) *np.array(vx[:npart[0]])*1e3/scale_v)**2 + (math.sqrt(scale_factor) *np.array(vy[:npart[0]])*1e3/scale_v)**2 + (math.sqrt(scale_factor) *np.array(vz[:npart[0]])*1e3/scale_v)**2) * 0.5
print(np.mean(ke))
print(np.mean(np.array(u)))
print(np.mean(ke2))
print(np.mean(np.array(u)/scale_e))

## u can be calculated again such as u = p / rho / (Gamma-1.0) when u, p, rho are in simulation units

minrho = np.max(np.array(rho))
maxrho = 0
# Write densities, velocities, pressures
nmesh = 0
for i in range(npart[0]):
    nmesh += 1
    rhotemp = rho[i] * 1.989e40 / scale_m / (3.086e22 / scale_l) ** 3 # 10^10 Msun / kpc^3 -> unit density
    if (rhotemp > maxrho): maxrho = rhotemp
    if (rhotemp < minrho): minrho = rhotemp
    if (cosmounit == 1):
        vxtemp = math.sqrt(scale_factor) * scale_factor * vx[i] * 1e3 / scale_v # (km/s) -> (m/s) -> unit velocity
        vytemp = math.sqrt(scale_factor) * scale_factor * vy[i] * 1e3 / scale_v
        vztemp = math.sqrt(scale_factor) * scale_factor * vz[i] * 1e3 / scale_v
    else:
        vxtemp = vx[i] * 1e3 / scale_v # (km/s) -> (m/s) -> unit velocity
        vytemp = vy[i] * 1e3 / scale_v
        vztemp = vz[i] * 1e3 / scale_v
    ptemp = plist[i]

    dummy = Howtouse_struct.pack('5d', rhotemp, vxtemp, vytemp, vztemp, ptemp)
    yicg.write(dummy)

# Write positions
for i in range(npart[0]):
    xtemp = x[i] # simulation units already
    ytemp = y[i]
    ztemp = z[i]
    
    if (xtemp == 0): xtemp += 2e-6
    if (ytemp == 0): ytemp += 2e-6
    if (ztemp == 0): ztemp += 2e-6
    
    dummy = Howtouse_struct.pack('3d', xtemp, ytemp, ztemp)
    yicg.write(dummy)

# Write mesh types
for i in range(npart[0]):
    dummy = Howtouse_struct.pack('i', 1) # Normal
    yicg.write(dummy)

yicg.close()

plt.figure(2)
plt.rc('font', family='Ubuntu')
plt.axes().set_aspect('equal')

print(nparticle, nmesh)
print(np.min((x[0:npart[0]])), np.max((x[0:npart[0]])), np.min((y[0:npart[0]])), np.max((y[0:npart[0]])), np.min((z[0:npart[0]])), np.max((z[0:npart[0]])))
print(np.min((x[npart[0]:npart[0]+npart[1]])), np.max((x[npart[0]:npart[0]+npart[1]])), np.min((y[npart[0]:npart[0]+npart[1]])), np.max((y[npart[0]:npart[0]+npart[1]])), np.min((z[npart[0]:npart[0]+npart[1]])), np.max((z[npart[0]:npart[0]+npart[1]])))

plt.plot((x[0:npart[0]]), (z[0:npart[0]]), 'r.', ms=0.4)
#plt.plot((x[0:]), (y[0:]), 'r.', ms=0.4)
plt.plot((x[npart[0]:npart[0]+npart[1]]), (z[npart[0]:npart[0]+npart[1]]), 'k.', ms=0.5)
#plt.plot((x[npart[0]:]), (y[npart[0]:]), 'k.', ms=0.5)

plt.axis([0,100,0,100])

plt.show()

#plt.savefig('disk_IC.png')

#------------------------------------------------------------------------
