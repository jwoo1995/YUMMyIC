import struct
import math
import sys
import numpy as np
#from objsize import get_deep_size
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.mplot3d import Axes3D

# Input the file information
# ----------------------------------------------------------
#base = "./"
filename = "DMonly_L100n32.dat"  # as GADGET format

yopname = "DMonly_L100n32.yop"
yicgname = "DMonly_L100n32.yicg"
# ----------------------------------------------------------

f = filename


# Calculate the unit scale
# ----------------------------------------------------------
Gamma = 5.0 /3.0
mu = 1.0
X = 1.0 # mu = 4 / (6X + Y + 2) for a fully ionized gas
scale_m = 1.989e40 # 10^10 Msun in kilograms
scale_l = 3.086e22 # Mpc in meters

# z ~ 50
# igmrho = 5e-7 # H/cm^3
# igmtemp = 30 # Kelvins

igmrho = 1e-10
igmtemp = 1e4

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
SimulationBoxActiveMin = np.array([0.0 ,0.0 ,0.0]) # Only cube available
SimulationBoxActiveMax = np.array([100.0 ,100.0 ,100.0])
Resolution = 32
MeshUnitLength = 100.0 /Resolution
# ----------------------------------------------------------

offset = 4
SimulationBoxMin = SimulationBoxActiveMin - offset * MeshUnitLength
SimulationBoxMax = SimulationBoxActiveMax + offset * MeshUnitLength
DisplayingBoxMin = SimulationBoxActiveMin
DisplayingBoxMax = SimulationBoxActiveMax
MeshWInflow = MeshWIGM

# Open the GADGET initial condition
GADGET = open(f, 'rb')
shtsize = struct.calcsize('h') # 2 byte (short)
intsize = struct.calcsize('i') # 4 byte (integer)
fltsize = struct.calcsize('f') # 4 byte (single precision)
dblsize = struct.calcsize('d') # 8 byte (double precision)
print('shtsize =', shtsize)
print('intsize =',intsize)
print('fltsize =',fltsize)
print('dblsize =', dblsize)


# =========================================== 01. read header ==================================================
header_init_dummy = GADGET.read(intsize)
print('header size =',struct.unpack('i',header_init_dummy))
dummy_npart = GADGET.read(intsize * 6)
npart = struct.unpack('6i' ,dummy_npart)
npart = list(npart) # number of partices for each particle type
print('npart =',npart)


dummy_massarr = GADGET.read(dblsize * 6)
massarr = struct.unpack('6d', dummy_massarr)
print('massarr =', massarr)

dummy_time = GADGET.read(dblsize)
time = struct.unpack('d', dummy_time)
print('time in seconds =', time)


dummy_redshift = GADGET.read(dblsize)
redshift = struct.unpack('d', dummy_redshift)
scale_factor = 1.0 / (1+redshift[0])
print('redshift =', redshift)


left = 256 - 6 * 4 - 6 * 8 - 8 - 8 # left file size in the header after reading above
dummy_left = GADGET.read(left)
header_final_dummy = GADGET.read(intsize)

N = sum(npart) # total number of particles


# ===================================== 02. Read positions ========================================
pos_init_dummy = GADGET.read(intsize)
print('position block size =', struct.unpack('i',pos_init_dummy), '=', fltsize * 3 * N)
dummy_pos = GADGET.read(fltsize * 3 * N)
pos = struct.unpack(str( 3 *N ) +'f', dummy_pos)

pos = np.array(pos)
pos = pos.reshape((N ,3))
pos_final_dummy = GADGET.read(intsize)

x = pos[: ,0]
y = pos[: ,1]
z = pos[: ,2]

# print('(min of x, max of x) =', np.min(x), ',', np.max(x))
# print('(min of y, max of y) =', np.min(y), ',', np.max(y))
# print('(min of z, max of z) =', np.min(z), ',', np.max(z))

# Read velocities
vel_init_dummy = GADGET.read(intsize)
print('velocity block size =', struct.unpack('i',vel_init_dummy),'=', fltsize * 3 * N)

dummy_vel = GADGET.read(fltsize * 3 * N)
vel = struct.unpack(str( 3 *N ) +'f', dummy_vel)

vel = np.array(vel)
vel = vel.reshape((N ,3))

vx = vel[: ,0]
vy = vel[: ,1]
vz = vel[: ,2]

# print('(min of x_vel, max of x_vel) =', np.min(vx), ',', np.max(vx))
# print('(min of y_vel, max of y_vel) =', np.min(vy), ',', np.max(vy))
# print('(min of z_vel, max of z_vel) =', np.min(vz), ',', np.max(vz))

vel_final_dummy = GADGET.read(intsize)


# Read IDs
ID_init_dummy = GADGET.read(intsize)
print('ID block size =', struct.unpack('i',ID_init_dummy),'=', intsize * N)

dummy_id = GADGET.read(intsize * N)
iden = struct.unpack(str(N ) +'i', dummy_id) # iden = ID of particles

# print('number of particels in IC file =',len(iden))

ID_final_dummy = GADGET.read(intsize)

# Read masses
# dummy = GADGET.read(intsize)
w = np.where((np.array(npart) > 0) & (np.array(massarr) == 0))
Nwithmass = np.array(npart)
Nwithmass = Nwithmass[w]
Nwithmass = sum(Nwithmass)


if Nwithmass > 0:
    masses_init_dummy = GADGET.read(intsize)
    print('masses block size =', struct.unpack('i', masses_init_dummy), '=', fltsize * Nwithmass)
    dummy_varmass = GADGET.read(fltsize * Nwithmass)
    mass = struct.unpack(str(Nwithmass ) +'f', dummy_varmass)
    masses_final_dummy = GADGET.read(intsize)

# dummy = GADGET.read(intsize)
# Read internal energies
NGas = npart[0]
"""
internalE_init_dummy = GADGET.read(intsize)
print(internalE_init_dummy)
print('internalE block size =', struct.unpack('i',internalE_init_dummy),'=', fltsize * NGas)
dummy_internalE = GADGET.read(fltsize * NGas)
u = struct.unpack(str(NGas) + 'f', dummy_internalE)

internalE_final_dummy = GADGET.read(intsize)
print(GADGET.read(2))
density_init_dummy = GADGET.read(intsize)
print('density block size =', struct.unpack('i',density_init_dummy),'=', fltsize * NGas)


dummy_density = GADGET.read( fltsize * NGas)
rho = struct.unpack('if', dummy_density)
density_final_dummy = GADGET.read(intsize)
if NGas > 0:
    internalE_init_dummy = GADGET.read(intsize)
    dummy_internalE = GADGET.read(fltsize * NGas)
    u = struct.unpack(str(NGas ) +'f', dummy_internalE)
    internalE_final_dummy = GADGET.read(intsize)

    density_init_dummy = GADGET.read(intsize)
    dummy_density = GADGET.read(fltsize * NGas)
    rho = struct.unpack(str(NGas ) +'f', dummy_density)
    density_final_dummy = GADGET.read(intsize)
"""
# print('file size from header to internalE =', fltsize * NGas +  fltsize * Nwithmass + intsize * N + fltsize * 3 * N + fltsize * 3 * N + 256)
GADGET.close()



E1 = sum(np.array(vx) ** 2 + np.array(vy) ** 2 + np.array(vz) ** 2)
print('E = ', E1)
print('N = ', N)
print('<V^2> = ', E1 /N)
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
plt.axis([0 ,100 ,0 ,100])
plt.axes().set_aspect('equal')

for j in range(Resolution):
    xlist = (np.arange(Resolution +1) + 0.5) * MeshUnitLength
    ylist = np.zeros(Resolution +1) + (j + 0.5) * MeshUnitLength
    plt.plot(xlist, ylist, 'k.', ms=0.4)

plt.tight_layout()
plt.show()

# print(np.mean(rho))
# print(np.mean((Gamma-1.0) * (np.array(u) / scale_e)))

# --------------------------------particle--------------------------------

# Write the header information
yop = open(yopname, 'wb')

mHeaders = 0
dummy = struct.pack('i', mHeaders)
yop.write(dummy)

# Write positions
for i in range(npart[0], N):
    xtemp = x[i] * 3.086e22 / scale_l # Mpc -> unit length
    ytemp = y[i] * 3.086e22 / scale_l
    ztemp = z[i] * 3.086e22 / scale_l
    dummy = struct.pack('3d', xtemp, ytemp, ztemp)
    yop.write(dummy)

# Write velocities
for i in range(npart[0], N):

    vxtemp = math.sqrt(scale_factor) * scale_factor * vx[i] * 1e3 / scale_v # (km/s) -> (m/s) -> unit velocity
    vytemp = math.sqrt(scale_factor) * scale_factor * vy[i] * 1e3 / scale_v
    vztemp = math.sqrt(scale_factor) * scale_factor * vz[i] * 1e3 / scale_v
    dummy = struct.pack('3d', vxtemp, vytemp, vztemp)
    yop.write(dummy)

# Write masses
for i in range(npart[1]): # in the zoom-in region
    mtemp = massarr[1] * 1.989e40 / scale_m # 10^10 Msun -> unit mass
    dummy = struct.pack('d', mtemp)
    yop.write(dummy)

for i in range(Nwithmass): # in the coarse-resolution region
    mtemp = mass[i] * 1.989e40 / scale_m # 10^10 Msun -> unit mass
    dummy = struct.pack('d', mtemp)
    yop.write(dummy)

yop.close()

# ----------------------------------gas-----------------------------------

# Construct Oct-tree

## Level-0 (Background)
xcen0 = np.arange(- 1 *offset, Resolution +offset) + 0.5
xcen0 = xcen0 * MeshUnitLength + SimulationBoxActiveMin[0]
ycen0 = np.arange(- 1 *offset, Resolution +offset) + 0.5
ycen0 = ycen0 * MeshUnitLength + SimulationBoxActiveMin[1]
zcen0 = np.arange(- 1 *offset, Resolution +offset) + 0.5
zcen0 = zcen0 * MeshUnitLength + SimulationBoxActiveMin[2]
vol0 = MeshUnitLength ** 3

xyzgas = []
for i in range(len(xcen0)):
    for j in range(len(ycen0)):
        for k in range(len(zcen0)):
            xyzgas.append([xcen0[i] ,ycen0[j] ,zcen0[k] ,0]) # 0=background, 1=SPH

# Allocate hydrodynamics properties

## Initialize : list = all hydro cells, sph = sph particles
## This procedure is only available for N-body simulations
## All SPH particles are neglected here
xlist = np.array([x[0] for x in xyzgas])
ylist = np.array([x[1] for x in xyzgas])
zlist = np.array([x[2] for x in xyzgas])
rholist = np.zeros(len(xlist)) + MeshWIGM[0] # as units of 10^10 Msun / kpc^3
vxlist = np.zeros(len(xlist)) + MeshWIGM[1]
vylist = np.zeros(len(xlist)) + MeshWIGM[2]
vzlist = np.zeros(len(xlist)) + MeshWIGM[3]
plist = np.zeros(len(xlist)) + MeshWIGM[4]

# Write the header information
yicg = open(yicgname, 'wb')

mHeaders = 26
dummy = struct.pack('i', mHeaders)
yicg.write(dummy)

dummy = struct.pack('3d', *SimulationBoxMin)
yicg.write(dummy)
dummy = struct.pack('3d', *SimulationBoxMax)
yicg.write(dummy)
dummy = struct.pack('3d', *SimulationBoxActiveMin)
yicg.write(dummy)
dummy = struct.pack('3d', *SimulationBoxActiveMax)
yicg.write(dummy)
dummy = struct.pack('3d', *DisplayingBoxMin)
yicg.write(dummy)
dummy = struct.pack('3d', *DisplayingBoxMax)
yicg.write(dummy)
dummy = struct.pack('d', MeshUnitLength)
yicg.write(dummy)
dummy = struct.pack('5d', *MeshWInflow)
yicg.write(dummy)
dummy = struct.pack('d', Gamma)
yicg.write(dummy)
dummy = struct.pack('d', Resolution)
yicg.write(dummy)

# Write densities, velocities, pressures
for i in range(len(xlist)):
    rhotemp = rholist[i] # simulation units already
    vxtemp = vxlist[i]
    vytemp = vylist[i]
    vztemp = vzlist[i]
    ptemp = plist[i]
    dummy = struct.pack('5d', rhotemp, vxtemp, vytemp, vztemp, ptemp)
    yicg.write(dummy)

# Write positions
for i in range(len(xlist)):
    xtemp = xlist[i] # simulation units already
    ytemp = ylist[i]
    ztemp = zlist[i]
    dummy = struct.pack('3d', xtemp, ytemp, ztemp)
    yicg.write(dummy)

# Write mesh types
for i in range(len(xlist)):
    dummy = struct.pack('i', 1) # Normal
    yicg.write(dummy)

yicg.close()

plt.figure(2)
plt.rc('font', family='Ubuntu')
plt.axes().set_aspect('equal')

plt.plot(xlist, ylist, 'k.', ms=0.4)
plt.plot(x, y, 'r.', ms=0.5)

plt.axis([0 ,100 ,0 ,100])

plt.show()

# plt.savefig('disk_IC.png')

# ------------------------------------------------------------------------
