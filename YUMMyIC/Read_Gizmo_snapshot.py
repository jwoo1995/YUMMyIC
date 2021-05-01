import struct
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.animation as animation

def snapshot(alpha,snapshotnumber):
    
    # Input the file information
    # ----------------------------------------------------------
    #base = "./"
    #project = "./"
    filename = 'snapshot_0'+str(snapshotnumber)
    # Input the file information
    # # ----------------------------------------------------------
    # #base = "./"
    # ----------------------------------------------------------
    f = filename
    # Open the GADGET initial condition
    GADGET = open(f, 'rb')
    shtsize = struct.calcsize('h') # 2 byte (short)
    intsize = struct.calcsize('i') # 4 byte (integer)
    fltsize = struct.calcsize('f') # 4 byte (single precision)
    dblsize = struct.calcsize('d') # 8 byte (double precision)
    # =========================================== 01. read header ==================================================
    header_init_dummy = GADGET.read(intsize)
    dummy_npart = GADGET.read(intsize * 6)
    npart = struct.unpack('6i' ,dummy_npart)
    npart = list(npart) # number of partices for each particle type
    print('npart =',npart)
    dummy_massarr = GADGET.read(dblsize * 6)
    massarr = struct.unpack('6d', dummy_massarr)
    dummy_time = GADGET.read(dblsize)
    time = struct.unpack('d', dummy_time)
    
    dummy_redshift = GADGET.read(dblsize)
    redshift = struct.unpack('d', dummy_redshift)
    redshift = np.array(redshift)
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


    print('np.min(x) =',np.min(x))
    print('np.min(y) =', np.min(y))
    print('Min Z , Max Z =', np.min(z), ',',np.max(z))

    # Add a new frame
    boxsize = 100
    z_slicewidth = boxsize/alpha
    L = boxsize
    w = np.where(np.abs(x-L/2) <= z_slicewidth)
    redshift = round(redshift[0],1)

    fig = plt.figure(figsize=[10,10])
    ax = plt.axes(xlim=(0,L), ylim=(0,L))
    ax.patch.set_facecolor('black')
    ax.scatter(y[w], z[w], s=0.1, marker='.', c='white', alpha=1.0)
    fig.savefig('z='+str(redshift)+'GIZMODMonlyL100n32.png')


    #plt.rc('font', family='Ubuntu')
    #ax.set_aspect('equal')
    plt.tight_layout()
    #W = struct.unpack(str(5*meshnum)+'d', meshWs)
    #W = np.array(W)
    #W = W.reshape(meshnum, 5)
    #rho = W[:,0]
    #print(np.min(rho), np.max(rho))

    return(fig)

for i in range(10,16):
    fig = snapshot(2,i)


