import struct
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.animation as animation

def snapshot(alpha):
    # Input the file information
    # ----------------------------------------------------------
    #base = "./"
    #project = "./"
    filename = 'DMonlyL100n32_2105011104_0001_00000000'

    DMon = 1
    Gason = 0

    isyicg = 0

    time_scalefactor = 0
    boxsize = 100

    conv = 0 # -1: normal->zoomin coord. / +1: zoomin->normal coord.
    center = np.array([boxsize/2, boxsize/2, boxsize/2])

    # ----------------------------------------------------------

    #f = base + project + filename
    f = filename

    shtsize = struct.calcsize('h') # 2 byte
    intsize = struct.calcsize('i') # 4 byte
    fltsize = struct.calcsize('f') # 4 byte
    dblsize = struct.calcsize('d') # 8 byte

    # Open the yog file
    if (Gason == 1):
        if (isyicg == 1):
            yog = open(f+'.yicg', 'rb')
        else:
            yog = open(f+'.yog', 'rb')

        dummy = yog.read(intsize)
        headernum = struct.unpack('i',dummy)
        headernum = headernum[0]
        header = yog.read(dblsize * headernum)

        if (isyicg == 1):
            filesize = os.path.getsize(f+'.yicg')
            meshnum = filesize - headernum * 8 - 4
            meshnum = meshnum / (8 * 8 + 4)
        else:
            filesize = os.path.getsize(f+'.yog')
            meshnum = filesize - headernum * 8 - 4
            meshnum = meshnum / (12 * 8 + 4)

        meshnum = int(meshnum)

        if (isyicg == 1):
            meshWs = yog.read(meshnum * 5 * dblsize)
            meshXYZs = yog.read(meshnum * 3 * dblsize)
            meshTypes = yog.read(meshnum * intsize)
        else:
            meshWs = yog.read(meshnum * 5 * dblsize)
            volumes = yog.read(meshnum * dblsize)
            meshXYZs = yog.read(meshnum * 3 * dblsize)
            COMXYZs = yog.read(meshnum * 3 * dblsize)
            meshTypes = yog.read(meshnum * intsize)

        yog.close()

    # Open the yop file
    if (DMon == 1):
        yop = open(f+'.yop', 'rb')
        
        dummy = yop.read(intsize)
        headernum = struct.unpack('i',dummy)
        headernum = headernum[0]
        header = yop.read(dblsize * headernum)

        filesize = os.path.getsize(f+'.yop')
        partnum = filesize - headernum * 8 - 4
        partnum = partnum / (7 * 8)
        partnum = int(partnum)
        print('partnum =',partnum)

        XYZs = yop.read(partnum * 3 * dblsize)
        Vs = yop.read(partnum * 3 * dblsize)
        Masses = yop.read(partnum * dblsize)

        yop.close()

    # Read positions
    if (DMon == 1) & (Gason == 0):
        pos = struct.unpack(str(3*partnum)+'d', XYZs)
        pos = np.array(pos)
        pos = pos.reshape(partnum, 3)
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
    elif (DMon == 0) & (Gason == 1):
        pos = struct.unpack(str(3*meshnum)+'d', meshXYZs)
        pos = np.array(pos)
        pos = pos.reshape(meshnum, 3)
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
    elif (DMon == 1) & (Gason == 1):
        pos1 = struct.unpack(str(3*partnum)+'d', XYZs)
        pos1 = np.array(pos1)
        pos1 = pos1.reshape(partnum, 3)
        pos2 = struct.unpack(str(3*meshnum)+'d', meshXYZs)
        pos2 = np.array(pos2)
        pos2 = pos2.reshape(meshnum, 3)
        x1 = pos1[:,0]
        y1 = pos1[:,1]
        z1 = pos1[:,2]
        x2 = pos2[:,0]
        y2 = pos2[:,1]
        z2 = pos2[:,2]
        x = np.hstack([x1,x2])
        y = np.hstack([y1,y2])
        z = np.hstack([z1,z2])

    # Change the center of the box
    if conv != 0:
        offsetx = center[0] - 0.5
        offsety = center[1] - 0.5
        offsetz = center[2] - 0.5
        x = np.array(x) + offsetx * boxsize * conv
        y = np.array(y) + offsety * boxsize * conv
        z = np.array(z) + offsetz * boxsize * conv

        x = x + (x < 0) * boxsize
        x = x - (x > boxsize) * boxsize
        y = y + (y < 0) * boxsize
        y = y - (y > boxsize) * boxsize
        z = z + (z < 0) * boxsize
        z = z - (z > boxsize) * boxsize

    print('np.min(x) =',np.min(x))
    print('np.min(y) =', np.min(y))
    print('Min Z , Max Z =', np.min(z), ',',np.max(z))

    # Add a new frame
    L = 100 #boxlength
    z_slicewidth = L/alpha
    w = np.where(np.abs(x-L/2) <= z_slicewidth)

    fig = plt.figure(figsize=[10,10])
    ax = plt.axes(xlim=(0,L), ylim=(0,L))
    ax.patch.set_facecolor('black')
    ax.scatter(y[w], z[w], s=0.1, marker='.', c='white', alpha=1.0)


    #plt.rc('font', family='Ubuntu')
    #ax.set_aspect('equal')
    plt.tight_layout()
    #W = struct.unpack(str(5*meshnum)+'d', meshWs)
    #W = np.array(W)
    #W = W.reshape(meshnum, 5)
    #rho = W[:,0]
    #print(np.min(rho), np.max(rho))

    return(fig)

fig = snapshot(2)
fig.savefig('z=0.0YUMMyDMonlyL100n32_3.png')
