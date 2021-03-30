# Problem 2: Transient temperature field in one-dimensional beam"

import numpy as np
import scipy 
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from math import sinh
Tbot = 0
Tleft = 100
Tright = 0
N = 4
l = 1.0
r = 0.00025

def plotSurfaceNeumannDirichlet(Temp, Tbot, Tleft, l, N, nxTicks=4, nyTicks=4):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    x = np.linspace(0, l, N + 1)
    y = np.linspace(0, l, N + 1)
    X, Y = np.meshgrid(x, y)
    T = np.zeros_like(X)
    #T[-1,:] = Ttop
    T[:,0] = Tleft
    k = 1
    for j in range(N):
        T[j,1:] = Temp[N*(k-1):N*k]
        k+=1
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(0, Tleft)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T [$^o$C]')
    xticks=np.linspace(0.0, l, nxTicks+1)
    ax.set_xticks(xticks)
    yticks=np.linspace(0.0, l, nyTicks+1)
    ax.set_yticks(yticks)
    plt.show()

def solveFTCS(Tleft,Tbot, Tright,l, N):
    main_diag = (1-2*r)*np.ones(N*N)
    main_diag[-1:0] = 1

    sup_diag1 = r*np.ones(N*N-1)
    sup_diag1[0]=0

    sub_diag1 = np.ones(N*N-1)
    sub_diag1[-1] = 0

    #lage matrise
    A = scipy.sparse.diags([main_diag,sup_diag1,sub_diag1],[0,1,-1], format = 'csc')

    #RHS
    u = np.zeros(N*N)

    u[0::N] += -Tleft
    
    Temp = scipy.sparse.linalg.spsolve(A,u)

    return Temp

N_list = [10,20,30]

for N in N_list:
    print("N = {}".format(N))
    
    Temp = solveFTCS(Tleft,Tbot, Tright,l, N)

    plotSurfaceNeumannDirichlet(Temp, Tbot, Tleft, l, N)
