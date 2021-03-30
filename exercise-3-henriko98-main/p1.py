# Problem 1: Stationary temperature field in beam cross-section"
import numpy as np
import scipy 
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from math import sinh
Ttop = 100
Tleft = 50
N = 4
l = 1.0

def plotSurfaceNeumannDirichlet(Temp, Ttop, Tleft, l, N, nxTicks=4, nyTicks=4):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    x = np.linspace(0, l, N + 1)
    y = np.linspace(0, l, N + 1)
    X, Y = np.meshgrid(x, y)
    T = np.zeros_like(X)
    T[-1,:] = Ttop
    T[:,0] = Tleft
    k = 1
    for j in range(N):
        T[j,1:] = Temp[N*(k-1):N*k]
        k+=1
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(0, Ttop)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T [$^o$C]')
    xticks=np.linspace(0.0, l, nxTicks+1)
    ax.set_xticks(xticks)
    yticks=np.linspace(0.0, l, nyTicks+1)
    ax.set_yticks(yticks)
    plt.show()

def solveneumann(Tleft,Ttop,l):
    main_diag = -4*np.ones(N*N)

    sup_diag1 = np.ones(N*N-1)
    sup_diag1[N-1::N]=0

    sub_diag1 = np.ones(N*N-1)
    sub_diag1[N-2::N] = 2
    sub_diag1[N-1::N] = 0

    sup_diagN = np.ones(N*N-N)
    sup_diagN[0:N] = 2

    sub_diagN = np.ones(N*N-N)

    #lage matrise
    A = scipy.sparse.diags([main_diag,sup_diag1,sub_diag1,sup_diagN,sub_diagN],[0,1,-1,N,-N], format = 'csc')

    #RHS
    b = np.zeros(N*N)
    print(b)
    b[0::N] += -Tleft
    b[-N:] += -Ttop

    Temp = scipy.sparse.linalg.spsolve(A,b)

    return Temp

N_list = [10,20,30]

for N in N_list:
    print("N = {}".format(N))
    
    Temp = solveneumann(Tleft,Ttop,l)

    plotSurfaceNeumannDirichlet(Temp, Ttop, Tleft, l, N)