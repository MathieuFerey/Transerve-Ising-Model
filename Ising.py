import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import scipy.stats as ss
import scipy.linalg as lin
import scipy.sparse as sp

from tqdm import tqdm
from functools import partial

from plot_tools import *


def plot_lattice2(lattice, N, Ny) :
    
    fig = plt.figure()
    x = np.arange(0,N,1)
    y = np.arange(0,Ny,1)

    for i in range(N) :
        for j in range(Ny) :
            if lattice[i,j] == 1 :
                color = 'r'
            else :
                color = 'b'
            plt.scatter([x[i]],[y[j]],color=color)

    plt.show()


def plot_lattice(config, Nx, Ny) :

    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    x = range(Nx)
    y = range(Ny)
    X,Y = np.meshgrid(range(Nx),range(Ny))

    cmap = colors.ListedColormap(['navy', 'darkred'])

    ax.pcolormesh(X,Y,config,cmap=cmap)
    plt.show()




def random_lattice(N, Ny) :

    lattice = np.array([[1 if np.random.uniform()<0.5 else -1 for i in range(N)] for j in range(Ny)])

    return lattice


def energy(lattice, N, Ny, g, beta):

    a = beta*g/Ny
    gamma = -np.log(np.tanh(a))

    E = 0

    lattice_periodic = np.zeros((N+1,Ny+1))
    lattice_periodic[0:N,0:Ny] = lattice
    lattice_periodic[-1,:-1] = lattice[0]
    lattice_periodic[:-1,-1] = lattice[:,0]

    for i in range(N) :
        for j in range(Ny) :

            ei =  -lattice_periodic[i,j]*(Ny*gamma/beta*lattice_periodic[i,j+1]+lattice_periodic[i+1,j])
            E += ei

    return E


def calcEnergy(config, N, Ny, g) : # regular Ising
    
    start = time.time()
    energy = 0
    for i in range(N):
        for j in range(Ny):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%Ny] #+ config[(i-1)%N, j] + config[i,(j-1)%Ny]
            energy += -nb*S
    end = time.time()
    return g*energy, end-start


def delta_energy_flip(config, i, j, N, Ny, g) : # smarter computation of the energy after one flip

    return 2*g*(config[i,j])*(config[(i+1)%N,j] + config[i,(j+1)%Ny] + config[(i-1)%N,j] + config[i,(j-1)%Ny])


def MC_step(config, N, Ny, g, beta) : # flips one spins and accept the new config with a certain probability

    #Eold = energy(candidate_lattice,N,Ny,g,beta)
    #Eold,delay1 = calcEnergy(candidate_lattice,N,Ny,g)
    # flip one randomly chosen site

    a = np.random.randint(low=0,high=N)
    b = np.random.randint(low=0,high=Ny)
    #candidate_lattice[a,b] = -candidate_lattice[a,b]

    start = time.time()
    dE = delta_energy_flip(config,a,b,N,Ny,g)
    end = time.time()

    # compute the new energy

    #Enew = energy(candidate_lattice,N,Ny,g,beta)
    #Enew,delay2 = calcEnergy(candidate_lattice,N,Ny,g)

    delay = end-start

    if dE<0 :
        config[a,b] *= -1
        return config,delay

    else :

        p = np.exp(-beta*dE)
        
        if np.random.uniform() < p: # accept candidate if alpha>=1 or accept candidate with proba alpha if 0<alpha<1
            config[a,b] *= -1
            return config,delay

        else: return config,delay


def mag(config) :
    return np.sum(config)
        


# Computation time ========================================

'''
for n in [16,32,64,128,256,512]:


    lattice = random_lattice(n,n)
    delays = np.zeros(N_steps)

    for i in range(N_steps):
        lattice,delay = MC_step(lattice,n,n,g,beta)
        delays[i] = delay

    #plt.plot(range(N_steps),delays,marker='.',label='N='+str(n))
    plt.scatter([n],[np.mean(delays)],color='r')


#plt.xlabel('MC step')
plt.xlabel('lattice size')
plt.ylabel('Energy computation time')
plt.legend()
plt.show()

'''

# Animation ==========================================

'''
beta, g = 10, 1
N_steps = 10000

 
N, Ny = 16, 16

lattice = random_lattice(N,Ny)

fig,ax = plt.subplots()
ax.set_aspect('equal')
x = range(N)
y = range(Ny)
X,Y = np.meshgrid(range(N),range(Ny))

cmap = colors.ListedColormap(['navy', 'darkred'])

pm = ax.pcolormesh(X,Y,lattice,cmap=cmap)


def update(frame) :
    
    global lattice 
    plt.title(str(frame))
    lattice = MC_step(lattice,N,Ny,g,beta)[0]
    pm.set_array(lattice.ravel())

anim =  animation.FuncAnimation(fig,update,frames=N_steps,interval=0.01,repeat=False)

plt.show()
#anim.save('ising_ferro_16x16.gif')
'''
