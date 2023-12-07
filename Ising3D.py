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


def random_lattice3D(Nx, Ny, Nz) :

    lattice = np.array([[[1 if np.random.uniform()<0.5 else -1 for i in range(Nx)] for j in range(Ny)] for k in range(Nz)])

    return lattice


def energy3D(config, J, Gamma, beta) :

    Nx,Ny,Nz = np.shape(config)
    K = beta*J
    Km = 0.5*np.log(1/np.tanh(beta*Gamma/Nz))

    H1 = 0
    H2 = 0

    for k in range(Nz) :
        for i in range(Nx) :
            for j in range(Ny) :
                H1 += config[i,j,k]*(config[(i+1)%Nx,j,k] + config[i,(j+1)%Ny,k])
                H2 += config[i,j,k]*config[i,j,(k+1)%Nz]

    return -K/Nz*H1 - Km*H2


def delta_energy_flip3D(config, a, b, c, J, Gamma, beta) :

    Nx,Ny,Nz = np.shape(config)
    K = beta*J
    Km = 0.5*np.log(1/np.tanh(beta*Gamma/Nz))

    dE = 2*config[a,b,c]*(K/Nz*(config[(a+1)%Nx,b,c] + config[a,(b+1)%Ny,c] + config[(a-1)%Nx,b,c] + config[a,(b-1)%Ny,c]) + Km*(config[a,b,(c+1)%Nz] + config[a,b,(c-1)%Nz]))
    
    return dE


def MC_step3D(config, J, Gamma, beta) : # flips one spins and accept the new config with a certain probability

    Nx,Ny,Nz = np.shape(config)

    # flip one randomly chosen site

    a = np.random.randint(low=0,high=Nx)
    b = np.random.randint(low=0,high=Ny)
    c = np.random.randint(low=0,high=Nz)

    dE = delta_energy_flip3D(config,a,b,c,J,Gamma,beta)

    if dE<0 :
        config[a,b,c] *= -1
        return config,dE,True

    else :

        p = np.exp(-beta*dE)

        if np.random.uniform() < p: # accept candidate if alpha>=1 or accept candidate with proba alpha if 0<alpha<1

            config[a,b,c] *= -1

            return config,dE,True
        
        else: 
            
            return config,dE,False
