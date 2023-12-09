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
    Km = 0.5*Nz/beta*np.log(1/np.tanh(beta*Gamma/Nz))

    H1 = 0
    H2 = 0

    for k in range(Nz) :
        for i in range(Nx) :
            for j in range(Ny) :
                H1 += config[i,j,k]*(config[(i+1)%Nx,j,k] + config[i,(j+1)%Ny,k])
                H2 += config[i,j,k]*config[i,j,(k+1)%Nz]

    return -J*H1 - Km*H2


def mag3D(config) :
    return np.sum(config)


def delta_energy_flip3D(config, a, b, c, J, Gamma, beta) :

    Nx,Ny,Nz = np.shape(config)
    Km = 0.5*Nz/beta*np.log(1/np.tanh(beta*Gamma/Nz))

    dE = 2*config[a,b,c]*(J*(config[(a+1)%Nx,b,c] + config[a,(b+1)%Ny,c] + config[(a-1)%Nx,b,c] + config[a,(b-1)%Ny,c]) + Km*(config[a,b,(c+1)%Nz] + config[a,b,(c-1)%Nz]))
    
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

        p = np.exp(-beta*dE/Nz)

        if np.random.uniform() < p: # accept candidate if alpha>=1 or accept candidate with proba alpha if 0<alpha<1

            config[a,b,c] *= -1

            return config,dE,True
        
        else: 
            
            return config,dE,False


def MC_configurations(Nx, Ny, Nz, J, Gamma, beta, Neq, Nsteps, plot_MC = False) : # generate Nsteps spin configurations with the MC algorithm with Neq equilibration steps 

    M_burnin = np.zeros(Neq+Nsteps)        
    E_burnin = np.zeros(Neq+Nsteps+1)

    config = random_lattice3D(Nx,Ny,Nz)
    E_burnin[0] = energy3D(config,J,Gamma,beta)

    # reach equilibrium
    for j in range(Neq) :
        config,dE,accepted = MC_step3D(config,J,Gamma,beta)
        M_burnin[j] = np.abs(mag3D(config))
        if accepted :
            E_burnin[1+j] = E_burnin[j] + dE
        else :
            E_burnin[1+j] = E_burnin[j]

    # sample for average
    for k in range(Nsteps) :
        config,dE,accepted = MC_step3D(config,J,Gamma,beta)
        M_burnin[Neq+k] = np.abs(mag3D(config))
        if accepted :
            E_burnin[Neq+1+k] = E_burnin[Neq+1+k-1] + dE
        else :
            E_burnin[Neq+1+k] = E_burnin[Neq+1+k-1]

    if plot_MC :
        fig, ax = plt.subplots(1,2,figsize=(15,3))

        ax[0].set_xlabel('MC step')
        ax[1].set_xlabel('MC step')
        ax[0].set_ylabel(r'$M/N$')
        ax[1].set_ylabel(r'$E/N$')

        ax[0].set_xscale('log')
        ax[1].set_xscale('log')

        ax[0].set_xlim([1,Neq+Nsteps])
        ax[1].set_xlim([1,Neq+Nsteps+1])
        ax[0].set_ylim([np.min(M_burnin/(Nx*Ny*Nz))-0.1,np.max(M_burnin/(Nx*Ny*Nz))+0.1])
        ax[1].set_ylim([np.min(E_burnin/(Nx*Ny*Nz))-0.1,np.max(E_burnin/(Nx*Ny*Nz))+0.1])

        ax[0].plot(range(1,Neq+Nsteps),M_burnin[1:]/(Nx*Ny*Nz))
        ax[1].plot(range(1,Neq+Nsteps+1),E_burnin[1:]/(Nx*Ny*Nz))

        ax[0].plot([Neq,Neq],[np.min(M_burnin/(Nx*Ny*Nz))-0.1,np.max(M_burnin/(Nx*Ny*Nz))+0.1],'--',color='black')
        ax[1].plot([Neq,Neq],[np.min(E_burnin/(Nx*Ny*Nz))-0.1,np.max(E_burnin/(Nx*Ny*Nz))+0.1],'--',color='black')

    Mt = M_burnin[Neq:]
    Et = E_burnin[Neq:]

    return Mt,Et

