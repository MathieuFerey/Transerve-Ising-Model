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

    return -J/Nz*H1 - Km*H2


def mag3D(config) :
    return np.sum(config)


def delta_energy_flip3D(config, a, b, c, J, Gamma, beta) :

    Nx,Ny,Nz = np.shape(config)
    Km = 0.5/beta*np.log(1/np.tanh(beta*Gamma/Nz))

    dE = 2*config[a,b,c]*(J/Nz*(config[(a+1)%Nx,b,c] + config[a,(b+1)%Ny,c] + config[(a-1)%Nx,b,c] + config[a,(b-1)%Ny,c]) + Km*(config[a,b,(c+1)%Nz] + config[a,b,(c-1)%Nz]))
    
    return dE


def MC_step3D(config, J, Gamma, beta) : # perform a whole sweep of the lattice by flipping N times a randomly chosen spin and checking the likelihood of this new configuration

    Nx,Ny,Nz = np.shape(config)

    for i in range(Nx*Ny*Nz) :
        # flip one randomly chosen site

        a = np.random.randint(low=0,high=Nx)
        b = np.random.randint(low=0,high=Ny)
        c = np.random.randint(low=0,high=Nz)

        dE = delta_energy_flip3D(config,a,b,c,J,Gamma,beta)

        if dE<0 :
            config[a,b,c] *= -1

        else :

            p = np.exp(-beta*dE)

            if np.random.uniform() < p: # accept candidate if alpha>=1 or accept candidate with proba alpha if 0<alpha<1

                config[a,b,c] *= -1            
                
    return config


def MC_configurations(Nx, Ny, Nz, J, Gamma, beta, Neq, Nsteps, initial_config, plot_MC = False) : # generate Nsteps spin configurations with the MC algorithm with Neq equilibration steps 

    M_burnin = np.zeros(Neq+Nsteps)        

    #config = random_lattice3D(Nx,Ny,Nz)
    config = initial_config

    # reach equilibrium
    for j in range(Neq) :
        config = MC_step3D(config,J,Gamma,beta)
        M_burnin[j] = np.abs(mag3D(config))

    # sample for average
    for k in range(Nsteps) :
        config = MC_step3D(config,J,Gamma,beta)
        M_burnin[Neq+k] = np.abs(mag3D(config))
    
    if plot_MC :
        create_plot('MC step',r'$M/N$',[0,Neq+Nsteps],ylim=[np.min(M_burnin/(Nx*Ny*Nz))-0.1,np.max(M_burnin/(Nx*Ny*Nz))+0.1])
        plt.plot(range(Neq+Nsteps),M_burnin/(Nx*Ny*Nz))
        plt.plot([Neq,Neq],[np.min(M_burnin/(Nx*Ny*Nz))-0.1,np.max(M_burnin/(Nx*Ny*Nz))+0.1],'--',color='black')

    Mt = M_burnin[Neq:]

    return Mt,config


def MC_averages(Nx, Ny, Nz, J, Gamma, T, Neq, Nsteps, plot_MC = []) : 

    M = np.zeros((len(Gamma),len(T)))
    chi = np.zeros((len(Gamma),len(T)))

    last_config = np.ones((Nx,Ny,Nz))

    for i in range(len(Gamma)) :
        gamma = Gamma[i]
        for j in tqdm(range(len(T))) :
            t = T[j]
            plot = False
            if j in plot_MC: plot = True
            if i != 0 or j != 0 : Nequ = 0
            else : Nequ=Neq
            Mt,last_config = MC_configurations(Nx,Ny,Nz,J,gamma,1/t,Nequ,Nsteps,last_config,plot_MC=plot)
            M[i,j] = np.sum(Mt)/Nsteps
            chi[i,j] = (np.sum(Mt**2)/Nsteps - M[i,j]**2)/t

    return M/(Nx*Ny*Nz),chi/(Nx*Ny*Nz)


def MC_averages2(Nx, Ny, Nz, J, Gamma, T, Neq, Nsteps, plot_MC = []) : 

    M = np.zeros((len(T),len(Gamma)))
    chi = np.zeros((len(T),len(Gamma)))

    last_config = np.ones((Nx,Ny,Nz))

    for i in range(len(T)) :
        t = T[i]
        for j in tqdm(range(len(Gamma))) :
            gamma = Gamma[j]
            plot = False
            if j in plot_MC: plot = True
            if i != 0 or j != 0 : Nequ = 0
            else : Nequ=Neq
            Mt,last_config = MC_configurations(Nx,Ny,Nz,J,gamma,1/t,Nequ,Nsteps,last_config,plot_MC=plot)
            M[i,j] = np.sum(Mt)/Nsteps
            chi[i,j] = (np.sum(Mt**2)/Nsteps - M[i,j]**2)/t

    return M/(Nx*Ny*Nz),chi/(Nx*Ny*Nz)

