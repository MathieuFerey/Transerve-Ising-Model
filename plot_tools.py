import numpy as np
import matplotlib.pyplot as plt


# DSNB plot tools =============================================================

def create_plot(xlabel, ylabel, xlim, ylim=[], title='', logx=False, logy=False, figsize=plt.rcParams.get('figure.figsize'), grid=True, axis_size=13, legend_size=11, tick_size=11) :

    plt.rc('axes',labelsize=axis_size)
    plt.rc('legend',fontsize=legend_size)
    plt.rc('xtick',labelsize=tick_size)
    plt.rc('ytick',labelsize=tick_size)
    plt.rc('font',size=legend_size)

    plt.figure(figsize=figsize)
    plt.tick_params(top=True,right=True,labeltop=False,labelright=False)

    plt.xlim(xlim)
    if len(ylim) != 0 : plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)

    if grid :
        plt.grid(visible=True, which="major", axis="both", linestyle="-", alpha=0.2, color="black",zorder=0)
        plt.grid(visible=True, which="minor", axis="both", linestyle="--", alpha=0.2, color="black",zorder=0)

    if logx :
        plt.xscale('log')

    if logy :
        plt.yscale('log')


def color_gradient(color, N) :

    if N-1 == 0 : N+=1

    if color == 'r' :
        colors = [(1,i/(N-1),0) for i in range(N)]
    elif color == 'g' :
        colors = [(i/(N-1),1,0) for i in range(N)]
    elif color == 'b' :
        colors = [(0,i/(N-1),1) for i in range(N)]
    else :
        colors = [color for i in range(N)]

    return colors



