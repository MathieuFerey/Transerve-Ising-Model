import numpy as np
import matplotlib.pyplot as plt



def Newton(func, x0, precision=1e-3,  max_iter=1000, epsilon=1e-5,plot=False) : # func functions we want to find the zero of, x0 initial point, precision desired precision of the result, max_iter maximum number of iterations if the program doesn't converge to the desired precision, epsilon small param to approximate the derivative

    x = x0
    i = 0
    
    if plot :
        plt.figure()
        xplot = np.linspace(x0-0.01,x0+0.1,100)
        plt.plot(xplot,func(xplot))
        plt.plot(xplot,np.zeros(len(xplot)),linestyle='--',color='r',alpha=0.5)

    while np.abs(func(x)) > precision and i <= max_iter :
        
        if plot :
            print(x,i,np.abs(func(x)))
            plt.scatter(x,func(x),color='black')
        df = (func(x+epsilon)-func(x))/epsilon # approximate derivative at x
        x = x-func(x)/df # find 0 of the tangent
        i += 1

    if plot : plt.show()
    
    return x


def func(x) : return np.tan(np.sqrt(3)*x)+np.sqrt(3)*np.tan(x)
        
