# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:26:23 2019

@author: wertd
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = plt.axes(xlim = (0, 2*np.pi), ylim =(-1.2, 1.2))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

h=1
u=1
R=1
m=2
ci= 0+1j
o= np.linspace(0, 2*np.pi, 1000)

I= u*(R**2)

hbar = 1
def PIR_En(n,I):
    return n**2*hbar**2/(2*I)

def PIR_Time(n,I, t):
    ci = (0+1j)
    En = PIR_En(n, I)
    return np.exp(ci*En*t/hbar)
def PIR_Func(n, theta):
    ci = 0+1j
    psi_n = (1/np.sqrt(2*np.pi))*np.exp(ci*n*theta)
    return psi_n

def animate(i):
    line.set_ydata( PIR_Func(m,o)*PIR_Time(m, I, i/10) )
    line.set_xdata(o)
    
    return line,

print (PIR_Func(1,1))

theta = np.linspace(0, 2*np.pi, 500)
psi = PIR_Func(5, theta)

  
def Triangle_Wave(theta):
    tw = np.zeros(len(theta))
    for i in range(0, len(theta)):
        xval = theta[i]
        if xval <= 2: 
            tw[i] = 0
        elif xval < 3:
            tw[i] = xval - 2
        elif xval < 4:
            tw[i] = -xval + 4
        else:
            tw[i] = 0
    return tw


    
def fourier_analysis(tw, m, o):
    psi = PIR_Func(m, o)
    psistar = np.conj(psi)
    integrand = psistar * tw
    ### regular rule width is called w
    w = o[1] - o[0]
    ### rectangle rule sum will be called rsum
    rsum = 0 +0j
    for i in range(0, len(tw)):
        rsum = rsum + integrand[i] * w
        
    return rsum
    
                                           



#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=10000, interval=20, blit=True)
marray = np.linspace(-100, 100, 201)
c_array = np.zeros(len(marray), dtype=complex)
psi_exp = np.zeros(len(trianglewave), dtype=complex)

for i in range(0, len(marray)):
    c_array[i] = fourier_analysis(trianglewave, marray[i], o)
    psi_exp = psi_exp + c_array[i]*PIR_Func(marray[i],o)

psi_1_times_tw = trianglewave * PIR_Func(1, o)

plt.plot(o, trianglewave, o, psi_exp)
plt.show()  