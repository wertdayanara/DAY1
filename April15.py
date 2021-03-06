# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:26:23 2019

@author: wertd
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from numpy.random import choice


fig = plt.figure()
ax = plt.axes(xlim = (0, 2*np.pi), ylim =(-10, 10))
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


    

print (PIR_Func(1,1))

theta = np.linspace(0, 2*np.pi, 500)
psi = PIR_Func(5, theta)

def gauss_packet(x_array, x0, sig, k0):
    ci= 0+1j
    G = np.exp(-0.5 * ((x_array - x0)/sig) **2)
    Norm = 1./(sig * np.sqrt(2*np.pi))
    P = np.exp(ci * k0 * x_array)
    return Norm * G * P
  
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
    
                                           




trianglewave = gauss_packet(o, np.pi/4, 0.1, 0.1)
marray = np.linspace(-100, 100, 201)
c_array = np.zeros(len(marray), dtype=complex)




for i in range(0, len(marray)):
    c_array[i] = fourier_analysis(trianglewave, marray[i], o)
print (np.conj(c_array[100])*(c_array[100]))
 
p_of_en = np.real( np.conj(c_array)*c_array )
norm = np.sum(p_of_en)
p_of_en = p_of_en / norm

#draw = choice(marray, 1, p=p_of_en)
#print("drew this number", draw)
#print(" Collapsed to random eigenfunction with m = ",marray[int(draw[0])])
draw = choice(marray, 1, p=p_of_en)
state = int(draw[0])


def animate(i):
    #print(c_array)
    psi_exp = np.zeros(len(trianglewave), dtype=complex)
    if i<500:
        for j in range(0, len(marray)):
            psi_exp = psi_exp + c_array[j]*PIR_Func(marray[j],o)*PIR_Time(marray[j], I, i/10000)
    elif i>=500:
       psi_exp = PIR_Func(state, o)*PIR_Time(state, I, i/1000)
  
        
       
            
    line.set_ydata( psi_exp )
    line.set_xdata(o)
    return line,
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=10000, interval=20, blit=True)

#psi_1_times_tw = trianglewave * PIR_Func(1, o)

#plt.plot(o, trianglewave, o, psi_exp)
#plt.show()  