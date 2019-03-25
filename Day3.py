# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:26:23 2019

@author: wertd
"""
import numpy as np
from matplotlib import pyplot as plt
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

theta = np.linspace(5, 2*np.pi, 500)
psi = PIR_Func(5, theta)
plt.plot(theta, psi)
plt.show()                                             


    