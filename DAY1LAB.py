# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt

def pib_func(n, x_array, L):
    return np.sqrt(2/L) * np.sin( n* np.pi * x_array / L)

def gauss_packet(x_array, x0, sig, k0):
    ci= 0+1j
    G = np.exp(-0.5 * ((x_array - x0)/sig) **2)
    Norm = 1./(sig * np.sqrt(2*np.pi))
    P = np.exp(ci * k0 * x_array)
    return Norm * G * P

def fourier_analysis(x_array, Psi, n, L):
   psi_n = pib_func(n, x_array, L)
   f_of_x = psi_n * Psi
   dx = x_array[1] - x_array[0]
   
   f_sum = 0.
   for i in range(0,len(x_array)):
       f_sum = f_sum + f_of_x[i] * dx
   return f_sum


L = 500
x = np.linspace(0, L, 2000)

psi_1 = pib_func(10, x, L)

Psi = gauss_packet(x, 200, 15, 0.4)
Psi_exp = np.zeros_like(Psi)

for i in range(1,200):
    c1 = fourier_analysis(x, Psi, 1, L)
    Psi_exp = Psi_exp + c1*pib_func(i, x, L)


plt.plot(x, Psi, 'purple')
plt.plot(x, Psi_exp, 'b--')
plt.show()