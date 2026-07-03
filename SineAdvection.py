# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:57:27 2026

@author: Elliott
"""

import numpy as np
import matplotlib.pyplot as plt

def rhs_4th_order(u, dx, c):
    """
    Computes the spatial derivative using 4th-order central difference
    with periodic boundary conditions.
    """
    #Shift arrays to calculate central difference
    u_plus_2 = np.roll(u, -2)
    u_plus_1 = np.roll(u, -1)
    u_minus_1 = np.roll(u, 1)
    u_minus_2 = np.roll(u, 2)
    
    #4th-order central difference formula
    du_dx = (-u_plus_2 + 8.0 * u_plus_1 - 8.0 * u_minus_1 + u_minus_2) / (12.0 * dx)
    
    return -c * du_dx

def solve_advection_rk4_4th_order():
    #Define parameters
    c = 10.0           #advection velocity
    L = 100.0         #domain length
    Nx = 200          #number of spatial grid points
    dx = L / Nx       #spatial step size
    
    #Courant-Friedrichs-Lewy (CFL) condition
    CFL = 2.0
    dt = CFL * dx / c 
    
    T = 50.0           #Total simulation time
    Nt = int(T / dt)  #Number of time steps
    
    #Create spatial grid
    x = np.linspace(0, L, Nx, endpoint=False)
    
    #Initial conditions
    u0 = np.sin(2.0 * np.pi * x / L)
    u = u0.copy()
    
    #RK4 loop
    for n in range(Nt):
        k1 = rhs_4th_order(u, dx, c)
        k2 = rhs_4th_order(u + 0.5 * dt * k1, dx, c)
        k3 = rhs_4th_order(u + 0.5 * dt * k2, dx, c)
        k4 = rhs_4th_order(u + dt * k3, dx, c)
        
        
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
    plt.figure(figsize=(10, 6))
    plt.plot(x, u0, '--', label='Initial Condition (t=0)')
    plt.plot(x, u, '-', linewidth=2, label=f'Final State (t={T:.2f})')
    plt.title('1D Advection Equation using wrong CFL number')
    plt.xlabel('Spatial Coordinate (x)')
    plt.ylabel('Amplitude (u)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
solve_advection_rk4_4th_order()