# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:33:48 2022

@author: Sikander
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy import integrate

def gompertz(t, Dc, G, P):
    return Dc*np.exp(-np.exp(G - P*t))

def alternative(t, Dc, G, P):
    return Dc*np.exp(-np.exp(np.log(np.log(2)) - P*t)) + G

def logistic(t, Dc, N, P):
    t0 = 0
    return Dc/(1 + np.exp(-P*((t - t0)))) + N - Dc/2

x = np.linspace(-5, 5, 500)
x1 = np.linspace(-2, 2, 500)
x2 = np.linspace(-1.5, 1.5, 500)
# y1 = logistic(x, 3, 0, 4)
y2 = logistic(x, 3, 0, 4/3)
y5 = logistic(x, 3, 0, 3/4)
y3 = logistic(x, 3, 0, 10)
# y4 = logistic(x, 3, 0, 0)
z1 = logistic(x1, 2, 0, 2)
z2 = logistic(x1, 1, 0, 4)
# z3 = logistic(x1, 6, 0, 2/3)
z4 = logistic(x1, 8, 0, 1/2)


# FIG 2. Reaction norm figure   ##
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(x, y3, color="k", linestyle=":", label=r"d=b$\Sigma$$P_{k}$=10")
# plt.plot(x, y1, color="k", linestyle="--", label=r"$\Sigma$P=4")
plt.plot(x, y2, color="k", linestyle="-.", label=r"d=b$\Sigma$$P_{k}$=1.33")
plt.plot(x, y5, color="k", linestyle="--", label=r"d=b$\Sigma$$P_{k}$=0.75")
# plt.plot(x, y4, color="k", linestyle="-.", label=r"$\Sigma$P=0")
plt.plot(x2, x2, color="k", label=r"$T_{opt}$ = E")
plt.ylabel("Phenotype, T", fontsize=18)
plt.xlabel("Environment, E", fontsize=18)
plt.xticks([])
plt.yticks([])
plt.grid()
leg1 = plt.legend(loc="lower right")
leg1.get_frame().set_edgecolor('k')
plt.text(-5*1.05, 1.5*0.9, "A", fontsize=20)
plt.subplot(1,2,2)
plt.plot(x1, z4, color="k", linestyle=":", label=r"D=8, d=b$\Sigma$$P_{k}$=0.5")
plt.plot(x1, z1, color="k", linestyle="-.", label=r"D=2, d=b$\Sigma$$P_{k}$=1")
plt.plot(x1, z2, color="k", linestyle="--", label=r"D=1, d=b$\Sigma$$P_{k}$=4")
# plt.plot(x1, z3, color="k", linestyle=(5, (10, 3)), label=r"D=6, $\Sigma$P=2/3")
plt.plot(x1, x1, color="k", label=r"$T_{opt}$ = E")
plt.xlabel("Environment, E", fontsize=18)
plt.xticks([])
plt.yticks([])
plt.grid()
leg1 = plt.legend(loc="lower right")
leg1.get_frame().set_edgecolor('k')
plt.text(-2*1.05, 2*0.9, "B", fontsize=20)
plt.tight_layout()
plt.savefig("figs/reaction_norm.png")




# plt.figure()
# plt.plot(x, 0.4*x, label="$T_{opt}$")
# plt.title("N = 0")
# plt.plot(x, z1, label="D=5, P=0.5")
# plt.plot(x, z2, label="D=12, P=0.3")
# plt.plot(x, z3, label="D=20, P=0.225")
# plt.plot(x, z4, label="D=20, P=4B/D")
# plt.xlabel("Environment, E")
# plt.grid()
# plt.legend()

# plt.figure()
# D = 20
# B = 10/25
# P = 4*B/D
# E_rng = np.linspace(-25,25)
# slope = P*np.exp(P*E_rng)*D/(1 + np.exp(P*E_rng))**2
# plt.plot(E_rng, slope, label="T'(E)")
# plt.axhline(B, label="O'(E)", color="r")
# plt.xlabel("E")
# plt.ylabel("Slope")
# plt.legend()

D_=20
B=0.4
P_ = 4*B/D_
# init_printing(use_unicode=False, wrap_line=False)
# E = Symbol('E')
# integrate(exp(-0.5*(E - D/(1 - exp(-P_*E)) + D/2)**2), E)

def approximation(D, epsilon):
    P = 4*B/D
    for x in np.linspace(0,8,1000):
        difference = D*(1/(1 + np.exp(-P*x)) - 1/2 - P*x/4)
        if abs(difference) < epsilon:
            large_x = x
            if x == 8:
                return x
        else:
            return large_x

def fitness_int(E, P, D):
    return np.exp(-0.5*(E - (D/(1 + np.exp(-P*E)) - D/2))**2)

def fitness_surface(P, D, disp_range):
    x_ = disp_range/2
    E_range = np.linspace(-x_, x_, 1000)
    F_E = fitness_int(E_range, P, D)
    M = integrate.simpson(F_E, E_range)/(2*x_)
    return M
        
# D_range = np.linspace(0.1,20,100)
# plt.figure()
# plt.xlabel("D")
# plt.ylabel("Largest x s.t. difference < \u03B5")
# plt.title("B = 0.4, P = 4B/D, \u03B5 = 0.01")
# x_r = []
# for d in D_range:
#     x_r.append(approximation(d, 0.01))
#     # print(approximation(d, 0.01))
# plt.plot(D_range, x_r)


# D_range = np.linspace(0, 12, 100)
# P_range = np.linspace(-1, 3, 100)
# DD, PP  = np.meshgrid(D_range, P_range)
# MM = np.zeros(shape=(100,100))
# for i, D in enumerate(D_range):
#     for j, P in enumerate(P_range):
#         MM[j,i] = fitness_surface(P, D, 10)
        
# plt.contourf(D_range, P_range, MM)
# plt.colorbar()
# plt.plot(D_range[11:], 4/D_range[11:], color="r")
# plt.xlabel("Developmental constraint")
# plt.ylabel("Steepness")


# DC = 12
# disp = np.linspace(0,10,100)
# dd, pp = np.meshgrid(disp, P_range) 
# mm = np.zeros(shape=(100,100))
# for i, d in enumerate(disp):
#     for j, P in enumerate(P_range):
#         mm[j,i] = fitness_surface(P, DC, d)

# plt.figure()
# plt.contourf(disp, P_range, mm)
# plt.colorbar()
# plt.xlabel("Dispersal range")
# plt.ylabel("Steepness")
# plt.title("D = {0}".format(DC))
    

# b_list = [0.2, 0.4, 0.6, 0.8, 1]
# E_new = 3
# P_range = np.linspace(0,5,100)
# D_C = 7
# for b in b_list:
#     plt.figure()
#     plt.plot(P_range, P_range**2 + (E_new - b*E_new*P_range)**2, color="k", linestyle="dashed", label="T(P) = b*E*P")
#     plt.plot(P_range, P_range**2 + (E_new - D_C/(1 + np.exp(-b*E_new*P_range)) + D_C/2)**2, color="k", label="T(P) = D/(1 + exp(-b*E*P)) - D/2")
#     plt.axhline(0, color="r", linestyle="dashed")
#     plt.xlabel("P")
#     plt.ylabel("P^2 + (E - T(P))^2")
#     plt.title("b = {0}, E=3, D={1}".format(b, D_C))
#     plt.legend()