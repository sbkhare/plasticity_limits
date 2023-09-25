# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:16:14 2022

@author: Sikander
"""
from plasticity_model import population
import numpy as np
import os
import sys
import multiprocessing as mp

stepchange = 3
generations = 500
D = int(sys.argv[1])
plasticity = float(sys.argv[2])/10
cost = int(sys.argv[3])
developnoise = float(sys.argv[4])/10
envnoise = False
autocorrel = 0.5
if float(sys.argv[5]) != 0:
    envnoise = float(sys.argv[5])/100
cpu_count = int(sys.argv[6])
print("SLURM CPU Count: ", cpu_count)


def run_trial(rep):
    res_dct = {}
    p = population(256, 4, 1, 0.05, 0.0005, stepchange, generations, D, plasticity, developnoise)
    p.simulate(cost=cost, tau=envnoise, rho=autocorrel)
    res_dct["rel_phen"] = p.rel_phen
    res_dct["rel_plas"] = p.rel_plas
    res_dct["phen_var"] = p.phen_var
    res_dct["N_var"] = p.N_var 
    res_dct["P_var"] = p.P_var 
    res_dct["NPcovar"] = p.NPcovar 
    res_dct["pop"] = p.pop
    res_dct["survival"] = p.survival
    return res_dct

if __name__ == '__main__':
    pool = mp.Pool(processes=cpu_count)
    results = pool.map(run_trial, range(1000))
    
    rel_phen = np.stack((res_dct['rel_phen'] for res_dct in results))
    rel_plas = np.stack((res_dct['rel_plas'] for res_dct in results))
    phen_var = np.stack((res_dct['phen_var'] for res_dct in results))
    N_var = np.stack((res_dct['N_var'] for res_dct in results))
    P_var = np.stack((res_dct['P_var'] for res_dct in results))
    NPcovar = np.stack((res_dct['NPcovar'] for res_dct in results))
    pop = np.stack((res_dct['pop'] for res_dct in results))
    survival = np.stack((res_dct['survival'] for res_dct in results))
    
    plasticity = round(plasticity, 1)
    plasticity = str(plasticity).replace(".", "")
    stepchange = round(stepchange, 1)
    stepchange = str(stepchange).replace(".", "")
    developnoise = round(developnoise, 1)
    developnoise = str(developnoise).replace(".", "")
    envnoise = str(envnoise).replace(".", "_")
    autocorrel = str(autocorrel).replace(".", "_")

    if cost == 0 and envnoise == False:
        np.save("results/{0}_{1}_{2}/rel_phen.npy".format(D, plasticity, developnoise), rel_phen)
        np.save("results/{0}_{1}_{2}/rel_plas.npy".format(D, plasticity, developnoise), rel_plas)
        np.save("results/{0}_{1}_{2}/phen_var.npy".format(D, plasticity, developnoise), phen_var)
        np.save("results/{0}_{1}_{2}/N_var.npy".format(D, plasticity, developnoise), N_var)
        np.save("results/{0}_{1}_{2}/P_var.npy".format(D, plasticity, developnoise), P_var)
        np.save("results/{0}_{1}_{2}/NPcovar.npy".format(D, plasticity, developnoise), NPcovar)
        np.save("results/{0}_{1}_{2}/pop.npy".format(D, plasticity, developnoise), pop)
        np.save("results/{0}_{1}_{2}/survival.npy".format(D, plasticity, developnoise), survival)
    elif cost == 1 and envnoise == False:
        np.save("results/{0}_{1}_{2}_cost/rel_phen.npy".format(D, plasticity, developnoise), rel_phen)
        np.save("results/{0}_{1}_{2}_cost/rel_plas.npy".format(D, plasticity, developnoise), rel_plas)
        np.save("results/{0}_{1}_{2}_cost/phen_var.npy".format(D, plasticity, developnoise), phen_var)
        np.save("results/{0}_{1}_{2}_cost/N_var.npy".format(D, plasticity, developnoise), N_var)
        np.save("results/{0}_{1}_{2}_cost/P_var.npy".format(D, plasticity, developnoise), P_var)
        np.save("results/{0}_{1}_{2}_cost/NPcovar.npy".format(D, plasticity, developnoise), NPcovar)
        np.save("results/{0}_{1}_{2}_cost/pop.npy".format(D, plasticity, developnoise), pop)
        np.save("results/{0}_{1}_{2}_cost/survival.npy".format(D, plasticity, developnoise), survival)
    elif cost == 0 and envnoise != False:
        np.save("results/{0}_{1}_{2}/envnoise_{3}_{4}/rel_phen.npy".format(D, plasticity, developnoise, envnoise, autocorrel), rel_phen)
        np.save("results/{0}_{1}_{2}/envnoise_{3}_{4}/rel_plas.npy".format(D, plasticity, developnoise, envnoise, autocorrel), rel_plas)
        np.save("results/{0}_{1}_{2}/envnoise_{3}_{4}/phen_var.npy".format(D, plasticity, developnoise, envnoise, autocorrel), phen_var)
        np.save("results/{0}_{1}_{2}/envnoise_{3}_{4}/N_var.npy".format(D, plasticity, developnoise, envnoise, autocorrel), N_var)
        np.save("results/{0}_{1}_{2}/envnoise_{3}_{4}/P_var.npy".format(D, plasticity, developnoise, envnoise, autocorrel), P_var)
        np.save("results/{0}_{1}_{2}/envnoise_{3}_{4}/NPcovar.npy".format(D, plasticity, developnoise, envnoise, autocorrel), NPcovar)
        np.save("results/{0}_{1}_{2}/envnoise_{3}_{4}/pop.npy".format(D, plasticity, developnoise, envnoise, autocorrel), pop)
        np.save("results/{0}_{1}_{2}/envnoise_{3}_{4}/survival.npy".format(D, plasticity, developnoise, envnoise, autocorrel), survival)
    
    