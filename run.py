# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:16:24 2022

@author: Sikander
"""
from plasticity_model import population
import numpy as np
import sys
import multiprocessing as mp

stepchange = 3
generations = 200
D = 5
plasticity = 0.6
cost = 0
developnoise = 0
replicates = 5

# rel_phen = np.zeros((replicates, generations+1))
# rel_plas = np.zeros((replicates, generations+1))
# phen_var = np.zeros((replicates, generations+1))
# N_var = np.zeros((replicates, generations+1))
# P_var = np.zeros((replicates, generations+1))
# NPcovar = np.zeros((replicates, generations+1))
# pop = np.zeros((replicates, generations+1))
# survival = np.zeros((replicates, generations+1))

def run_trials(rep):
    rel_phen = np.zeros((200, generations+1))
    rel_plas = np.zeros((200, generations+1))
    phen_var = np.zeros((200, generations+1))
    N_var = np.zeros((200, generations+1))
    P_var = np.zeros((200, generations+1))
    NPcovar = np.zeros((200, generations+1))
    pop = np.zeros((200, generations+1))
    survival = np.zeros((200, generations+1))
    
    for count in range(200):
        p = population(256, 4, 1, 0.05, 0.0005, stepchange, generations, D, plasticity, developnoise)
        p.simulate()
        rel_phen[count] = p.rel_phen
        rel_plas[count] = p.rel_plas
        phen_var[count] = p.phen_var
        N_var[count] = p.N_var 
        P_var[count] = p.P_var 
        NPcovar[count] = p.NPcovar 
        pop[count] = p.pop
        survival[count] = p.survival
    
    plas = round(plasticity, 1)
    plas = str(plas).replace(".", "")
    sc = round(stepchange, 1)
    sc = str(sc).replace(".", "")
    dn = round(developnoise, 1)
    dn = str(dn).replace(".", "")
    np.save("results/{0}_{1}_{2}_cost/rel_phen_{3}.npy".format(sc, plas, dn, rep), rel_phen)
    np.save("results/{0}_{1}_{2}_cost/rel_plas_{3}.npy".format(sc, plas, dn, rep), rel_plas)
    np.save("results/{0}_{1}_{2}_cost/phen_var_{3}.npy".format(sc, plas, dn, rep), phen_var)
    np.save("results/{0}_{1}_{2}_cost/N_var_{3}.npy".format(sc, plas, dn, rep), N_var)
    np.save("results/{0}_{1}_{2}_cost/P_var_{3}.npy".format(sc, plas, dn, rep), P_var)
    np.save("results/{0}_{1}_{2}_cost/NPcovar_{3}.npy".format(sc, plas, dn, rep), NPcovar)
    np.save("results/{0}_{1}_{2}_cost/pop_{3}.npy".format(sc, plas, dn, rep), pop)
    np.save("results/{0}_{1}_{2}_cost/survival_{3}.npy".format(sc, plas, dn, rep), survival)
    
def run_trial(rep):
    res_dct = {}
    p = population(256, 4, 1, 0.05, 0.0005, stepchange, generations, D, plasticity, developnoise)
    p.simulate(cost=cost)
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
    # t1 = mp.Process(target=run_trials, args=(0,))
    # t2 = mp.Process(target=run_trials, args=(200,))
    # t3 = mpltiprocessing.Process(target=run_trials, args=(400,))
    # t4 = mp.Process(target=run_trials, args=(600,))
    # t5 = mp.Process(target=run_trials, args=(800,))
    
    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()
    # t5.start()
    
    # t1.join()
    # t2.join()
    # t3.join()
    # t4.join()
    # t5.join()
    
    
    
    pool = mp.Pool(processes=7)
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

    if cost == 0:
        np.save("results/{0}_{1}_{2}/rel_phen.npy".format(stepchange, plasticity, developnoise), rel_phen)
        np.save("results/{0}_{1}_{2}/rel_plas.npy".format(stepchange, plasticity, developnoise), rel_plas)
        np.save("results/{0}_{1}_{2}/phen_var.npy".format(stepchange, plasticity, developnoise), phen_var)
        np.save("results/{0}_{1}_{2}/N_var.npy".format(stepchange, plasticity, developnoise), N_var)
        np.save("results/{0}_{1}_{2}/P_var.npy".format(stepchange, plasticity, developnoise), P_var)
        np.save("results/{0}_{1}_{2}/NPcovar.npy".format(stepchange, plasticity, developnoise), NPcovar)
        np.save("results/{0}_{1}_{2}/pop.npy".format(stepchange, plasticity, developnoise), pop)
        np.save("results/{0}_{1}_{2}/survival.npy".format(stepchange, plasticity, developnoise), survival)
    elif cost == 1:
        np.save("results/{0}_{1}_{2}_cost/rel_phen.npy".format(stepchange, plasticity, developnoise), rel_phen)
        np.save("results/{0}_{1}_{2}_cost/rel_plas.npy".format(stepchange, plasticity, developnoise), rel_plas)
        np.save("results/{0}_{1}_{2}_cost/phen_var.npy".format(stepchange, plasticity, developnoise), phen_var)
        np.save("results/{0}_{1}_{2}_cost/N_var.npy".format(stepchange, plasticity, developnoise), N_var)
        np.save("results/{0}_{1}_{2}_cost/P_var.npy".format(stepchange, plasticity, developnoise), P_var)
        np.save("results/{0}_{1}_{2}_cost/NPcovar.npy".format(stepchange, plasticity, developnoise), NPcovar)
        np.save("results/{0}_{1}_{2}_cost/pop.npy".format(stepchange, plasticity, developnoise), pop)
        np.save("results/{0}_{1}_{2}_cost/survival.npy".format(stepchange, plasticity, developnoise), survival)

