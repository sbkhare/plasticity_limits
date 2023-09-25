# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:59:03 2022

@author: s.khare
"""

import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('fast') #bmh, seaborn-darkgrid'

marker_dct = {"0.0": "o", "0.2": "v", "0.4": "s", "0.5":"s", "0.6": "D", "0.8": "^", "1.0": "h"}
linestyle_dct = {"00": "-", "02": ":", "04": "--", "0_5": "--", "06": (0, (3, 1, 1, 1, 1, 1)), 
                 "08": (5, (10, 3)), "10": "-."}

    
def load_data(D, data, var, scenario="none"):
    if scenario == "none":
        joined = np.load("results/{0}_{1}_00/{2}.npy".format(D, var, data))
        pop = np.load("results/{0}_{1}_00/pop.npy".format(D, var))
        table = []
        for p, j  in zip(pop, joined):
            if p[-1] != 0:
                table.append(j)
        table = np.stack(table)
        return table
    elif scenario == "cost":
        joined = np.load("results/{0}_{1}_00_cost/{2}.npy".format(D, var, data))
        pop = np.load("results/{0}_{1}_00_cost/pop.npy".format(D, var))
        table = []
        for p, j  in zip(pop, joined):
            if p[-1] != 0:
                table.append(j)
        table = np.stack(table)
        return table
    elif scenario == "noise":
        joined = np.load("results/{0}_06_{1}/{2}.npy".format(D, var, data))
        pop = np.load("results/{0}_06_{1}/pop.npy".format(D, var))
        table = []
        for p, j  in zip(pop, joined):
            if p[-1] != 0:
                table.append(j)
        table = np.stack(table)
        return table
    elif scenario == "env_noise_corr":
        joined = np.load("results/{0}_06_00/envnoise_{1}/{2}.npy".format(D, var, data))
        pop = np.load("results/{0}_06_00/envnoise_{1}/pop.npy".format(D, var))
        table = []
        for p, j  in zip(pop, joined):
            if p[-1] != 0:
                table.append(j)
        print(len(table))
        table = np.stack(table)
        return table
    elif scenario == "env_noise_corr05":
        joined = np.load("results/{0}_06_00/envnoise_{1}_0_5/{2}.npy".format(D, var, data))
        pop = np.load("results/{0}_06_00/envnoise_{1}_0_5/pop.npy".format(D, var))
        table = []
        for p, j  in zip(pop, joined):
            if p[-1] != 0:
                table.append(j)
        print(len(table))
        table = np.stack(table)
        return table
    elif scenario == "env_noise_uncorr":
        joined = np.load("results/{0}_06_00/envnoise_{1}_0/{2}.npy".format(D, var, data))
        pop = np.load("results/{0}_06_00/envnoise_{1}_0/pop.npy".format(D, var))
        table = []
        for p, j  in zip(pop, joined):
            if p[-1] != 0:
                table.append(j)
        print(len(table))
        table = np.stack(table)
        return table
         
def make_fig(title, D, data, scenario, lim=True, gen=200, save=True):
    plt.figure()
    plt.xlabel("Generation")
    plt.ylabel(title) 
    if scenario == "cost":
        # plt.title("b = ")
        for var in ["02", "04","06", "08", "10"]: #"00",  "08", "1"
            table = load_data(D, data, var, scenario)
            var = var[0] + "." + var[1]
            plt.plot(np.mean(table, axis=0)[:gen+1], label=var)
        plt.legend(title="b = ")
    else:
        if scenario == "noise":
            # plt.title("s = ")
            for var in ["02", "04", "06", "08", "10"]: # "00", 
                table = load_data(D, data, var, scenario) 
                var = var[0] + "." + var[1]
                plt.plot(np.mean(table, axis=0)[:gen+1], label=var)
            plt.legend(title="s = ")
        else:
            # plt.title("b = ")
            for var in ["02", "04", "06", "08", "10"]: # "00", 
                table = load_data(D, data, var, scenario) 
                var = var[0] + "." + var[1]
                plt.plot(np.mean(table, axis=0)[:gen+1], label=var)
            plt.legend(title="b = ")
    if lim:
        plt.ylim(0,1)
    plt.legend()
    if save:
        plt.savefig("figs/{0}_{1}_{2}.png".format(D, scenario, data))

def check_pop(scenario):
    if scenario == "cost":
        for var in ["04", "06", "08", "1"]:
            print(scenario, var)
            table = load_data("pop", var, scenario)
            print(np.count_nonzero(table == 0))
    else:
        for var in ["0", "02", "04", "06", "08", "1"]:
            print(scenario, var)
            table = load_data("pop", var, scenario)
            print(np.count_nonzero(table == 0))
            
def survival(D, scenario, save=True):
    plt.figure()
    plt.xlabel("Generation")
    plt.ylabel("Probability of survival")
    if scenario == "none":
        plt.title("b = ")
        for var in ["00", "02", "04", "06", "08", "10"]:
            pop = np.load("results/{0}_{1}_00/pop.npy".format(D, var))
            var = var[0] + "." + var[1]
            plt.plot(np.count_nonzero(pop, axis=0)[:51]/1000, label=var)
    elif scenario == "cost":
        plt.title("b = ")
        for var in ["00", "02", "04", "06", "08", "10"]: # , 
            pop = np.load("results/{0}_{1}_00_cost/pop.npy".format(D, var))
            var = var[0] + "." + var[1]
            plt.plot(np.count_nonzero(pop, axis=0)[:51]/1000, label=var)
    elif scenario == "noise":
        plt.title("s = ")
        for var in ["00", "02", "04", "06", "08", "10"]:
            pop = np.load("results/{0}_06_{1}/pop.npy".format(D, var))
            var = var[0] + "." + var[1]
            plt.plot(np.count_nonzero(pop, axis=0)[:51]/1000, label=var)
    plt.ylim(0,1)
    plt.legend()
    if save:
        plt.savefig("figs/{0}_{1}_survival.png".format(D, scenario))

def cross_D_comparison(title, data, scenario, save=True, leg_loc=None):
    # plt.figure()
    if title:
        plt.ylabel(title, fontsize=18) 
    # plt.xlabel("Developmental limit", fontsize=16)
    D_range = np.arange(1,9)
    if scenario == "none":
        # plt.title("b = ")
        for var in ["02", "04", "06", "08", "10"]: #"00", 
            results = []
            err = []
            for D in D_range:
                if data == "survival":
                    table = load_data(D, "pop", var, scenario)
                    results.append(np.count_nonzero(table, axis=0)[-1]/1000)
                else:
                    table = load_data(D, data, var, scenario)
                    results.append(np.mean(table[:,-1]))
                    err.append(np.std(table[:,-1]))
                    # print(np.mean(table[:,-1]))
            var = var[0] + "." + var[1]
            plt.plot(D_range/6, results, color="k", marker=marker_dct[var], linewidth=2, markersize=10, label=var)
            # plt.errorbar(D_range/6, results, err, capsize=5, color="k", marker=marker_dct[var], markersize=10, label=var)
        leg = plt.legend(title="b = ", loc=leg_loc)
        leg.get_frame().set_edgecolor('k')
    elif scenario == "cost":
        # plt.title("b = ")
        for var in ["02", "04", "06", "08", "10"]: #"00", 
            results = []
            for D in D_range:
                if data == "survival":
                    table = load_data(D, "pop", var, scenario)
                    results.append(np.count_nonzero(table, axis=0)[-1]/1000)
                else:
                    table = load_data(D, data, var, scenario)
                    results.append(np.mean(table[:,-1]))
            var = var[0] + "." + var[1]
            plt.plot(D_range/6, results, color="k", marker=marker_dct[var], linewidth=2, markersize=10, label=var)
        leg = plt.legend(title="b = ", loc=leg_loc)
        leg.get_frame().set_edgecolor('k')
    elif scenario == "noise":
        # plt.title("s = ")
        for var in ["00", "02", "04", "06", "08", "10"]: # 
            results = []
            for D in D_range:
                if data == "survival":
                    table = load_data(D, "pop", var, scenario)
                    results.append(np.count_nonzero(table, axis=0)[-1]/1000)
                else:
                    table = load_data(D, data, var, scenario)
                    results.append(np.mean(table[:,-1]))
            var = var[0] + "." + var[1]
            plt.plot(D_range/6, results, color="k", marker=marker_dct[var], linewidth=2, markersize=10,label=var)
        leg = plt.legend(title="s = ", loc=leg_loc)
        leg.get_frame().set_edgecolor('k')
    elif scenario[:9] == "env_noise":
        # plt.title("s = ")
        for var in ["0_0", "0_5", "1_0"]: # 
            results = []
            for D in D_range:
                if data == "survival":
                    if var == "0_0":
                        table = load_data(D, "pop", "06", "none") #tau = 0 case is just the none case
                        results.append(np.count_nonzero(table, axis=0)[-1]/1000)
                    else:
                        table = load_data(D, "pop", var, scenario)
                        results.append(np.count_nonzero(table, axis=0)[-1]/1000)
                else:
                    if var == "0_0":
                        table = load_data(D, data, "06", "none")
                        results.append(np.mean(table[:,-1]))
                    else:
                        table = load_data(D, data, var, scenario)
                        results.append(np.mean(table[:,-1]))
            var = var[0] + "." + var[2:]
            plt.plot(D_range/6, results, color="k", marker=marker_dct[var], linewidth=2, markersize=10,label=var)
        leg = plt.legend(title=r"$\tau$ = ", loc=leg_loc)
        leg.get_frame().set_edgecolor('k')
    plt.ylim(-0.06,1.06)
    plt.xticks(np.arange(0.2,1.5,0.1), [0.2, None, 0.4, None, 0.6, None, 0.8, None, 1. , None, 1.2, None, 1.4], fontsize=18)
    plt.yticks(np.arange(0,1.1,0.1), [0. , None, 0.2, None, 0.4, None, 0.6, None, 0.8, None, 1. ], fontsize=18)
    plt.tick_params(length=8)
    # plt.axes().minorticks_on()
    if data == "survival":
        res_zero = []
        for D in D_range:
            tbl1 = load_data(D, "pop", "00", "none")
            res_zero.append(np.count_nonzero(tbl1, axis=0)[-1]/1000)
            tbl2 = load_data(D, "pop", "00", "cost")
            res_zero.append(np.count_nonzero(tbl2, axis=0)[-1]/1000)
        plt.axhline(np.mean(res_zero), color="k")
    if save:
        plt.savefig("figs/{0}_{1}.png".format(data, scenario))

def compare_ts(D, save=True):
    tb_none = load_data(D, "rel_plas", "06")
    tb_cost = load_data(D, "rel_plas", "06", "cost") 
    tb_noise = load_data(D, "rel_plas", "06", "noise")
    plt.figure()
    plt.title("D = " + str(D), fontsize=18)
    plt.plot(np.mean(tb_none, axis=0), label="b=0.6, c=0, s=0", color='k') #
    plt.plot(np.mean(tb_cost, axis=0), label="b=0.6, c=1, s=0", color='k', linestyle=":") #
    plt.plot(np.mean(tb_noise, axis=0), label="b=0.6, c=0, s=0.6", color='k', linestyle="--") #, color='k', linestyle="-."
    plt.legend()
    plt.xlabel("Generation", fontsize=16) 
    plt.ylabel("Relative plasticity", fontsize=16)      
    if save:
        plt.savefig("figs/compare_ts_D{0}.png".format(D))
        
def compare_ts_fig(Dlist=[2,4,6,8], data="rel_plas", ylab="Relative plasticity", ylimit=1, leg_loc='center left', save=True):
    plt.figure(figsize=(12,10))
    for i, D in enumerate(Dlist):
        tb_none = load_data(D, data, "06")
        tb_cost = load_data(D, data, "06", "cost") 
        tb_noise = load_data(D, data, "06", "noise")
        tb_envnoise = load_data(D, data, "1_0", "env_noise_corr05")
        print(tb_envnoise.shape)
        plt.subplot(2,2,i+1)
        # plt.title("Developmental limit = " + str(round(D/6,2)), fontsize=14)
        # plt.text(45, ylimit*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
        plt.plot(np.mean(tb_none, axis=0)[:301], label=r"b=0.6, c=0, s=0, $\tau$=0", color='k') #
        plt.plot(np.mean(tb_cost, axis=0)[:301], label=r"b=0.6, c=1, s=0, $\tau$=0", color='k', linestyle=":") #
        plt.plot(np.mean(tb_noise, axis=0)[:301], label=r"b=0.6, c=0, s=0.6, $\tau$=0", color='k', linestyle="--")
        plt.plot(np.mean(tb_envnoise, axis=0)[:301], label=r"b=0.6, c=0, s=0, $\tau$=1, $\rho$=0.5", color='k', linestyle="-.")#, color='k', linestyle="-."
        if i==0:
            if data == "rel_plas":
                leg = plt.legend(loc=leg_loc, prop={'size': 14})
                leg.get_frame().set_edgecolor('k')
            plt.ylabel(ylab, fontsize=22)
            if data == "NPcovar":
                plt.text(45, ylimit*0.92*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
                plt.text(-5, ylimit*0.9*0.9, "A", fontsize=24)
            else:
                plt.text(45, ylimit*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
                plt.text(-5, ylimit*0.9, "A", fontsize=24)
        elif i == 1:
            if data == "NPcovar":
                plt.text(45, ylimit*0.92*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
                plt.text(-5, ylimit*0.9*0.9, "B", fontsize=24)
            else:
                plt.text(45, ylimit*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
                plt.text(-5, ylimit*0.9, "B", fontsize=24)
        elif i == 2:
            if data == "NPcovar":
                plt.text(45, ylimit*0.92*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
                plt.text(-5, ylimit*0.9*0.9, "C", fontsize=24)
            else:
                plt.text(45, ylimit*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
                plt.text(-5, ylimit*0.9, "C", fontsize=24)
            plt.xlabel("Generation", fontsize=22)
            plt.ylabel(ylab, fontsize=22)
        elif i==3:
            # if data == "N_var" or "P_Var" or "NPCovar":
            #     leg = plt.legend(loc=leg_loc, prop={'size': 14})
            #     leg.get_frame().set_edgecolor('k')
            plt.xlabel("Generation", fontsize=22)
            if data == "NPcovar":
                plt.text(45, ylimit*0.92*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
                plt.text(-5, ylimit*0.9*0.9, "D", fontsize=24)
            else:
                plt.text(45, ylimit*0.92, "Developmental limit = " + str(round(D/6,2)), fontsize=18)
                plt.text(-5, ylimit*0.9, "D", fontsize=24)
        if data == "NPcovar":
            plt.ylim(-ylimit, ylimit)
        else:
            plt.ylim(0, ylimit)
        plt.xticks(np.arange(0,325,25), [  0,  None,  50,  None, 100, None, 150, None, 200, None, 250, None, 300], fontsize=18)
        if data == "rel_plas":
            plt.yticks(np.arange(0,1.1,0.1), [0. , None, 0.2, None, 0.4, None, 0.6, None, 0.8, None, 1. ], fontsize=18)
        else:
            plt.yticks(fontsize=18)
        plt.tick_params(length=10)
        # if i in [2,3]:
        #     plt.xlabel("Generation", fontsize=20)
        # if i in [0,2]:
        #     plt.ylabel("Relative plasticity", fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig("figs/compare_ts_{0}.png".format(data))
        
def compare_ts_envnoise(Dlist=[2,4,6,8], typ="corr", save=True):
    plt.figure(figsize=(12,10))
    for i, D in enumerate(Dlist):
        # print(D)
        scen = "env_noise_{0}".format(typ)
        # tb_25 = load_data(D, "rel_plas", "0_25", scen)
        tb_50 = load_data(D, "rel_plas", "0_5", scen) 
        # tb_75 = load_data(D, "rel_plas", "0_75", scen)
        tb_100 = load_data(D, "rel_plas", "1_0", scen)
        # tb_150 = load_data(D, "rel_plas", "1_5", scen)
        # tb_200 = load_data(D, "rel_plas", "2_0", scen)
        # tb_300 = load_data(D, "rel_plas", "3_0", "env_noise")
        # tb_400 = load_data(D, "rel_plas", "4_0", "env_noise")
        plt.subplot(2,2,i+1)
        # plt.title("Developmental limit = " + str(round(D/6,2)), fontsize=14)
        plt.text(0, 0.9, "Developmental limit = " + str(round(D/6,2)), fontsize=14)
        # plt.plot(np.mean(tb_25, axis=0)[:301], label=r"$\tau$=0.25") #
        plt.plot(np.mean(tb_50, axis=0)[:301], label=r"$\tau$=0.5") #, color='k', linestyle=":")
        # plt.plot(np.mean(tb_75, axis=0)[:301], label=r"$\tau$=0.75") #, color='k', linestyle="--"
        plt.plot(np.mean(tb_100, axis=0)[:301], label=r"$\tau$=1")#, color='k', linestyle="-."
        # plt.plot(np.mean(tb_150, axis=0)[:301], label=r"$\tau$=1.5")
        # plt.plot(np.mean(tb_200, axis=0)[:301], label=r"$\tau$=2")
        # plt.plot(np.mean(tb_300, axis=0)[:301], label=r"$\tau$=3")
        # plt.plot(np.mean(tb_400, axis=0)[:301], label=r"$\tau$=4")
        if i==3:
            leg = plt.legend(prop={'size': 14})
            leg.get_frame().set_edgecolor('k')
        plt.ylim(0,1)
        plt.xticks(np.arange(0,325,25), [  0,  None,  50,  None, 100, None, 150, None, 200, None, 250, None, 300], fontsize=16)
        plt.yticks(np.arange(0,1.1,0.1), [0. , None, 0.2, None, 0.4, None, 0.6, None, 0.8, None, 1. ], fontsize=16)
        if i in [2,3]:
            plt.xlabel("Generation", fontsize=18)
        if i in [0,2]:
            plt.ylabel("Relative plasticity", fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig("figs/compare_ts_envnoise_{0}.png".format(typ))

def cost_difference(D_range=np.arange(1,9), save=True):
    plt.figure()
    for var in ["02", "04", "06", "08", "10"]: #"00", 
        results = []
        for D in D_range:
            surv_none = load_data(D, "pop", var, "none")
            surv_cost = load_data(D, "pop", var, "cost")
            surv_diff = (np.count_nonzero(surv_none, axis=0)[-1] - np.count_nonzero(surv_cost, axis=0)[-1])/1000
            results.append(surv_diff)
        var = var[0] + "." + var[1]
        plt.plot(D_range/6, results, color="k", marker=marker_dct[var], markersize=10, label=var)
    plt.legend(title="b = ")
    plt.xlabel("Developmental limit", fontsize=16)
    plt.ylabel("Difference in survival", fontsize=16)
    if save:
        plt.savefig("figs/cost_difference.png")
        
def assimilation_speed(D_range=np.arange(1,9), save=True):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Plasticity cost")
    for var in ["02", "04", "06", "08", "10"]: #"00", 
        results = []
        for D in D_range:
            print(var, D)
            relplas_cost = load_data(D, "rel_plas", var, "cost")
            # max_300 = np.mean(relplas_cost, axis=0)[301]/np.max(np.mean(relplas_cost, axis=0)) #ratio of plasticity at t=300 to maximum relative plasticity
            speed = (np.max(np.mean(relplas_cost, axis=0)) - np.mean(relplas_cost, axis=0)[301])/(300 - np.argmax(np.mean(relplas_cost, axis=0)))
            results.append(speed) #max_300
        var = var[0] + "." + var[1]
        plt.plot(D_range/6, results, color="k", marker=marker_dct[var], markersize=10, label=var)
    plt.legend(title="b = ")
    plt.xlabel("Developmental limit", fontsize=16)
    plt.ylabel("Speed of genetic assimilation", fontsize=16)
    plt.subplot(1,2,2)
    plt.title("Developmental noise")
    for var in ["02", "04", "06", "08", "10"]: #"00", 
        results = []
        for D in D_range:
            print(var, D)
            relplas_noise = load_data(D, "rel_plas", var, "noise")
            # max_300 = np.mean(relplas_noise, axis=0)[301]/np.max(np.mean(relplas_noise, axis=0)) #ratio of plasticity at t=300 to maximum relative plasticity
            speed = (np.max(np.mean(relplas_noise, axis=0)) - np.mean(relplas_noise, axis=0)[301])/(300 - np.argmax(np.mean(relplas_noise, axis=0)))
            results.append(speed) #max_300
        var = var[0] + "." + var[1]
        plt.plot(D_range/6, results, color="k", marker=marker_dct[var], markersize=10, label=var)
    plt.legend(title="s = ")
    plt.xlabel("Developmental limit", fontsize=16)
    if save:
        plt.savefig("figs/assimilation_speed.png")
    
        
# #  FIG. 3: SURVIVAL figure  ##   
# plt.figure(figsize=(9,8))      
# plt.subplot(2,2,1)
# cross_D_comparison("Probability of survival", "survival", "none", save=False, leg_loc="lower right")
# plt.text(0.15, 0.95, "A", fontsize=20)
# plt.subplot(2,2,2)
# cross_D_comparison(False, "survival", "cost", save=False, leg_loc="lower right")
# plt.text(0.15, 0.95, "B", fontsize=20)
# plt.subplot(2,2,3)
# plt.xlabel("Developmental limit", fontsize=18)
# cross_D_comparison("Probability of survival", "survival", "noise", save=False, leg_loc="center right")
# plt.text(0.15, 0.95, "C", fontsize=20)
# plt.subplot(2,2,4)
# plt.xlabel("Developmental limit", fontsize=18)
# cross_D_comparison(False, "survival", "env_noise_corr05", save=False, leg_loc="center right")
# plt.text(0.15, 0.95, "D", fontsize=20)
# plt.tight_layout()
# plt.savefig("figs/survival.png")

# # # FIG. 4: RELATIVE PLASTICITY figure  ## 
# plt.figure(figsize=(9,8))
# plt.subplot(2,2,1)
# cross_D_comparison("Relative plasticity", "rel_plas", "none", save=False, leg_loc="lower right")
# plt.text(0.15, 0.95, "A", fontsize=20)
# plt.subplot(2,2,2)
# cross_D_comparison(False, "rel_plas", "cost", save=False, leg_loc="upper center")
# plt.text(0.15, 0.95, "B", fontsize=20)
# plt.subplot(2,2,3)
# plt.xlabel("Developmental limit", fontsize=18)
# cross_D_comparison("Relative plasticity", "rel_plas", "noise", save=False, leg_loc=(0.02, 0.35))
# plt.text(0.15, 0.95, "C", fontsize=20)
# plt.subplot(2,2,4)
# plt.xlabel("Developmental limit", fontsize=18)
# cross_D_comparison(False, "rel_plas", "env_noise_corr05", save=False, leg_loc=(0.76, 0.03))
# plt.text(0.15, 0.95, "D", fontsize=20)
# plt.tight_layout()
# plt.savefig("figs/rel_plas.png")

# #  FIG. 5: RELATIVE PLASTICITY timeseries   ##
# compare_ts_fig()

# #  FIG. S1: PLASTIC GENETIC VARIANCE ts  ###
# compare_ts_fig(data="P_var", ylab="Plastic genetic variance", ylimit=0.1, leg_loc=(0.21,0.5))

# #  FIG. S2: NON-PLASTIC GENETIC VARIANCE ts  ###
# compare_ts_fig(data="N_var", ylab="Non-plastic genetic variance", ylimit=0.13, leg_loc=(0.21,0.58))

# #  FIG. S3: PLASTIC & NON-PLASTIC GENETIC VARIANCE ts  ###
# compare_ts_fig(data="NPcovar", ylab="Genetic covariance", ylimit=0.008, leg_loc="lower right")

#  FIG. S4: RELATIVE PLASTICITY FOR RHO=0 and RHO=1
plt.figure(figsize=(9,4))      
plt.subplot(1,2,1)
plt.xlabel("Developmental limit", fontsize=18)
plt.text(0.15, 0.95, "A", fontsize=20)
cross_D_comparison("Relative plasticity", "rel_plas", "env_noise_uncorr", save=False, leg_loc="lower right")
plt.text(0.8, 0.2, r"$\rho$ = 0", fontsize=16)
# plt.subplot(1,3,2)
# plt.xlabel("Developmental limit", fontsize=16)
# cross_D_comparison(False, "survival", "env_noise_corr05", save=False)
# plt.text(0.8, 0.2, r"$\rho$ = 0.5", fontsize=14)
plt.subplot(1,2,2)
plt.xlabel("Developmental limit", fontsize=18)
plt.text(0.15, 0.95, "B", fontsize=20)
cross_D_comparison(False, "rel_plas", "env_noise_corr", save=False, leg_loc="lower right")
plt.text(0.8, 0.2, r"$\rho$ = 1", fontsize=16)
plt.tight_layout()
plt.savefig("figs/rel_plas_envnoise.png")





##  RELATIVE PLASTICITY figure  (all env noise cases)  ## 
# plt.figure(figsize=(15,5))      
# plt.subplot(1,3,1)
# plt.xlabel("Developmental limit", fontsize=16)
# cross_D_comparison("Relative plasticity", "rel_plas", "env_noise_uncorr", save=False)
# plt.text(0.8, 0.2, r"$\rho$ = 0", fontsize=14)
# plt.subplot(1,3,2)
# plt.xlabel("Developmental limit", fontsize=16)
# cross_D_comparison(False, "rel_plas", "env_noise_corr05", save=False)
# plt.text(0.8, 0.2, r"$\rho$ = 0.5", fontsize=14)
# plt.subplot(1,3,3)
# plt.xlabel("Developmental limit", fontsize=16)
# cross_D_comparison(False, "rel_plas", "env_noise_corr", save=False)
# plt.text(0.8, 0.2, r"$\rho$ = 1", fontsize=14)
# plt.tight_layout()
# plt.savefig("figs/rel_plas_envnoise.png")

# plt.figure(figsize=(5,11))
# plt.subplot(3,1,1)
# cross_D_comparison("Plastic variance", "P_var", "none", save=False)
# plt.ylim(0,0.1)
# plt.subplot(3,1,2)
# cross_D_comparison("Plastic variance", "P_var", "cost", save=False)
# plt.ylim(0,0.1)
# plt.subplot(3,1,3)
# cross_D_comparison("Plastic variance", "P_var", "noise", save=False)
# plt.xlabel("Developmental limit", fontsize=16)
# plt.ylim(0,0.1)
# plt.tight_layout()
# plt.savefig("figs/P_var.png")

# plt.figure(figsize=(5,11))
# plt.subplot(3,1,1)
# cross_D_comparison("Nonplastic variance", "N_var", "none", save=False)
# plt.ylim(0,0.05)
# plt.subplot(3,1,2)
# cross_D_comparison("Nonplastic variance", "N_var", "cost", save=False)
# plt.ylim(0,0.05)
# plt.subplot(3,1,3)
# cross_D_comparison("Nonplastic variance", "N_var", "noise", save=False)
# plt.xlabel("Developmental limit", fontsize=16)
# plt.ylim(0,0.05)
# plt.tight_layout()
# plt.savefig("figs/N_var.png")

# compare_ts(8)

# compare_ts_envnoise(Dlist=[2,4,6,8], typ="uncorr")



# cost_difference()

# assimilation_speed()

# survival("none")
# make_fig("Relative phenotype", "rel_phen", "none")
# make_fig("Relative plasticity", "rel_plas", "none")
# make_fig("Nonplastic genetic variance", "N_var", "none", lim=False)
# make_fig("Plastic genetic variance", "P_var", "none", lim=False)
# make_fig("N-P covariance", "NPcovar", "none", lim=False)

# survival("cost")
# make_fig("Relative phenotype", "rel_phen", "cost")
# make_fig("Relative plasticity", "rel_plas", "cost")
# make_fig("Nonplastic genetic variance", "N_var", "cost", lim=False)
# make_fig("Plastic genetic variance", "P_var", "cost", lim=False)
# make_fig("N-P covariance", "NPcovar", "cost", lim=False)

# survival("noise")
# make_fig("Relative phenotype", "rel_phen", "noise")
# make_fig("Relative plasticity", "rel_plas", "noise")
# make_fig("Phenotypic variance", 3, "phen_var", "noise", lim=False)
# make_fig("Nonplastic genetic variance", "N_var", "noise", lim=False)
# make_fig("Plastic genetic variance", "P_var", "noise", lim=False)
# make_fig("N-P covariance", "NPcovar", "noise", lim=False)

