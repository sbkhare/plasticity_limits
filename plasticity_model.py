# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:14:07 2022

@author: Sikander
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats


E0 = 0
n = 5 #number non-plastic loci
m = 5 #number plastic loci
omega_p = 1 #strength of selection on plasticity

class individual():
    def __init__(self, Nk, Pk, E, Dc, b, s):
        """Individual class
        
        Instance variables:
            Nk = non-plastic alleles (array)
            Pk = plastic alleles (array)
            E = environmental value
            Dc = size of developmental limit
            b = plasticity parameter
            s = noise parameter
            """
        self.Nk = Nk
        self.Pk = Pk
        self.E = E
        self.Dc = Dc
        self.b = b
        self.s = s
        self.zi = np.random.normal(0, 1) #zero mean, unit variance independent Gaussian deviate
        self.yi = np.random.normal(0, 1) #zero mean, unit variance independent Gaussian deviate
        self.T = (Dc/(1 + np.exp(-b*np.sum(Pk)*(E - E0))) - Dc/2)*(1 + self.yi*s) + np.sum(Nk) + self.zi
        self.plasticity = Dc/(1 + np.exp(-b*np.sum(Pk)*(E - E0))) - Dc/2 #Is Dc/4 right?
        self.plas_noise = self.yi*self.s*self.plasticity
        self.rel_plas = 0 #only for debuggin
                    
    def recalc_T(self, new_env):
        zi = np.random.normal(0, 1) #zero mean, unit variance independent Gaussian deviate
        yi = np.random.normal(0, 1) #zero mean, unit variance independent Gaussian deviate
        self.T = (self.Dc/(1 + np.exp(-self.b*np.sum(self.Pk)*(self.E - E0))) - self.Dc/2)*(1 + yi*self.s) + np.sum(self.Nk) + zi
        self.plasticity = self.Dc/(1 + np.exp(-self.b*np.sum(self.Pk)*(self.E - E0))) - self.Dc/2
        self.rel_plas = self.Dc/(1 + np.exp(-self.b*np.sum(self.Pk)*(new_env - E0))) - self.Dc/2
        
class population():
    def __init__(self, K, f, omega2, alpha2, mu, size, generations, Dc, b, s):
        """Population class
        
        Instance variables:
            K = max population size
            f = fecundity
            omega2 = width of selection function
            alpha2 = variance of mutations
            mu = mutation rate
            size = size of environmental step change
            generations = number of generations after env change
            Dc = size of developmental limit
            b = plasticity parameter
            s = noise parameter
            """
        self.individuals = []
        self.K = K
        self.f = f
        self.omega2 = omega2
        self.alpha2 = alpha2 #mutation variance
        self.mu = mu
        self.size = size #step change size
        self.generations = generations
        self.Dc = Dc
        self.b = b
        self.s = s
        self.rel_phen = np.zeros(generations+1)
        self.rel_plas = np.zeros(generations+1)
        self.phen_var = np.zeros(generations+1)
        self.N_var = np.zeros(generations+1)
        self.P_var = np.zeros(generations+1)
        self.NPcovar = np.zeros(generations+1)
        self.pop = np.zeros(generations+1)
        self.survival = np.zeros(generations+1)
        self.prev_E = 0
    
    def initialize(self):
        """Initialize population"""
        N_lst = []
        P_lst = []
        #initiated nonplastic alleles with same steady state genetic variance: sel-mut-drift balance
        for i in range(self.K):
            Ne = self.f*self.K/(self.f - 1)
            Vs = self.omega2 + 1 #width of Guassian selection function and environemental variance 
            exp_var = 4*n*self.mu*self.alpha2*Ne/(1 + self.alpha2*Ne/Vs)
            Ni = np.random.normal(loc=0, scale=np.sqrt(exp_var), size=(n,2))
            Pi = np.random.normal(loc=0, scale=np.sqrt(exp_var), size=(m,2))
            new_indiv = individual(Ni, Pi, 0, self.Dc, self.b, self.s)
            self.individuals.append(new_indiv)
            N_lst.append(np.sum(Ni))
            P_lst.append(np.sum(Pi))
    
    # def record_data(self, E, i):
    #     Topt = E
    #     T_list = []
    #     P_list = []
    #     rel_T = np.sum
        
    def step(self, E, c=1, gen=False, equil=False, tau=0, rho=0):
        """Simulate one generaton: selection, reproduction, and development"""
        individuals_nextstep = [] 
        T_list = []
        N_list = []
        P_list = []
        plas_list = []
        devnoise_list = []
        Topt = self.prev_E
        if gen: # if this is not an equilibration step
            ## Autocorrelation between environment of deveopement and selection ###
            prev_deviate = self.prev_E - self.size
            xi = np.random.normal(0, 1) #deviate for environment of selection
            Topt = self.size + rho*prev_deviate + tau*xi*np.sqrt(1 - rho**2) #so that selection and devlopement happen in the same environment
        self.prev_E = E
        #reporoduction folows selection
        survivors = []
        for indiv in self.individuals:  #selection
            Wi = np.exp(-0.5*(((indiv.T - Topt)**2)/self.omega2 + c*(np.sum(indiv.Pk)/omega_p)**2))
            if np.random.rand() < Wi: #survive with probability Wi
                survivors.append(indiv)
                # if gen:
                #     T_list.append(indiv.T)
                #     N_list.append(np.sum(indiv.Nk))
                #     P_list.append(np.sum(indiv.Pk))
                #     plas_list.append(indiv.plasticity)
        if len(survivors) == 0:
            self.individuals = []
            print("Population extinct")
            return
        #If fewer than K individuals, all mate as female, if more than K individuals, select K without replacement
        females = np.random.choice(survivors, size=min(self.K, len(survivors)), replace=False)
        males = np.random.choice(survivors, size=min(self.K, len(survivors)), replace=True)
        for fm, ml in zip(females, males): #reproduction
            for j in range(self.f): #fecundity
                #produce "female" gamete
                Ngamete1 = np.choose(np.random.randint(0,2,n), fm.Nk.T) #free recombination
                Pgamete1 = np.choose(np.random.randint(0,2,m), fm.Pk.T) #free recombination
                #produce "male" gamete
                Ngamete2 = np.choose(np.random.randint(0,2,n), ml.Nk.T) #free recombination
                Pgamete2 = np.choose(np.random.randint(0,2,m), ml.Pk.T) #free recombination
                #mutate
                Ngamete1[:] = np.where(np.random.rand(*Ngamete1.shape) < self.mu, 
                                       Ngamete1 + stats.norm.rvs(loc=0, scale=np.sqrt(self.alpha2), size = Ngamete1.shape), 
                                       Ngamete1)
                Ngamete2[:] = np.where(np.random.rand(*Ngamete2.shape) < self.mu, 
                                       Ngamete2 + stats.norm.rvs(loc=0, scale=np.sqrt(self.alpha2), size = Ngamete2.shape), 
                                       Ngamete2)
                Pgamete1[:] = np.where(np.random.rand(*Pgamete1.shape) < self.mu, 
                                       Pgamete1 + stats.norm.rvs(loc=0, scale=np.sqrt(self.alpha2), size = Pgamete1.shape), 
                                       Pgamete1)
                Pgamete2[:] = np.where(np.random.rand(*Pgamete2.shape) < self.mu, 
                                       Pgamete2 + stats.norm.rvs(loc=0, scale=np.sqrt(self.alpha2), size = Pgamete2.shape), 
                                       Pgamete2)
                #binomial gives the wrong per haplo mutation rate
                # num_loci = np.random.binomial(len(Ngamete1), self.mu) #how many loci to be mutated
                # change = np.random.normal(0, np.sqrt(self.alpha2), num_loci) #changes due to mutations
                # Ngamete1[np.random.randint(0, len(Ngamete1), size=num_loci)] = change #chooses indices to change
                # num_loci = np.random.binomial(len(Ngamete2), self.mu) #how many loci to be mutated
                # change = np.random.normal(0, np.sqrt(self.alpha2), num_loci) #changes due to mutations
                # Ngamete2[np.random.randint(0, len(Ngamete2), size=num_loci)] = change #chooses indices to change
                # num_loci = np.random.binomial(len(Pgamete1), self.mu) #how many loci to be mutated
                # change = np.random.normal(0, np.sqrt(self.alpha2), num_loci) #changes due to mutations
                # Pgamete1[np.random.randint(0, len(Pgamete1), size=num_loci)] = change #chooses indices to change
                # num_loci = np.random.binomial(len(Pgamete2), self.mu) #how many loci to be mutated
                # change = np.random.normal(0, np.sqrt(self.alpha2), num_loci) #changes due to mutations
                # Pgamete2[np.random.randint(0, len(Pgamete2), size=num_loci)] = change #chooses indices to change
                # coded this way (below), two loci could have the same mutation 8-18 times per simulation (700-1500 generations)
                # Ngamete1[np.random.rand(*Ngamete1.shape) < self.mu] += np.random.normal(0, np.sqrt(self.alpha2))
                # Pgamete1[np.random.rand(*Pgamete1.shape) < self.mu] += np.random.normal(0, np.sqrt(self.alpha2))
                # Ngamete2[np.random.rand(*Ngamete2.shape) < self.mu] += np.random.normal(0, np.sqrt(self.alpha2))
                # Pgamete2[np.random.rand(*Pgamete2.shape) < self.mu] += np.random.normal(0, np.sqrt(self.alpha2))
                #produce offspring
                newNk = np.zeros(shape=(n,2), dtype=float)
                newPk = np.zeros(shape=(m,2), dtype=float)
                newNk[:,0] = Ngamete1
                newNk[:,1] = Ngamete2
                newPk[:,0] = Pgamete1
                newPk[:,1] = Pgamete2 
                new_indiv = individual(newNk, newPk, E, self.Dc, self.b, self.s)
                individuals_nextstep.append(new_indiv)
                if gen:
                    T_list.append(new_indiv.T)
                    N_list.append(np.sum(new_indiv.Nk))
                    P_list.append(np.sum(new_indiv.Pk))
                    plas_list.append(new_indiv.plasticity)
                    if self.s > 0:
                        devnoise_list.append(new_indiv.plas_noise)
        if gen:
            print("Saving")
            # individuals_nextstep = np.array(individuals_nextstep)
            # T_list = [indiv.T for indiv in individuals_nextstep]
            # N_list = [np.sum(indiv.Nk) for indiv in individuals_nextstep]
            # P_list = [np.sum(indiv.Pk) for indiv in individuals_nextstep]
            self.rel_phen[gen] = np.mean(T_list)/self.size
            # print(np.mean(T_list)/self.size)
            if self.s > 0:
                self.rel_plas[gen] = np.mean(np.array(plas_list) + np.array(devnoise_list))/self.size
                self.phen_var[gen] = np.var(devnoise_list)*100/np.var(T_list)
            else:
                self.rel_plas[gen] = np.mean(plas_list)/self.size
            # print(np.var(T_list))
            self.N_var[gen] = np.var(N_list)
            # print(np.var(N_list))
            self.P_var[gen] = np.var(P_list)
            # print(np.var(P_list))
            self.NPcovar[gen] = np.cov(N_list, P_list)[0,1]
            # print(np.cov(N_list, P_list)[0,1])
            self.pop[gen] = len(individuals_nextstep)
            self.survival[gen] = len(survivors)
        self.individuals = individuals_nextstep
    
    def equilibrate(self, nonzero=False, show_var=True):
        """Simulation of equilibration period"""
        self.initialize()
        for i in range(200):
            if i%50 == 0:
                print("Equilibration generation: ", i)
                print("Population: ", len(self.individuals))
                if show_var:
                    N_lst = []
                    P_lst = []
                    T_lst = []
                    PT_lst = []
                    for indiv in self.individuals:
                        N_lst.append(np.sum(indiv.Nk))
                        P_lst.append(np.sum(indiv.Pk))
                        T_lst.append(indiv.T)
                        PT_lst.append(indiv.plasticity)
                    print("Sum N: ", np.mean(N_lst), ", Variance: ", np.var(N_lst))
                    print("Sum P: ", np.mean(P_lst), ", Variance: ", np.var(P_lst))
                    print("Relative phenotype: ", np.mean(T_lst)/self.size)
                    print("Relative plasticity: ", (self.Dc/(1 + np.exp(-self.b*np.mean(P_lst)*(self.size - E0))) - self.Dc/2)/self.size)
            self.step(E=0)
            if i == 199:
                T_list = [indiv.T for indiv in self.individuals]
                N_list = [np.sum(indiv.Nk) for indiv in self.individuals]
                P_list = [np.sum(indiv.Pk) for indiv in self.individuals]
                self.rel_phen[0] = np.mean(T_list)/self.size
                self.rel_plas[0] = np.mean([indiv.plasticity for indiv in self.individuals])/self.size
                self.phen_var[0] = np.var(T_list)
                self.N_var[0] = np.var(N_list)
                self.P_var[0] = np.var(P_list)
                self.NPcovar[0] = np.cov(N_list, P_list)[0,1]
                self.pop[0] = len(self.individuals)
        if nonzero:
            print("End equilibration")
            T_lst = []
            RP_lst = []
            for indiv in self.individuals:
                indiv.Pk += 0.2/(2*m*self.b) #is this right?
                indiv.recalc_T(new_env=self.size)
                T_lst.append(indiv.T)
                RP_lst.append(indiv.rel_plas)
            print("Relative phenotype: ", np.mean(T_lst)/self.size)
            print("Relative plasticity: ", np.mean(RP_lst)/self.size)
        
                
            
    def simulate(self, cost=0, tau=False, rho=0.5, nonzero=False):
        """Full simulation for the specified number of generations"""
        self.equilibrate()
        for gen in range(1, self.generations+1):
            print("Generation ", gen)
            if len(self.individuals) != 0:
                if tau: #environemntal noise
                    deviate = np.random.normal(0, tau)
                    self.step(E=self.size+deviate, c=cost, gen=gen, tau=tau, rho=rho)
                else:
                    self.step(E=self.size, c=cost, gen=gen)
            else:
                print("Population extinct")
        
            