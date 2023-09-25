# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:22:56 2022

@author: Sikander
"""
from plasticity_model import population
import numpy as np

stepchange = 3
generations = 200
plasticity = 0
developnoise = 0

p = population(256, 4, 1, 0.05, 0.0005, stepchange, generations, plasticity, developnoise)
p.simulate()