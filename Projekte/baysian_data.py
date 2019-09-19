#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:33:58 2019

@author: pandoora

baysian
"""

# Import libraries
import pandas as pd
import numpy as np

# Number of random draws from the prior
n_draws = 10000
space = 16
observed_data = 6
#np.random.binomial(n_draws, p, size) [n for n in parameters if np.round(n, decimals =3) == np.round(6/16, decimals=3)]

# Here you sample n_draws draws from the prior into a pandas Series (to have convenient
# methods available for histograms and descriptive statistics, e.g. median)
prior = pd.Series(np.random.uniform(0, 1, size = n_draws))  

prior.hist() # It's always good to eyeball the prior to make sure it looks ok.

# Here you define the generative model
def generative_model(parameters):
    
    
    return np.random.binomial(16, parameters)



# Here you simulate data using the parameters from the prior and the 
# generative model
sim_data = list()
for p in prior:
    sim_data.append(generative_model(p))
                    
# Here you filter off all draws that do not match the data.
posterior = prior[list(map(lambda x: x == observed_data, sim_data))]

posterior.hist() # Eyeball the posterior


# See that we got enought draws left after the filtering. 
# There are no rules here, but you probably want to aim for >1000 draws.

# Now you can summarize the posterior, where a common summary is to take the mean or the median posterior, 
# and perhaps a 95% quantile interval.


print('Number of draws left: %d, Posterior median: %.3f, Posterior quantile interval: %.3f-%.3f' % 
      (len(posterior), posterior.median(), posterior.quantile(.025), posterior.quantile(.975)))









