#%%
import numpy as np
from FuNVol import FuNVol
import math
import torch
import random
import matplotlib.pyplot as plt
import pickle as pickle
import datetime
from tqdm import tqdm

#%%
# perform FPC projection
model= FuNVol('./data/', ['AMZN', 'IBM', 'INTC', 'TSLA'], K=8)

#%%
# learn the neural SDE model for time series of FPCCs and prices
model.Learn_Neural_SDE()
  
# minimize MSE in stage 1
for i in range(70):
    
    now = datetime.datetime.now()
    print(i, str(now))
     
    model.neural_sde.Train(n_iter=10_000, LOSS='MSE') 
    model.neural_sde.Plot_Band()
    model.neural_sde.Plot_Band(M=2593)
    model.neural_sde.Plot_Loss()
    model.neural_sde.Plot_PIT()
    
# load saved model with least MSE
with open('least_MSE.pkl', 'rb') as inp:
    model.neural_sde = pickle.load(inp)
        
# minimize penalized negative log-likelihood by fixing the drift neural network in stage 2
for i in range(20):
    
    now = datetime.datetime.now()
    print(i, str(now))
     
    model.neural_sde.Train(n_iter=10_000, LOSS='LL_diffusion') 
    model.neural_sde.Plot_Band()
    model.neural_sde.Plot_Band(M=2593)
    model.neural_sde.Plot_Loss()
    model.neural_sde.Plot_PIT()

# load saved model with least penalized negative log-likelihood at end of stage 2
with open('least_penalized_LL_diffusion.pkl', 'rb') as inp:
    model.neural_sde = pickle.load(inp)

# minimize penalized negative log-likelihood by learning both the drift and diffusion neural networks in stage 3    
for i in range(20):
    
    now = datetime.datetime.now()
    print(i, str(now))
     
    model.neural_sde.Train(n_iter=10_000, LOSS='LL_combined') 
    model.neural_sde.Plot_Band()
    model.neural_sde.Plot_Band(M=2593)
    model.neural_sde.Plot_Loss()
    model.neural_sde.Plot_PIT()

# load saved model with least penalized negative log-likelihood at end of stage 3 to be used as final learnt model
with open('least_penalized_LL_combined.pkl', 'rb') as inp:
    model.neural_sde = pickle.load(inp)
    
#%%
# Generate data for future scenarios
# Can provide the number of independent scenarios and time steps as arguments to the function

with torch.no_grad():
    delta_grid_t, tau_grid_t, IV, price, b_sim = model.Generate_Data(nsims=100_000, nsteps=31)
    
""" delta_grid_t and tau_grid_t are lists containing the transformed delta-tau meshgrid for each equity
By default, IV values are calculated on the same grid as that for which data was available
IV is a list containing the implied vol values on the above grid for each equity
price is a list containing price paths for each equity
b_sim contains the simulated time series of FPCCs and equity prices """

""" b_sim is nsims x nsteps x 36 (4 equities times 8 FPCCs plus 1 price) dimensional
The first index corresponds to the generated (independent) scenarios
The second index corresponds to the sequence of days where day 0 is the last day of training (observed) and the remaining 29 days are generated coefficients
Hence day 0 will give the FPC coefficients (FPCCs) for the IV surface that is observed on the last day of training and is not a synthetic generated surface
The last index corresponds to the different assets' FPCCs and transformed equity prices, details for third index below:
0-7 give FPCCs for AMZN, 8 gives transformed equity price for AMZN
9-16 give FPCCs for IBM, 17 gives transformed equity price for IBM
18-25 give FPCCs for INTC, 26 gives transformed equity price for INTC
27-34 give FPCCs for TSLA, 35 gives transformed equity price for TSLA """

""" Each element of price is nsims x nsteps dimensional """

"""Each element in the list IV is nsims x nsteps x len(tau) x len(delta) dimensional
Each element of delta_grid_t and tau_grid_t is len(tau) x len(delta) containing the transformed
values of delta and tau on a grid at which IV is calculated"""

# to plot surfaces and get IV values at a different grid, refer to generated_data.ipynb
with open('sim_data_coeffs.npy', 'wb') as f:
    np.save(f, b_sim)
    
with open('sim_data_prices.npy', 'wb') as f:
    np.save(f, np.array(price).transpose(1,2,0))
