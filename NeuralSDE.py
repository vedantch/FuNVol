#%%
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import copy
import pdb

#%%
# neural network to model the drift of the SDE
class GRU_Net_Drift(nn.Module):

    def __init__(self, nIn, nHidden, nLayers, nOut):
        super(GRU_Net_Drift, self).__init__()

        self.nIn = nIn
        self.nHidden = nHidden
        self.nLayers = nLayers
        self.nOut = nOut

        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # GRU layers of the drift neural network
        self.gru_drift = torch.nn.GRU(input_size=nIn, hidden_size=nHidden, num_layers=nLayers, batch_first=True).to(self.device)

        # feed-forward neural network for mapping hidden states of the GRU at the last time step into drift
        self.prop_gru_to_drift = nn.Linear(nHidden*nLayers, nOut).to(self.device)
            

    def forward(self, x):

        self.gru_drift.flatten_parameters()
        # extract hidden states at final time step when input time series is x
        _, h = self.gru_drift(x)

        # hidden states of GRU to drift - no activation
        nu = self.prop_gru_to_drift(h.transpose(0, 1).flatten(start_dim=1))

        return nu
 
#%%
# neural network to model the diffusion of the SDE
class GRU_Net_Diffusion(nn.Module):

    def __init__(self, nIn, nHidden, nLayers, nOut):
        super(GRU_Net_Diffusion, self).__init__()

        self.nIn = nIn
        self.nHidden = nHidden
        self.nLayers = nLayers
        self.nOut = nOut

        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # GRU layers of the diffusion neural network
        self.gru_diffusion = torch.nn.GRU(input_size=nIn, hidden_size=nHidden, num_layers=nLayers, batch_first=True).to(self.device)
        
        # feed-forward neural network for mapping hidden states of the GRU at the last time step 
        # into Cholesky decomposition of diffusion
        self.prop_gru_to_L = nn.Linear(nHidden*nLayers, int(nOut * (nOut + 1) / 2)).to(self.device)
        
        self.prop_gru_to_L.bias.data.fill_(0.001)
        
        initial_wt = torch.empty(int(nOut * (nOut + 1) / 2), nHidden*nLayers)
        nn.init.uniform_(initial_wt, -0.01, 0.01).to(self.device)
        with torch.no_grad():
            self.prop_gru_to_L.weight.copy_(initial_wt)
            

    def forward(self, x):

        self.gru_diffusion.flatten_parameters()
        # extract hidden states at final time step when input time series is x
        _, h = self.gru_diffusion(x)     

        # following steps generate the diffusion by getting its Cholesky decomposition L first
        L = torch.zeros((h.shape[1], self.nOut, self.nOut)).to(self.device)

        tril_indices = torch.tril_indices(row=self.nOut, col=self.nOut, offset=0).to(self.device)

        L[:, tril_indices[0], tril_indices[1]] = self.prop_gru_to_L(h.transpose(0, 1).flatten(start_dim=1))

        I = torch.zeros(L.shape).to(self.device)
        rng = range(L.shape[1])
        I[:, rng, rng] = 1

        # L L' + eps**2 * I
        Sigma = torch.matmul(L, torch.transpose(L, 1, 2)) + 1e-3 * I
        self.L = L

        return Sigma

#%%
def MovingAverage(x, n):
    y = np.zeros(len(x))
    y[0] = np.NaN

    for i in range(1, len(x)):
        y[i] = np.mean(x[np.maximum(0, i - n):i])

    return y

#%%
class NeuralSDE:

    def __init__(self, data, T, ticker, params, lr=1e-3, n_lags=10):

        self.params = params
        self.ticker = ticker

        self.normalize = params["normalize"]

        self.pi_drift = GRU_Net_Drift(nIn=params["n_features"],
                               nLayers=params["n_GRU_layers"],
                               nHidden=params["GRU_hidden_size"],
                               nOut=params["nOut"])

        self.device = self.pi_drift.device
        
        self.pi_diffusion = GRU_Net_Diffusion(nIn=params["n_features"],
                               nLayers=params["n_GRU_layers"],
                               nHidden=params["GRU_hidden_size"],
                               nOut=params["nOut"])
        
        self.n_lags = n_lags

        self.qtl = []  # stores coefficients for normalizing data
        
        self.data = torch.from_numpy(self.Normalize_Data(data)).float().to(self.device)
        
        self.T = torch.from_numpy(T).to(self.device)
        # time difference between 2 consecutive observations
        self.dT = torch.diff(self.T, axis=0)
        
        # store the time series as collection of data sequences of length n_lags
        self.data_seq = torch.zeros((self.data.shape[0]-self.n_lags, self.n_lags, self.data.shape[1])).to(self.device)
        for i in range(self.data.shape[0]-self.n_lags):
            self.data_seq[i, :, :] = self.data[i :i+ self.n_lags, :]

        # create lists to store loss metrics as training proceeds
        self.loss_mse = []
        self.loss_mse_comb = []
        self.loss_ll = []
        self.loss_density = []
        self.loss_ll_penalized = []
        self.loss_mse_min = np.inf
        self.loss_ll_min = np.inf
        self.loss_ll_penalized_min = np.inf
        
        # parameters, optimizer and scheduler corresponding to the different training objectives
        params_mse = list(self.pi_drift.gru_drift.parameters())
        params_mse += list(self.pi_drift.prop_gru_to_drift.parameters())            
        self.optimizer_mse = optim.AdamW(params_mse, lr)
        self.scheduler_mse = optim.lr_scheduler.StepLR(self.optimizer_mse, step_size=20_000, gamma=0.9)
             
        params_LL_diffusion = list(self.pi_diffusion.gru_diffusion.parameters())
        params_LL_diffusion += list(self.pi_diffusion.prop_gru_to_L.parameters())
        self.optimizer_LL_diffusion = optim.AdamW(params_LL_diffusion, lr) 
        self.scheduler_LL_diffusion = optim.lr_scheduler.StepLR(self.optimizer_LL_diffusion, step_size=20_000, gamma=0.8)
        
        params_LL_combined = params_mse
        params_LL_combined += params_LL_diffusion
        self.optimizer_LL_combined = optim.AdamW(params_LL_combined, lr=1e-4)
        self.scheduler_combined = optim.lr_scheduler.StepLR(self.optimizer_LL_combined, step_size=20_000, gamma=0.9)
        
        sqrt_2 = 1.4142135623730951
        # CDF of standard normal distribution
        self.Phi = lambda x: 0.5 * (1 + torch.erf(x/sqrt_2) )
        sqrt_2pi = 2.5066282746310002
        # PDF of standard normal distribution
        self.phi = lambda x: torch.exp(-0.5*x**2)/sqrt_2pi
        
        # hyperparameters corresponding to weight of PIT penalty
        self.alpha_diffusion = 1
        self.alpha_combined = 100

    # normalize input data received prior to training the NeuralSDE
    def Normalize_Data(self, data):

        features = copy.deepcopy(data)
        for i in range(data.shape[1]):

            if self.normalize[i]:
                self.qtl.append(np.quantile(features[:, i], [0.25, 0.5, 0.75]))
            else:
                self.qtl.append([-0.5, 0, 0.5])

            features[:, i] -= self.qtl[i][1]
            features[:, i] /= (self.qtl[i][2] - self.qtl[i][0])

        return features

    # unnormalize synthetic data before returning for downstream usage
    def Un_Normalize_Data(self, features):

        data = copy.deepcopy(features)
        for i in range(data.shape[1]):
            data[:, i] *= (self.qtl[i][2] - self.qtl[i][0])
            data[:, i] += self.qtl[i][1]

        return data

    # function to train the neural SDE
    def Train(self, n_iter=1_000, mini_batch_size=256, LOSS='MSE'):

        # update parameters of GRU_Net_Drift by minimizing MSE
        if LOSS == 'MSE':           
            optimizer = self.optimizer_mse

        # update parameters of GRU_Net_Diffusion by minimizing penalized NLL
        elif LOSS == 'LL_diffusion':            
            optimizer = self.optimizer_LL_diffusion
          
        # update parameters of GRU_Net_Drift and GRU_Net_Diffusion by minimizing penalized NLL    
        elif LOSS == 'LL_combined':
            optimizer = self.optimizer_LL_combined
            
        try:

            for iter in tqdm(range(n_iter)):

                idx, data = self.Grab_Mini_Batch(mini_batch_size)
                X_obs = self.data[idx, :]
                
                dX = self.data[idx,:] - self.data[idx-1,:]
                dT = self.dT[idx-1]

                optimizer.zero_grad()

                nu = self.pi_drift.forward(data)  
                sigma = self.pi_diffusion.forward(data)

                if LOSS == 'MSE':
                    loss = torch.mean((nu * dT - dX) ** 2)
                    
                elif LOSS == 'LL_diffusion':
                    Z = torch.distributions.MultivariateNormal(nu * dT, sigma * dT.reshape(-1,1,1))
                    loss_ll = -torch.mean(Z.log_prob(dX))                    
                    loss_density = self.Density_Error(nu, sigma, dX, dT)/nu.shape[1]
                    loss = loss_ll + self.alpha_diffusion * loss_density
                    
                elif LOSS == 'LL_combined':
                    Z = torch.distributions.MultivariateNormal(nu * dT, sigma * dT.reshape(-1,1,1))
                    loss_ll = -torch.mean(Z.log_prob(dX))
                    loss_mse = torch.mean((nu * dT - dX) ** 2)
                    loss_density = self.Density_Error(nu, sigma, dX, dT)/nu.shape[1]
                    loss = loss_ll + self.alpha_combined * loss_density

                loss.backward()

                optimizer.step()
    
                if LOSS == 'MSE':
                    self.scheduler_mse.step()
                    self.loss_mse.append(loss.item())
                    
                    if loss.item() < self.loss_mse_min:
                        self.loss_mse_min = loss.item()
                        with open('least_MSE.pkl', 'wb') as outp:
                            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
                        
                elif LOSS == 'LL_diffusion':
                    self.scheduler_LL_diffusion.step()
                    self.loss_ll.append(loss_ll.item())
                    self.loss_ll_penalized.append(loss.item())
                    self.loss_density.append(loss_density.item())
                    
                    if loss.item() < self.loss_ll_penalized_min:
                        self.loss_ll_penalized_min = loss.item()
                        with open('least_penalized_LL_diffusion.pkl', 'wb') as outp:
                            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
                            
                    if loss_ll.item() < self.loss_ll_min:
                        self.loss_ll_min = loss_ll.item()
                        with open('least_LL_diffusion.pkl', 'wb') as outp:
                            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
                        
                elif LOSS == 'LL_combined':
                    self.scheduler_combined.step()
                    self.loss_mse_comb.append(loss_mse.item())
                    self.loss_ll.append(loss_ll.item())
                    self.loss_ll_penalized.append(loss.item())
                    self.loss_density.append(loss_density.item())
                    
                    if loss.item() < self.loss_ll_penalized_min:
                        self.loss_ll_penalized_min = loss.item()
                        with open('least_penalized_LL_combined.pkl', 'wb') as outp:
                            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
                            
                    if loss_ll.item() < self.loss_ll_min:
                        self.loss_ll_min = loss_ll.item()
                        with open('least_LL_combined.pkl' %self.ticker, 'wb') as outp:
                            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

        except KeyboardInterrupt:

            print("...stopping process")
            
    # grabs a mini-batch of data for training
    def Grab_Mini_Batch(self,  mini_batch_size=256):
         
        idx = torch.tensor(random.sample(range(self.data.shape[0] - self.n_lags-1), mini_batch_size)).to(self.device)

        x = self.data_seq[idx,:,:]

        return idx+self.n_lags, x

    # generate synthetic time series for the future
    def Simulate_Synthetic_Time_Series(self, nsims=1, nsteps=31, dT=1/365, start=None):
        
        if start is None:
            start = self.data.shape[0]+1
        
        input_data = self.data[start-1-self.n_lags:start-1,:].unsqueeze(0).repeat(nsims,1,1)
        sim_coeffs = torch.zeros((nsteps, nsims, self.data.shape[1])).to(self.device)
        
        # the data at first time step of generated time series contains the observation at last time step of the inputs
        sim_coeffs[0,:,:] = input_data[:,-1,:]
        
        for i in range(nsteps-1):
            
            nu_pred = self.pi_drift.forward(input_data)
            sigma_pred = self.pi_diffusion.forward(input_data)
            
            Z = torch.distributions.MultivariateNormal(nu_pred*dT, sigma_pred*dT)
            a_sim = Z.sample()
            
            sim_coeffs[i+1,:,:] = a_sim + sim_coeffs[i,:,:]
            input_data_clone = torch.clone(input_data)
            
            input_data[:, :self.n_lags-1,:] = input_data_clone[:,1:,:]
            input_data[:,self.n_lags-1,:] = sim_coeffs[i+1,:,:] 

        # the generated data needs to be unnormalized before it is returned
        sim_final = self.Un_Normalize_Data(sim_coeffs.reshape(-1, sim_coeffs.shape[-1]).to('cpu').detach().numpy()).reshape(sim_coeffs.shape)
        
        return sim_final
        
    # compute the PITs of the time series
    def PIT(self, data=None):
        
        if data is None:
            
            nu = self.pi_drift.forward(self.data_seq)  
            sigma = self.pi_diffusion.forward(self.data_seq)
            
            dT = self.dT[self.n_lags-1:].reshape(-1)
            dX = self.data[self.n_lags:, :] - self.data[self.n_lags-1:-1, :]
            
        else:
            
            nu = data["nu"]
            sigma = data["sigma"]
            dX = data["dX"]
            dT = data["dT"].reshape(-1)
            
        U = torch.zeros(nu.shape).to(self.device)
        
        for i in range(nu.shape[1]):

            Z = torch.distributions.normal.Normal(nu[:,i]*dT, torch.sqrt(sigma[:,i,i]*dT))
            U[:,i] = Z.cdf(dX[:,i])            
    
        return U 
    
    # compute the penalty term due to deviation of PIT's density from that of a standard uniform distribution
    def Density_Error(self, nu, sigma, dX, dT):
        
        u_batch = self.PIT({"nu" : nu,
                            "sigma" : sigma,
                            "dX" : dX,
                            "dT": dT})
        
        n = u_batch.shape[0]
        
        u = torch.linspace(0,1,101).reshape(1,-1).to(self.device)
        du = u[0,1]-u[0,0]
        
        error = 0
        # loop over each feature of the multivariate PIT
        for i in range(u_batch.shape[1]):
            
            # use 0.1 naive Silverman's estimate of bandwidth
            h = 0.1*1.069*torch.std(u_batch[:,i].detach())*(n)**(-1/5)
            
            z = (u-u_batch[:,i].reshape(-1,1))/h
            f = (1/n)*torch.sum(self.phi(z)/h, axis=0)     
            
            error += torch.sum( (f-1)**2 *du)            
        
        return error
        
    # plot the learnt confidence bands of the drift over time
    def Plot_Band(self, M=200, tickers=['AMZN', 'IBM', 'INTC', 'TSLA'], savefig=False):
        
        nu = self.pi_drift.forward(self.data_seq[-M:,:,:])
        sigma = self.pi_diffusion.forward(self.data_seq[-M:,:,:])
        
        nu = nu.to('cpu').detach().numpy()
        sigma = sigma.to('cpu').detach().numpy()
        
        data = self.data[-M:,:].to('cpu').detach().numpy()
        dT = self.dT[-M:,:].to('cpu').detach().numpy().reshape(-1)

        t = np.arange(0,M)+2603-M
        m = int(np.ceil(data.shape[1]) / 4) 
        
        if self.ticker is not None:
            n_tickers = len(self.ticker)
            tickers = self.ticker
        else:
            n_tickers = len(tickers)
        
        fig = plt.figure(figsize=(30,40))
        
        for i in range(int(data.shape[1]/n_tickers)):
            for j in range(n_tickers):
                
                ax = plt.subplot(m, 4, 4*i + j + 1)
                
                mid = self.data[-M-1:-1,9*j+i].to('cpu').numpy() + nu[:,9*j+i] * dT
                lower = mid - 2*np.sqrt(sigma[:,9*j+i,9*j+i]*dT)
                upper = mid + 2*np.sqrt(sigma[:,9*j+i,9*j+i]*dT)
        
                plt.fill_between(t, lower, upper, alpha=0.25, color='g')
                plt.locator_params(axis='y', nbins=4)
                plt.locator_params(axis='x', nbins=5)
                plt.plot(t, mid, color='k')
                plt.plot(t, data[:, 9*j+i], alpha=0.5, color='r')
                
                if i!= int(data.shape[1]/n_tickers)-1:
                    plt.xticks([])
                    
                plt.yticks(fontsize=30)
                plt.xticks(fontsize=30)
                
                if i==0:
                    plt.title(tickers[j], fontsize=40)
                    
                if j==0:
                    if i!=m-1:
                        plt.ylabel(r"$b_{%s}$"%(i+1), fontsize=40, labelpad=20)
                        ax.yaxis.set_label_coords(-.15, .5)
                    else:
                        plt.ylabel(r"$S$", fontsize=40, labelpad=20)
                        ax.yaxis.set_label_coords(-.15, .5)

        fig.supxlabel('Day of training data', y=0.001, fontsize=40)
        plt.tight_layout()
        if savefig == True:
            plt.savefig("confidence_bands.pdf", bbox_inches="tight")
        plt.show()   
        
    # plot the evolution of the losses as training progresses
    def Plot_Loss(self, savefig=False):
        
        L_mse = np.concatenate((np.array(self.loss_mse), np.array(self.loss_mse_comb)))
        t_mse = np.cumsum(np.ones(len(L_mse)))
        
        L_ll = np.array(self.loss_ll)
        t_ll = np.cumsum(np.ones(len(L_ll)))
        
        L_density = np.array(self.loss_density)
        t_W = np.cumsum(np.ones(len(L_density)))
        
        L_ll_W = np.array(self.loss_ll_penalized)
        t_ll_W = np.cumsum(np.ones(len(L_ll_W)))
        
        fig,ax = plt.subplots(2,2, figsize=(20,12))
        plt.tick_params(axis='both', labelsize=22)
        
        plt.subplot(2, 2, 1)
        plt.plot(t_mse, L_mse, alpha=0.5)
        if len(self.loss_mse) > 0:
            plt.plot(MovingAverage(L_mse, 500), color='k', label='MA(500)') 
        ax[0,0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: format(x/1000,'1.0f')+'k'))
        ax[0,0].set_xlabel('Iterations', fontsize=20)
        ax[0,0].set_ylabel('MSE', fontsize=20)
        plt.yscale('log')
        ax[0,0].tick_params(axis='both', which='both', labelsize=20)
        
        plt.subplot(2, 2, 2)
        plt.plot(t_ll, L_ll, alpha=0.5)
        if len(self.loss_ll) > 0:
            plt.plot(MovingAverage(self.loss_ll, 500), color='k', label='MA(500)')
        plt.yscale('symlog', linthresh=1)
        ax[0,1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: format(x/1000,'1.0f')+'k'))
        ax[0,1].set_xlabel('Iterations', fontsize=20)
        ax[0,1].set_ylabel('NLL', fontsize=20)
        ax[0,1].tick_params(axis='both', labelsize=20)
        ax[0,1].yaxis.set_ticks([-1e2, -1e0, 1e0, 1e2, 1e4])
        
        plt.subplot(2, 2, 3)
        plt.plot(t_ll_W, L_ll_W, alpha=0.5)
        if len(self.loss_ll_penalized) > 0:
            plt.plot(MovingAverage(self.loss_ll_penalized, 500), color='k', label='MA(500)')
        ax[1,0].set_yscale('symlog', linthresh=1)
        ax[1,0].set_xlabel('Iterations', fontsize=20)
        ax[1,0].set_ylabel('NLL+'+r'$\alpha$'+'PIT', fontsize=20)
        ax[1,0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: format(x/1000,'1.0f')+'k'))
        ax[1,0].tick_params(axis='both', labelsize=20)
        ax[1,0].yaxis.set_ticks([-1e2, -1e0, 1e0, 1e2, 1e4])
        
        plt.subplot(2, 2, 4)
        plt.plot(t_W, L_density, alpha=0.5)
        if len(self.loss_ll_penalized) > 0:
            plt.plot(MovingAverage(self.loss_density, 500), color='k', label='MA(500)')
        plt.yscale('log')
        ax[1,1].set_xlabel('Iterations', fontsize=20)
        ax[1,1].set_ylabel('PIT', fontsize=20)
        ax[1,1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: format(x/1000,'1.0f')+'k'))
        ax[1,1].tick_params(axis='both', labelsize=20)
        
        if savefig == True:
            plt.savefig("training_loss.pdf", bbox_inches="tight")
        plt.show()
        
    # plot the histogram of the PITs
    def Plot_PIT(self, tickers=['AMZN', 'IBM', 'INTC', 'TSLA'], savefig=False):
        
        u = self.PIT().detach().cpu().numpy()
        
        if self.ticker is not None:
            n_tickers = len(self.ticker)
            tickers = self.ticker
        else:
            n_tickers = len(tickers)
        
        fig = plt.figure(figsize=(30,30))
        m = int(np.ceil(u.shape[1]) / n_tickers)
        
        for i in range(n_tickers):
            for j in range(m):
                plt.subplot(m,4,4*j+1+i)
                plt.hist(u[:,4*i+j], bins=np.linspace(0,1,26), density=True)
                plt.ylim(0,2)
                plt.axhline(y=1, color='r', linestyle='-')
                plt.locator_params(axis='y', nbins=3)
                
                if j!= m-1:
                    plt.xticks([])
                    
                plt.yticks(fontsize=30)
                plt.xticks(fontsize=30)
                
                if j==0:
                    plt.title(tickers[i], fontsize=45)
                    
                if i!=0:
                    plt.yticks([])
                    
                if i==0:
                    if j!=m-1:
                        plt.ylabel(r"$b_{%s}$"%(j+1), fontsize=40, labelpad=10)
                    else:
                        plt.ylabel(r"$S$", fontsize=40, labelpad=10)
                        
        fig.supxlabel('Probability Integral Transform U', y=0.001, fontsize=40)
        fig.supylabel('Density', x=0.001, fontsize=40)
        plt.tight_layout()
        
        if savefig == True:
            plt.savefig("PIT_histogram.pdf", bbox_inches="tight")
        plt.show()
        