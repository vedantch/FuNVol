#%%
import numpy as np
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from matplotlib import cm
import seaborn as sns
import random
import datetime
import math
from sklearn.linear_model import LinearRegression
import torch
import imageio

from FDA_Projection import FDA
from NeuralSDE import NeuralSDE

#%%
class FuNVol:
    
    def __init__(self, data_dir, tickers, K=3):
        
        # load all data
        self.raw_data = []
        self.raw_price_data = []
        self.data = []
        
        self.tickers = np.array(tickers)
        
        latest_start_date = 0
        
        print("loading data for various tickers...")
        # load the data and normalize individually        
        for ticker in tickers:
            
            # the raw data files cannot be provided due to licensing issues
            with open(data_dir + ticker + '.pkl', 'rb') as handle:
                self.raw_data.append(pickle.load(handle))

            with open(data_dir + ticker + '_price.pkl', 'rb') as handle:
                self.raw_price_data.append(pickle.load(handle))

            # use only overlaping price and option data
            start = np.max((self.raw_price_data[-1]['dates'][0], self.raw_data[-1]['dates'][0]))
            
            mask = (self.raw_price_data[-1]['dates']>=start)
            self.raw_price_data[-1]['dates'] = self.raw_price_data[-1]['dates'][mask]
            self.raw_price_data[-1]['prices'] = self.raw_price_data[-1]['prices'][mask]
            
            mask = (self.raw_price_data[-1]['dates']>=start)
            self.raw_data[-1]['dates'] = self.raw_data[-1]['dates'][mask]
            self.raw_data[-1]['IV'] = self.raw_data[-1]['IV'][mask]
            d1, d2, d3 = self.raw_data[-1]['IV'].shape
            self.raw_data[-1]['IV'] = self.raw_data[-1]['IV'].reshape(d1, d3, d2)
            
            self.raw_data[-1]['prices'] = copy.deepcopy(self.raw_price_data[-1]['prices'])
            
            latest_start_date = np.max((self.raw_data[-1]['dates'][0], latest_start_date))
        
        print("normalizing IVs...")
        # normalizing coefficients
        self.nrm = []
        self.reg = []
        
        for data in copy.deepcopy(self.raw_data):
            
            a_, b_, IV = self.Normalise(np.log(np.exp(data['IV'])-1))
            data['IV'] = IV
            
            c_, d_, prices = self.Normalise(np.log(data['prices']))
            data['prices'] = prices

            self.nrm.append({'IV':[a_,b_], 'price' : [c_,d_]})
            self.data.append(data)
            
        print("truncating data to common set of dates...")
        # only keep data on dates where all tickers have data and
        # collate the data into a single source
        self.data_all = {'dates' : [],
                         'tau' : self.data[0]['tau'],
                         'Delta' : self.data[0]['Delta'],
                         'IV' : [],
                         'prices' : [],
                         'ticker_idx' : []}

        self.delta = self.data[0]['Delta']
        self.tau = self.data[0]['tau']
        
        for i, ticker in enumerate(tickers):
            
            self.data[i]['IV'] = self.data[i]['IV'][self.data[i]['dates']>=latest_start_date,:,:]
            self.data[i]['prices'] = self.data[i]['prices'][self.data[i]['dates']>=latest_start_date]
            self.data[i]['dates'] = self.data[i]['dates'][self.data[i]['dates']>=latest_start_date]
            alpha_, beta_, prices_d = self.Detrend(self.data[i]['prices'])
            self.reg.append([alpha_,beta_])
            self.data[i]['prices'] = prices_d
            
            if i ==0 :
                self.data_all['IV'] = self.data[i]['IV']
                self.data_all['dates'] = self.data[i]['dates']
                self.data_all['ticker_idx'] = np.zeros(len(self.data[i]['dates']), int)
                self.data_all['prices'] = self.data[i]['prices']
                
            else:
                self.data_all['IV'] = np.concatenate((self.data_all['IV'], self.data[i]['IV']))
                self.data_all['dates'] = np.concatenate((self.data_all['dates'], self.data[i]['dates']))
                self.data_all['prices'] = np.concatenate((self.data_all['prices'], self.data[i]['prices']))
                self.data_all['ticker_idx'] = np.concatenate((self.data_all['ticker_idx'], i + np.zeros(len(self.data[i]['dates']), int)))
            
        self.rng_IV = np.nanquantile(self.data_all["IV"].flatten(), [0.025, 0.975])
        
        print("projecting onto functional basis...")
        self.fda_model = FDA()
        self.x = 2 * self.data_all["Delta"] - 1
        self.y = 2*np.sqrt(self.data_all["tau"])/np.max(np.sqrt(self.data_all["tau"])) - 1
        self.X, self.Y = np.meshgrid(self.x, self.y)     

        self.a = []
        self.a_all = np.zeros((0,len(self.fda_model.phi)+1))
        for data in self.data:
            self.a.append(self.Perform_Projection(data['IV']))
            self.a_all = np.concatenate((self.a_all, self.a[-1]))
        
        print("plot some sample fits...")
        idx = np.random.randint(0, self.a_all.shape[0], 5)
        self.Plot_Sample_Fits(idx)
        
        print("plotting time-series of coefficients")
        self.Plot_Time_Series_Coeff(self.a)
        
        print("plotting histogram of coefficients")
        self.Plot_Histogram_Coeff(self.a)
        
        print("plot pairwise scatter")
        self.Plot_Pairwise_Scatter_Coeff(self.a, "a")     
        
        print("Compute the common FPCs")
        mask = np.sum(np.isnan(self.a_all),axis=1)>0
        self.fda_model.Compute_FPC_coeff(self.a_all[~mask], PlotFPC=True)
        
        print("Project onto common FPCs")

        self.K = K        
        self.b = []
        self.b_all = np.zeros((0, K))
        self.b_sim = []
        self.b_price = []
        for data in self.data:
            self.b.append(self.Perform_FPC_Projection(data['IV'], K))
            self.b_all = np.concatenate((self.b_all, self.b[-1]))
            self.b_price.append(np.concatenate((self.Perform_FPC_Projection(data['IV'], K),
                                data['prices'].values.reshape(-1,1)), axis=1))
        
        print("plot some sample fits...")
        self.Plot_Sample_Fits_FPC(idx)        
        
        print("plotting time-series of coefficients")
        self.Plot_Time_Series_Coeff(self.b, "b")
        
        print("plotting histogram of coefficients")
        self.Plot_Histogram_Coeff(self.b, "b")
        
        print("plot pairwise scatter")
        self.Plot_Pairwise_Scatter_Coeff(self.b, "b")
        
    def Learn_Neural_SDE(self, ticker=None, train_percent=0.9):

        self.current_ticker = ticker
        
        N = (self.b[0].shape[1]+1)   
        M = math.floor(self.b[0].shape[0]*train_percent)
        self.train_size = M
        
        # concatenate coefficients of FDA with prices 
        if ticker is not None:
            
            ticker_idx = np.where(self.tickers == ticker)[0][0]

            data = np.concatenate((self.b[ticker_idx][:M,:N-1].reshape(-1,N-1), 
                                    self.data[ticker_idx]['prices'].values[:M].reshape(-1,1)), 
                                  axis=1)
            dates = pd.to_datetime(self.data[ticker_idx]['dates'][:M], format='%Y%m%d')
            T = (dates-dates[0])/ np.timedelta64(1, 'D')
            T = T.values.reshape(-1,1)/365
            
        else:
            # concatenate coefficients of FDA with prices across all tickers
            for i in range(len(self.tickers)):
                if i == 0:
                    data = np.concatenate((self.b[i][:M,:N-1].reshape(-1,N-1), self.data[i]['prices'].values[:M].reshape(-1,1)), axis=1)
                    dates = pd.to_datetime(self.data[i]['dates'][:M], format='%Y%m%d')
                    T = (dates-dates[0])/ np.timedelta64(1, 'D')
                    T = T.values.reshape(-1,1)/365
                else:
                    data = np.concatenate((data, self.b[i][:M,:N-1].reshape(-1,N-1), self.data[i]['prices'].values[:M].reshape(-1,1)), axis=1)

        N = data.shape[1]
        
        params= {'normalize' : np.ones(N, bool),
                  'n_features' : N,
                  'n_GRU_layers' : 3,
                  'GRU_hidden_size' :N,
                  'nOut' : N}
        
        self.neural_sde = NeuralSDE(data, T, self.current_ticker, params)
      
    # generate IV, price and FPCCs for future scenarios
    def Generate_Data(self, nsims=1, nsteps=31):

        b_sim = self.neural_sde.Simulate_Synthetic_Time_Series(nsims=nsims, nsteps=nsteps)

        self.b_sim = b_sim 
        
        def Data_per_Ticker(b_sim, ticker_idx):
            
            price = self.Trend(*self.reg[ticker_idx], np.arange(start=self.train_size+1, 
                            stop=self.train_size+1+b_sim.shape[0], step=1).repeat(repeats=b_sim.shape[1]), b_sim[:,:,-1].reshape(-1)) 
            price = np.exp(self.UnNormalise(*self.nrm[ticker_idx]['price'], price)).reshape(b_sim.shape[0], b_sim.shape[1])
            b = b_sim[:,:,:-1]
            delta_grid_t, tau_grid_t, IV_t = self.fda_model.Generate_IV_Grid(b=b, delta = self.x, tau = self.y)
            IV = np.log(1+ np.exp(self.UnNormalise(*self.nrm[ticker_idx]['IV'], IV_t)))
            IV = IV.transpose(1,0,2,3)
            
            return delta_grid_t, tau_grid_t, IV, price
        
        if self.current_ticker is not None:
            
            ticker_idx = np.where(self.tickers == self.current_ticker)[0][0]
            price = self.Trend(*self.reg[ticker_idx], np.arange(start=self.train_size+1, stop=self.train_size+1+b_sim.shape[0],
                                                        step=1).repeat(repeats=b_sim.shape[1]), b_sim[:,:, -1].reshape(-1)) 
            price = np.exp(self.UnNormalise(*self.nrm[ticker_idx]['price'], price)).reshape(b_sim.shape[0], b_sim.shape[1])
            
            
            b = b_sim[:,:,:-1]
            delta_grid_t, tau_grid_t, IV_t = self.fda_model.Generate_IV_Grid(b=b, delta = self.x, tau = self.y)
            IV = np.log(1+ np.exp(self.UnNormalise(*self.nrm[ticker_idx]['IV'], IV_t)))
            IV = IV.transpose(1,0,2,3)

        else:  
            
            seq_length = int(b_sim.shape[-1]/len(self.tickers))
            delta_grid_t = []
            tau_grid_t = []
            IV = []
            price = []
            for i in range(len(self.tickers)):
                delta_grid_t_i, tau_grid_t_i, IV_i, price_i = Data_per_Ticker(b_sim[:,:,i*seq_length:(i+1)*seq_length], i) 
                delta_grid_t.append(delta_grid_t_i)
                tau_grid_t.append(tau_grid_t_i)
                IV.append(IV_i)
                price.append(price_i.transpose(1,0))
            
        return delta_grid_t, tau_grid_t, IV, price, b_sim.transpose(1,0,2)
        
    # perform functional projection on basis functions for each day and store results         
    def Perform_Projection(self, IV):
        
        a = np.zeros((IV.shape[0], 1 + len(self.fda_model.phi)))
    
        for i, _IV in enumerate(IV):
    
            if np.sum(~np.isnan(_IV )) > 0:
                a_, a0_ = self.fda_model.Fit_IV(self.X, self.Y, _IV , PlotFit=False)
            else:
                a[i, :] = a[i-1,:]
                continue
    
            a[i, 0] = a0_
            a[i, 1:] = a_
    
            if np.mod(i + 1, 10) == 0:
                print('.', end="")
            if np.mod(i + 1, 200) == 0:
                print(' ' + str(i + 1) + ' of ' + str(IV.shape[0]))
    
        return a
        
    # perform functional projection on FPCs for each day and store results     
    def Perform_FPC_Projection(self, IV, K=5):

        b = np.zeros((IV.shape[0], K))

        for i in range(IV.shape[0]):

            if np.sum(~np.isnan(IV[i])) > 0:
                b[i, :] = self.fda_model.Fit_IV_FPC(self.X, self.Y, IV[i], K, PlotFit=False)                
            else:
                b[i, :] = b[i-1,:]
                
                continue

            if np.mod(i + 1, 10) == 0:
                print('.', end="")
            if np.mod(i + 1, 200) == 0:
                print(' ' + str(i + 1) + ' of ' + str(IV.shape[0]))
                
        return b

    # normalize input values
    def Normalise(self, X):
        
        qtl = np.quantile(X[~np.isnan(X[:])], [0.1, 0.9])
        
        b = 2 / (qtl[1] - qtl[0])
        a = -(qtl[1] + qtl[0])/(qtl[1] - qtl[0])
        
        return a, b, a + b * X
    
    # recover values in original scale from transformed values
    def UnNormalise(self, a, b, X):
        
        return  (X -a)/b
    
    # remove time trend from price time series
    def Detrend(self, X):
        
        t = np.arange(start=1, stop=X.shape[0]+1, step=1).reshape(-1,1)
        
        reg_model = LinearRegression()
        reg_model.fit(t, X)
        
        a0_, a_ = reg_model.intercept_, reg_model.coef_
        
        price = X - reg_model.predict(t)
        
        return a0_, a_, price
    
    # reintroduce time trend in synthetic price data
    def Trend(self, a0_, a_, t, X):
        
        price = a0_ + a_*t + X
        
        return price
    
    # plots pairwise scatter plot of FPCCs
    def Plot_Pairwise_Scatter_Coeff(self, a, title, savefig=False):
        
        b = [a[3], a[2], a[1], a[0]]
        a = b
        tickers = ['TSLA', 'INTC', 'IBM', 'AMZN']
        for i in range(len(a)):
            a_df = pd.DataFrame(np.vstack(a))
        
        a_df.columns = [r'$%s_{%s}$'%(title, i+1) for i in range(a[0].shape[1])]
        a_df['Asset'] = pd.Series(tickers).repeat(a[0].shape[0]).values
        
        scatter = sns.pairplot(a_df, hue='Asset', markers=["o", "^", "*", "X"], aspect=1, corner='True',plot_kws=dict(s=6, alpha=0.5))
        sns.move_legend(scatter, "upper right", bbox_to_anchor=(0.75,0.75), fontsize='30', title_fontsize='40', markerscale=3)

        for ax in scatter.axes[:,0]:
            
            ax.tick_params(axis='both', labelsize=20)
            ax.get_yaxis().set_label_coords(-0.4,0.5)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            label = ax.get_ylabel()
            ax.set_ylabel(label, size=30)
        
        for ax in scatter.axes[-1,:]:
            ax.tick_params(axis='both', labelsize=20)
            ax.get_xaxis().set_label_coords(0.5,-0.2)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            label = ax.get_xlabel()
            ax.set_xlabel(label, size=30)
            
        if savefig:    
            plt.savefig("vector plots/pairwise scatter_%s.pdf" %title, dpi=300, bbox_inches="tight")

    # plots histogram of FPCCs
    def Plot_Histogram_Coeff(self, a, varname="a", ncol=4):

        m = int(np.ceil(a[0].shape[1] / ncol))

        plt.figure(figsize=(10, 10))

        for a in a:
            for i in range(a.shape[1]):
                
                plt.subplot(m, ncol, i + 1)
                plt.hist(a[:, i], bins=25, alpha=0.5)
    
                plt.title('$' + varname + '_{' + str(i) + '}$', fontsize=18)
                plt.xticks(fontsize=14)
                plt.locator_params(nbins=4)
                plt.yticks(fontsize=14)
                plt.locator_params(nbins=4)

        plt.tight_layout(pad=1.5)
        plt.show()

    # plots time series of FPCCs
    def Plot_Time_Series_Coeff(self, a, varname="a", ncol=4):
        
        m = int(np.ceil(a[0].shape[1] / ncol))
        plt.figure(figsize=(10, 10))

        for a_ in a:
            
            for i in range(a_.shape[1]):
                plt.subplot(m, ncol, i + 1)
                plt.plot(a_[:, i])
                plt.xlabel('$t$', fontsize=18)
                plt.ylabel('$' + varname + '_{' + str(i) + ',t}$', fontsize=18)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlim(100,150)

        plt.tight_layout()
        plt.show()        
    
    # plot samples of fitted surfaces using basis functions
    def Plot_Sample_Fits(self, idx, save_fig=False):
    
        count = 1
        for m in range(len(idx)):
            X_fit, Y_fit, Z_fit = self.fda_model.Generate_Surface(self.a_all[idx[m], 0], self.a_all[idx[m], 1:])
    
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(X_fit, Y_fit, Z_fit, linewidth=0, antialiased=True, alpha=0.7)
            ax.scatter(self.X.flatten(), self.Y.flatten(), self.data_all["IV"][idx[m]].flatten(), color='r')
            # ax.set_zlim(self.rng_IV)
            ax.set_xlabel(r"$\widetilde{\Delta}$")
            ax.set_ylabel(r"$\widetilde{\tau}$")
            ax.set_zlabel(r"$\widetilde{\sigma}$")
            ax.set_title(idx[m])
            
            if save_fig:
                fig.savefig("Sample_" + str(count) + ".png")
                
            count += 1
            
            print(count)
    
    # plot samples of fitted surfaces using FPCs       
    def Plot_Sample_Fits_FPC(self, idx):

        for m in range(len(idx)):
            X_fit, Y_fit, Z_fit = self.fda_model.Generate_Surface(self.a_all[idx[m], 0], self.a_all[idx[m], 1:])
            X_fit_FPC, Y_fit_FPC, Z_fit_FPC = self.fda_model.Generate_FPC_Surface(self.b_all[idx[m], :])

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(X_fit, Y_fit, Z_fit, linewidth=0, antialiased=True, alpha=0.7)
            ax.plot_surface(X_fit_FPC, Y_fit_FPC, Z_fit_FPC, linewidth=0, antialiased=True, alpha=0.7)
            ax.scatter(self.X.flatten(), self.Y.flatten(), self.data_all["IV"][idx[m]].flatten(), color='r')
            ax.set_xlabel(r"$\widetilde{\Delta}$")
            ax.set_ylabel(r"$\widetilde{\tau}$")
            ax.set_zlabel(r"$\widetilde{\sigma}$")
            ax.set_title(idx[m])

            plt.show()