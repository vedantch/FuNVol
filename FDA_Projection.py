#%%
import numpy as np
import matplotlib.pyplot as  plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA
from scipy.special import eval_legendre as Legendre
from scipy.linalg import fractional_matrix_power

#%%
class FDA():
    
    def __init__(self, Order = 4, data=[], PlotBasis=False):
        
        self.phi = []       # stores lambda functions for set of empirical/Legendre basis
        self.order = []     # stores the Legendre polynomial powers of the basis

        # x corresponds to option delta, y corresponds to option time to expiry (tau)
        # modify the ranges of x and y to reflect the range in which transformed values lie
        # plot_x and plot_y are used to get the grid values for plotting surfaces
        self.plot_x, self.plot_y = np.linspace(-1,1,25), np.linspace(-1,1,25)           
        # weight_x and weight_y are used for integration over [-1,1]x[-1,1] during the computation of basis weight matrix W
        self.weight_x, self.weight_y = np.linspace(-1,1,101), np.linspace(-1,1,101)
            
        # add in the Legendre basis
        for i in range(1,Order+1):
            for j in range(i+1):
                
                temp = lambda x, y, j=j, i=i : \
                    Legendre(j,x)/np.sqrt(2/(2*j+1)) * Legendre(i-j,y)/np.sqrt(2/(2*(i-j)+1)) 
                    
                self.phi.append(temp)            
                self.order.append([j,(i-j)])        
                
        # create a list of all basis evaluated on mesh or vector of pairs of (x,y)
        self.Phi = lambda x,y : np.array([ self.phi[i](x,y)  for i in range(len(self.phi)) ])
        
        if PlotBasis:
           self.Plot_Basis()
        
    # plot basis functions
    def Plot_Basis(self):
        
        m = int(np.ceil(np.sqrt(len(self.phi))))
        X,Y = np.meshgrid(self.plot_x, self.plot_y)
        
        fig = plt.figure()
        for i in range(len(self.phi)):
            
            ax = fig.add_subplot(m, m, i+1, projection='3d')
            if self.order[i][0] < 0:
                Z = self.phi[i](X.flatten(),Y.flatten()).reshape(X.shape)
            else:
                Z = self.phi[i](X,Y) 
            
            ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
            ax.set_title(r"$L_{" + str(self.order[i][0]) + "}(x)\;L_{" + str(self.order[i][1]) + "}(y)$")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            ax.view_init(elev=20, azim=230)
            
        plt.tight_layout(pad=1.0)
        plt.show()        
        
    # project data onto basis functions   
    def Fit_IV(self, X, Y, Z, PlotFit=False):
        
        # flatten the data
        Z_flat = Z.reshape(-1)
        X_flat = X.reshape(-1)
        Y_flat = Y.reshape(-1)
        
        # evalute basis on flatten (x,y) pairs 
        basis_value = self.Phi(X_flat, Y_flat).T
        
        # regress to obtain basis coefficients
        nan_posn = np.argwhere(np.isnan(Z_flat))    
        Z_flat = np.delete(Z_flat, nan_posn[:,0]).reshape(-1,1)
        basis_value = np.delete(basis_value, nan_posn[:,0], axis=0)   
        reg = LinearRegression().fit(basis_value, Z_flat)
        a0 = reg.intercept_[0] 
        a = reg.coef_[0]

        # plot the fit
        if PlotFit:
            self.Plot_Fit(X,Y,Z,a0,a)
        
        return a, a0

    # project data onto FPCs
    def Fit_IV_FPC(self, X, Y, Z, K, PlotFit=False):
        
        # flatten the data
        Z_flat = Z.reshape(-1)
        X_flat = X.reshape(-1)
        Y_flat = Y.reshape(-1)
        
        # evalute basis on flatten (x,y) pairs, also need to use the mean curve as part of the fit
        basis_value = np.zeros((len(X_flat), K)) 
        for i in range(K):
            basis_value[:,i] = self.psi[i](X_flat, Y_flat)
        
        nan_posn = np.argwhere(np.isnan(Z_flat))    
        Z_flat = np.delete(Z_flat, nan_posn[:,0]).reshape(-1,1)
        basis_value = np.delete(basis_value, nan_posn[:,0], axis=0)   

        # regress to obtain basis coefficients
        mean = np.delete(self.mean(X_flat, Y_flat).reshape(-1,1), nan_posn[:,0]).reshape(-1,1)
        Z_flat_zero_mean = Z_flat - mean
        reg = LinearRegression(fit_intercept=False).fit(basis_value, Z_flat_zero_mean)
        b = reg.coef_[0]

        # plot the fit
        if PlotFit:
            self.Plot_FPC_Fit(X,Y,Z, b)
        
        return b

    # return implied vol (transformed) values on a grid when projected on basis functions
    def Generate_Surface(self, a0, a):

        X_fit, Y_fit = np.meshgrid(self.plot_x, self.plot_y)

        Z_fit = np.zeros((X_fit.shape))
        Z_fit += a0
        for i in range(len(self.phi)):
            Z_fit += a[i] * self.phi[i](X_fit,Y_fit)
            
        return X_fit, Y_fit, Z_fit
            
    # return implied vol (transformed) values on a grid when projected on FPCs
    def Generate_FPC_Surface(self, b):

        X_fit, Y_fit = np.meshgrid(self.plot_x, self.plot_y)
        
        Z_fit = self.mean(X_fit, Y_fit)
        for i in range(len(b)):
            Z_fit += b[i] * self.psi[i](X_fit,Y_fit)            
        
        return X_fit, Y_fit, Z_fit 
    
    # return implied vol (transformed) values on a provided grid
    def Generate_IV_Grid(self, b, delta, tau):
        
        X_fit, Y_fit = np.meshgrid(delta,tau)
        
        Z_fit = np.zeros((b.shape[0], b.shape[1], X_fit.shape[0], X_fit.shape[1]))
        
        Z_fit[:,:] = self.mean(X_fit, Y_fit)
        for i in range(b.shape[-1]):
            Z_fit += np.matmul(b[:,:,i].reshape(-1,1), self.psi[i](X_fit,Y_fit).reshape(1,-1)).reshape(b.shape[0], b.shape[1], X_fit.shape[0], X_fit.shape[1])
        
        return X_fit, Y_fit, Z_fit 

    # plot IV surface obtained in terms of the basis functions
    def Plot_Fit(self, X, Y, Z, a0, a):

        X_fit, Y_fit = np.meshgrid(self.plot_x, self.plot_y)

        Z_fit = np.zeros((X_fit.shape))
        Z_fit += a0
        for i in range(len(self.phi)):
            Z_fit += a[i] * self.phi[i](X_fit,Y_fit)

        fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
        ax.plot_surface(X_fit, Y_fit, Z_fit,linewidth=0, alpha=0.7)
        
        ax.scatter(X, Y, Z, color='r')
    
        plt.show()
               
    # plot IV surface obtained in terms of the FPCs
    def Plot_FPC_Fit(self, X, Y, Z, b):

        X_fit, Y_fit = np.meshgrid(self.plot_x, self.plot_y)
        
        Z_fit = self.mean(X_fit, Y_fit)
        for i in range(len(b)):
            Z_fit += b[i] * self.psi[i](X_fit,Y_fit)

        fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
        ax.plot_surface(X_fit, Y_fit, Z_fit,linewidth=0, alpha=0.7)
        
        ax.scatter(X, Y, Z-self.mean(X, Y), color='r')
    
        plt.show()     
        
    # compute basis weight matrix needed 
    def Get_Weight_Matrix(self):
        
        a, b = self.weight_x, self.weight_y
        x,y = np.meshgrid(a, b)
        
        W = np.zeros((len(self.phi)+1,len(self.phi)+1))
        
        W[0,0] = np.sum(x.shape[0]*x.shape[1]*(a[1]-a[0])*(b[1]-b[0]))
        
        for i in range(len(self.phi)):
            
            fi = self.phi[i](x,y)
            
            W[i+1,0] = np.sum(fi*(a[1]-a[0])*(b[1]-b[0]))
            W[0,i+1] = W[i+1,0]
            
            for j in range(i+1,len(self.phi)):
                
                fj = self.phi[j](x,y)
            
                W[i+1,j+1] = np.sum(fi*fj*(a[1]-a[0])*(b[1]-b[0]))
                W[j+1,i+1] = W[i+1,j+1]
                
            W[i+1,i+1] = np.sum(fi*fi*(a[1]-a[0])*(b[1]-b[0]))
             
        return W
        
    # compute FPCCs when data projected on FPCs
    def Compute_FPC_coeff(self, a, PlotFPC=True, savefig=False):
        
        self.mean = lambda x,y,a=a :  np.mean(a[:,0]) + np.sum([ np.mean(a[:,j+1]) *self.phi[j](x,y) for j in range(len(self.phi))], axis=0)        
        
        W = self.Get_Weight_Matrix()
        
        c = a - np.mean(a,axis=0)
        
        A = np.matmul(c.T,c)

        # below can be used for ortogonal basis only
        # self.kappa, self.b = LA.eig(1/a.shape[0] * A)

        sqrt_W = fractional_matrix_power(W, 0.5)
        B = np.matmul( np.matmul(sqrt_W, A), sqrt_W) / a.shape[0]
        
        self.kappa, u = LA.eig(B)
        
        self.b = np.matmul(LA.inv(sqrt_W), u)
        
        self.psi = []
        
        for i in range(len(self.kappa)):
            
            self.psi.append(lambda x,y,i=i : self.b[0,i] + np.sum([ self.b[j+1,i] *self.phi[j](x,y) for j in range(len(self.phi))], axis=0) )
            
        if PlotFPC:
            
            N = np.sum( np.cumsum(self.kappa)/np.sum(self.kappa) < 0.995)
            x, y = np.meshgrid(self.plot_x, self.plot_y)

            m = int(np.ceil(N/4))            
            fig = plt.figure(figsize=(20,12))
            
            for i in range(N):
                
                z = self.psi[i](x,y)
                
                ax = fig.add_subplot(m, 4, i+1, projection='3d')
                
                ax.plot_surface(x,y,z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
                ax.set_title(r"$\psi_"+str(i+1)+"$" + "({:2.1f}".format(100*self.kappa[i]/np.sum(self.kappa))+"%)", fontsize=30)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                
                if i!=N-1:
                    ax.yaxis.set_major_formatter(plt.NullFormatter())            
                    ax.xaxis.set_major_formatter(plt.NullFormatter())    
                    
                if i==N-1:
                    ax.set_zlabel(r"$\widetilde{\sigma}$", fontsize=30, labelpad=15)
                    ax.set_xlabel(r"$\widetilde{\delta}$", fontsize=30, labelpad=25)
                    ax.tick_params(axis='x', which='major', labelsize=20)
                    ax.set_ylabel(r"$\widetilde{\tau}$", fontsize=30, labelpad=25)
                    ax.tick_params(axis='y', which='major', labelsize=20)
   
                ax.tick_params(axis='z', which='major', labelsize=20)
                ax.grid(True)

            plt.tight_layout(w_pad=4)
            if savefig:    
                plt.savefig("fpcs.pdf", bbox_inches="tight")
            plt.show()
