import torch

from sbi import utils as sbi_utils
from sbi.inference import posteriors
from sbi.utils import BoxUniform
from sbi.utils import process_prior
from collections.abc import Iterable

from scipy.stats import truncnorm

import torch.nn as nn
import torch.nn.functional as F
import artpop
import astropy.units as u
import numpy as np

default_sersic_dict = {'n':0.5, 'r_eff_as':10, 'theta': 0,'ellip':0,'dx':0,'dy':0}

def block_mean(x, num_block):
    r1 = x.shape[1]%num_block
    r2 = x.shape[2]%num_block
    x = x[:,:-r1,:-r2]
    x = x.reshape(x.shape[0],int(x.shape[1]/num_block), num_block,int(x.shape[2]/num_block), num_block)
    return x.mean(axis = (2,4) )

#Basic function to reutrn common imagres
def get_DECam_imager():
    return artpop.image.ArtImager('DECam', diameter = 4.0*u.m, read_noise = 7)

def get_HSC_imager():
    return artpop.image.ArtImager('HSC', diameter = 8.4*u.m, read_noise = 4.5)

def get_injec_cutouts(num, size,files = None, array = None ,output = 'numpy', pad = 0):
    #function to create cutouts to inject real images into

    if files is not None:
        obs_ims = []
        for f in files:
            obs_ims.append(fits.getdata(f) )
        obs_ims = np.asarray(obs_ims)
    elif array is not None:
        obs_ims = np.asarray(array)
    else:
        print("Must specify files or array")

    x_max = obs_ims.shape[1]
    y_max = obs_ims.shape[2]

    #add padding to not deal with edges
    x = np.arange(0,size[0])
    y = np.arange(0,size[1])
    X,Y = np.meshgrid(x,y)

    inds_X = X + np.random.randint(low = pad,high = x_max - size[0]-pad, size = num)[:,None,None]
    inds_Y = Y + np.random.randint(low = pad,high = y_max - size[1]-pad, size = num)[:,None,None]

    #Extract cutouts
    cutouts = obs_ims[:,inds_X,inds_Y]

    cutouts = np.moveaxis(cutouts,0,1)

    if output == 'torch': cutouts = torch.from_numpy(cutouts.astype(np.float32)).type(torch.float)
    return cutouts

def load_post(prior, enet, state_dict, im_shape, flow = 'maf', net_kwargs = {},device ='cpu'):

    #Need example data
    t_start = prior.sample((2,)).to(device)
    x_start = torch.ones((2,*im_shape)).to(device)
    
    #initialize model
    nde = sbi_utils.posterior_nn(model=flow, embedding_net=enet,**net_kwargs)
    net = nde(t_start,x_start)
    
    for key, value in state_dict.items():
        state_dict[key] = state_dict[key].to(device)
    
    #Load trained parameters
    net.load_state_dict(state_dict)
    
    #Return sbi object
    return posteriors.direct_posterior.DirectPosterior(net, prior, x_shape = (1,*im_shape),device =device )

def get_reddening(coords,filts):
    from dustmaps.sfd import SFDQuery
    import extinction
    
    ebv = SFDQuery()(coords)
    
    tab = artpop.filters.get_filter_properties()
    lam_eff = np.hstack([tab[tab['bandpass'] == filt]['lam_eff'].value for filt in filts] )
    flux = np.ones(len(filts))
    
    return extinction.apply(extinction.calzetti00(lam_eff, 3.1*ebv, 3.1), flux)

class MSSP_Prior:
    def __init__(self,Dlims,Mlims,logAgelims,expand_fac = 1.,device = 'cpu'):
        
        self.Dlims = torch.tensor(Dlims)
        self.D_dist = BoxUniform(low = [Dlims[0],],high = [Dlims[1],])
        self.Mlims = torch.tensor(Mlims)
        self.logAgelims = torch.tensor(logAgelims)
        self.M_dist_list = []
        self.A_dist_list = []

        if self.Mlims.ndim > 1:
            self.n_pop = self.Mlims.shape[0]
            assert self.Mlims.shape[0] == self.logAgelims.shape[0]
            for i in range(self.n_pop):
                self.M_dist_list.append( BoxUniform([self.Mlims[i,0],],[self.Mlims[i,1],]) )
                self.A_dist_list.append( BoxUniform([self.logAgelims[i,0],],[self.logAgelims[i,1],]) )
        else:
            self.n_pop = 1
            assert self.logAgelims.ndim == 1
            self.M_dist_list.append( BoxUniform([self.Mlims[0],],[self.Mlims[1],]) )
            self.A_dist_list.append( BoxUniform([self.logAgelims[0],],[self.logAgelims[1],]) )
        
        self.MZR_sig = 0.17*expand_fac
        self.Z_min = -2.25
        self.Z_max = 0.25
        self.device = device

    def MZR(self, logM):
        return -1.69 + 0.3*(logM - 6.)
    
    def sample_Z_SSP(self, M_tot,sample_shape):
        Z_mean = self.MZR(M_tot)
        a = (self.Z_min - Z_mean)/self.MZR_sig
        b = (self.Z_max - Z_mean)/self.MZR_sig

        Z_samps  =  truncnorm.ppf(np.random.uniform(size = sample_shape),a,b, Z_mean, self.MZR_sig )
        return Z_samps
    
    def log_prob_Z_SSP(self,Z, M_tot):
        Z_mean = self.MZR(M_tot)
        a = (self.Z_min - Z_mean)/self.MZR_sig
        b = (self.Z_max - Z_mean)/self.MZR_sig
        return truncnorm.logpdf(Z,a,b, Z_mean, self.MZR_sig )

    def sample(self, sample_shape=torch.Size([])):
        

        M_samps = []
        A_samps = []
        for i in range(self.n_pop):
            M_samps.append(self.M_dist_list[i].sample(sample_shape) )
            A_samps.append(self.A_dist_list[i].sample(sample_shape) )
        
        M_samps = torch.stack(M_samps)
        A_samps = torch.stack(A_samps)
        M_tot = torch.log10((10**M_samps.squeeze()).sum(axis = 0) )
        
        Z_samps = []
        for i in range(self.n_pop):
            Z_samps.append( torch.Tensor(np.array(self.sample_Z_SSP(M_tot,sample_shape)) ) )
        
        Z_samps = torch.stack(Z_samps)
        
        samps = []
        samps.append( self.D_dist.sample(sample_shape) )
        for i in range(self.n_pop):
            samps.append(M_samps[i])
            samps.append(Z_samps[i].view(-1,*sample_shape).T )
            samps.append(A_samps[i])
            
        return torch.stack(samps).to(torch.float).T[0].to(self.device)

    def log_prob(self, values):

        values = values.to('cpu')
        
        log_prob = torch.zeros(values.shape[0])
        
        log_prob += self.D_dist.log_prob(values[:,0])
        
        M_tot = (10**values[:,1::3]).sum(axis = 1).log10()

        for i in range(self.n_pop):
            log_prob += self.M_dist_list[i].log_prob(values[:,3*i+1])
            log_prob += self.log_prob_Z_SSP(values[:,3*i+2], M_tot)
            log_prob += self.A_dist_list[i].log_prob(values[:,3*i+3])
        
        
        return log_prob.to(self.device)

def get_m_ssp_prior(Dlims,Mlims,logAgelims,device = 'cpu'):
    
    custom_prior = MSSP_Prior(Dlims,Mlims,logAgelims,device = device)
    
    lower_bounds = []
    upper_bounds = []
    
    lower_bounds.append(Dlims[0])
    upper_bounds.append(Dlims[1])
    
    if custom_prior.n_pop == 1:
        lower_bounds.append(Mlims[0])
        upper_bounds.append(Mlims[1])
        
        lower_bounds.append(-2.25)
        upper_bounds.append(0.25)
        
        lower_bounds.append(logAgelims[0])
        upper_bounds.append(logAgelims[1])
    
    else:
        for i in range(custom_prior.n_pop):
            lower_bounds.append(Mlims[i][0])
            upper_bounds.append(Mlims[i][1])
        
            lower_bounds.append(-2.25)
            upper_bounds.append(0.25)
        
            lower_bounds.append(logAgelims[i][0])
            upper_bounds.append(logAgelims[i][1])
            
    prior,_,_ = process_prior(
        custom_prior,
        custom_prior_wrapper_kwargs=
          dict(lower_bound=torch.Tensor(lower_bounds).to(device),
          upper_bound=torch.Tensor(upper_bounds).to(device))
        )

    return prior

class Default_Prior:
    def __init__(self,Dlims,Mlims, MZR_expand_fac = 1.,f_Y_max = 0.1,f_Y_sig = 0.02, device = 'cpu'):
        
        self.Dlims = torch.tensor(Dlims)
        self.D_dist = BoxUniform(low = [Dlims[0],],high = [Dlims[1],])
        self.Mlims = torch.tensor(Mlims)
        self.M_dist = BoxUniform(low = [Mlims[0],],high = [Mlims[1],])
        
        self.f_Y_max = f_Y_max 
        self.f_Y_sig = f_Y_sig
        self.f_M_dist = BoxUniform(low = [0,],high = [1,])

        self.age_Y_dist = BoxUniform(low = [8,],high = [8.7,])
        self.age_M_dist = BoxUniform(low = [8.7,],high = [9.7,])
        
        self.MZR_sig = 0.17*MZR_expand_fac
        self.Z_min = -2.25
        self.Z_max = 0.25
        self.device = device

    def MZR(self, logM):
        #Kirby+13 MZR
        return -1.69 + 0.3*(logM - 6.)
    
    def sample_Z_SSP(self, M_tot,sample_shape):
        Z_mean = self.MZR(M_tot).numpy() 
        a = (self.Z_min - Z_mean)/self.MZR_sig
        b = (self.Z_max - Z_mean)/self.MZR_sig
        Z_samps =  truncnorm.ppf(np.random.uniform(size = sample_shape),a,b, Z_mean, self.MZR_sig ) 
        if not isinstance(Z_samps,Iterable):
            Z_samps = [Z_samps,]
        return torch.Tensor(Z_samps)
    
    def log_prob_Z_SSP(self,Z, M_tot):
        Z_mean = self.MZR(M_tot).numpy()
        a = (self.Z_min - Z_mean)/self.MZR_sig
        b = (self.Z_max - Z_mean)/self.MZR_sig
        return truncnorm.logpdf(Z,a,b, Z_mean, self.MZR_sig )

    def sample_f_Y(self,sample_shape):
        a = 0
        b = (self.f_Y_max )/self.f_Y_sig
        f_Y_samps  =  truncnorm.ppf(np.random.uniform(size = sample_shape),a,b, 0, self.f_Y_sig )
        if not isinstance(f_Y_samps,Iterable):
            f_Y_samps = [f_Y_samps,]
        return  torch.Tensor(f_Y_samps )
    
    def log_prob_f_Y(self,f_Y):
        a = 0
        b = (self.f_Y_max )/self.f_Y_sig
        return truncnorm.logpdf(f_Y,a,b, 0, self.f_Y_sig )
    
    def sample(self, sample_shape=torch.Size([])):

        samps = []
        samps.append( self.D_dist.sample(sample_shape).view(sample_shape) )
        samps.append( self.M_dist.sample(sample_shape).view(sample_shape) )
        
        samps.append( self.sample_f_Y(sample_shape).view(sample_shape) )
        samps.append( self.f_M_dist.sample(sample_shape).view(sample_shape) )
        
        samps.append( self.age_Y_dist.sample(sample_shape).view(sample_shape) )
        samps.append( self.age_M_dist.sample(sample_shape).view(sample_shape) )
        
        samps.append( self.sample_Z_SSP(samps[1],sample_shape).view(sample_shape))

        
        return torch.stack(samps).to(torch.float).T.to(self.device)

    def log_prob(self, values):

        values = values.to('cpu')
        if values.ndim == 1:
            values = values.view(-1,values.shape[0])
        
        log_prob = torch.zeros(values.shape[0])
        
        log_prob += self.D_dist.log_prob(values[:,0])
        log_prob += self.M_dist.log_prob(values[:,1])
        
        log_prob += self.log_prob_f_Y(values[:,2])
        log_prob += self.f_M_dist.log_prob(values[:,3])
        
        log_prob += self.age_Y_dist.log_prob(values[:,4])
        log_prob += self.age_M_dist.log_prob(values[:,5])
        
        log_prob += self.log_prob_Z_SSP(values[:,6], values[:,1])
        
        return log_prob.to(self.device)

def get_default_prior(Dlims,Mlims, MZR_expand_fac = 1.,f_Y_max = 0.1,f_Y_sig = 0.02, device = 'cpu'):
    
    custom_prior = Default_Prior(Dlims,Mlims,MZR_expand_fac =MZR_expand_fac,f_Y_max = f_Y_max,f_Y_sig = f_Y_sig, device = device)
    
    lower_bounds = []
    upper_bounds = []
    
    #D
    lower_bounds.append(Dlims[0])
    upper_bounds.append(Dlims[1])
    
    #logM
    lower_bounds.append(Mlims[0])
    upper_bounds.append(Mlims[1])
    
    #f_Y
    lower_bounds.append(0)
    upper_bounds.append(f_Y_max)
    
    #f_M
    lower_bounds.append(0)
    upper_bounds.append(1)
        
    #log_age_Y
    lower_bounds.append(8)
    upper_bounds.append(8.7)
    
    #log_Age_M
    lower_bounds.append(8.7)
    upper_bounds.append(9.7)
    
    # Z
    lower_bounds.append(-2.25)
    upper_bounds.append(0.25)

    prior,_,_ = process_prior(
        custom_prior,
        custom_prior_wrapper_kwargs=
          dict(lower_bound=torch.Tensor(lower_bounds).to(device),
          upper_bound=torch.Tensor(upper_bounds).to(device))
        )

    return prior