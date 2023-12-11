import torch
import numpy as np
from sbi.utils import BoxUniform,MultipleIndependent
from typing import Iterable,Optional
from torch.distributions.constraints import independent,interval
from sbi.utils.user_input_checks_utils import ScipyPytorchWrapper, MultipleIndependent, CustomPriorWrapper
from scipy.stats import truncnorm
from scipy.special import erf
from torch import tensor
from torch.distributions import Uniform
from typing import Iterable, Optional
import numpy as np
from sbi.utils import BoxUniform

class MZRPrior():
    def __init__(self,logM_bounds, frac_expand = 1.5, device = 'cpu'):
        self.logM_bounds = logM_bounds
        self.logM_dist = build_uniform_dist(logM_bounds,device)
        self.MZR_sig = 0.17*frac_expand
        self.Z_bounds = [-2.25,0.5]
        self.device = device

    # def MZR(self, logM): ## COMMENTED OUT AS A SUGGESTED CHANGE
    
    #     if logM <= 8.7:
    #         feh = -1.69 + 0.3*(logM - 6.) #Kirby+2013 https://ui.adsabs.harvard.edu/abs/2013ApJ...779..102K/abstract
    #         #scatter is ~0.2 dex

    #     else:
    #         logmass_ = [8.91, 9.11, 9.31, 9.51, 9.72, 9.91, 10.11, 10.31, 10.51, 10.72, 10.91, 11.11, 11.31, 11.51, 11.72, 11.91]
    #         feh_ = [-0.60, -0.61, -0.65, -0.61, -0.52, -0.41, -0.23, -0.11, -0.01, 0.04, 0.07, 0.1, 0.12, 0.13, 0.14, 0.15]
    #         mzr_fit = np.polyfit(logmass_, feh_, 6)
    #         feh = np.poly1d(mzr_fit)(logM) #Gallazzi+2005 https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G/abstract
    #         #scatter is ~0.3 dex

    #     return feh
        
    @staticmethod
    def KirbyMZR(logM):
        return -1.69 + 0.3*(logM - 6.)
    
    def _log_prob_Z(self,Z,logM):
        Z_mean = self.KirbyMZR(logM).cpu().numpy()
        a = (self.Z_bounds[0]- Z_mean)/self.MZR_sig
        b = (self.Z_bounds[1]- Z_mean)/self.MZR_sig
        return  torch.Tensor ( truncnorm.logpdf(Z.cpu(),a,b, Z_mean, self.MZR_sig ) ).to(self.device)
    
    def _sample_Z(self, logM, sample_shape): 
        Z_mean = self.KirbyMZR(logM.cpu().view(sample_shape)).cpu()
        a = (self.Z_bounds[0] - Z_mean)/self.MZR_sig
        b = (self.Z_bounds[1] - Z_mean)/self.MZR_sig
        Z_samps =  truncnorm.ppf(np.random.uniform(size = sample_shape),a,b, Z_mean, self.MZR_sig ) 
        if not isinstance(Z_samps,Iterable):
            return  torch.tensor(Z_samps, dtype=torch.float32).to(self.device)
        return torch.Tensor(Z_samps).to(self.device)
    
    def log_prob(self,x):
        log_prob = self.logM_dist.log_prob(x[:,0])
        log_prob += self._log_prob_Z(x[:,1],x[:,0])
        return log_prob
    
    def sample(self, sample_shape = torch.Size([])):
        logM_samps = self.logM_dist.sample(sample_shape=sample_shape).view(sample_shape)
        Z_samps = self._sample_Z(logM_samps, sample_shape)
        return torch.stack([logM_samps, Z_samps]).T

class MyTruncNorm():
    def __init__(self,loc,sig, bounds, device = 'cpu'):
        self.loc = loc
        self.sig = sig
        self.a = (bounds[0] - self.loc)/self.sig
        self.b = (bounds[1] - self.loc)/self.sig
        self.Z = 0.5* ( erf(self.b/np.sqrt(2)) - erf(self.a/np.sqrt(2)) )
        self.phi_a = 1./np.sqrt(2*np.pi)* np.exp(-0.5*self.a**2)
        self.phi_b = 1./np.sqrt(2*np.pi)* np.exp(-0.5*self.b**2)
        self.device = device
        
    @property
    def mean(self):
        return self.loc + (self.phi_a - self.phi_b)/self.Z * self.sig
    
    @property
    def variance(self):
        var_frac = 1. - (self.b*self.phi_b - self.a*self.phi_a)/self.Z - (self.phi_a - self.phi_b)**2/self.Z**2 
        return self.sig**2 *var_frac
    
    def log_prob(self, x):
        x_cpu = x.cpu().numpy()
        return torch.Tensor ( truncnorm.logpdf(x_cpu,self.a,self.b, self.loc, self.sig ) ).to(self.device)

    def sample(self,sample_shape = torch.Size([])):
        samps =  truncnorm.ppf(np.random.uniform(size = sample_shape),self.a,self.b, self.loc, self.sig ) 
        if not isinstance(samps,Iterable):
            return  torch.tensor([samps,], dtype=torch.float32).to(self.device)
        return torch.tensor(samps).to(self.device).reshape(-1,1)
    
def build_uniform_dist(bounds, device):
    return Uniform(tensor([bounds[0]], dtype = torch.float32).to(device), tensor([bounds[1]], dtype = torch.float32).to(device) )

def build_truncnorm_dist(loc,scale, bounds, device):
    
    custom_dist = MyTruncNorm(loc,scale,bounds,device= device)
    
    lb = torch.tensor([bounds[0]]).to(device)
    ub = torch.tensor([bounds[1]]).to(device)
    
    return CustomPriorWrapper(custom_dist, event_shape=torch.Size([1]),lower_bound = lb, upper_bound = ub)

def build_mzr_dist(logMs_range, device):
    custom_dist = MZRPrior(logMs_range, device = device)
    lb =  torch.tensor([logMs_range[0], -2.25]).to(device)
    ub =  torch.tensor([logMs_range[1], 0.5]).to(device)
    return CustomPriorWrapper(custom_dist, event_shape=torch.Size([2]), lower_bound = lb, upper_bound = ub )



def get_default_dwarf_fixed_age_prior(
        D_range: Iterable,
        logMs_range: Iterable,
        device: Optional[str] = 'cpu'
    )-> torch.distributions.distribution.Distribution:
    

    D_dist = build_uniform_dist(D_range, device)
    M_and_Z_dist = build_mzr_dist(logMs_range, device)
    fy_dist = build_truncnorm_dist(0, 0.05, [0.,0.2], device )
    ay_n_dist = build_uniform_dist([0.5,5.], device)

    fm_dist = build_truncnorm_dist(0.4, 0.2, [0.,0.8], device )
    prior = MultipleIndependent([D_dist,M_and_Z_dist,fy_dist, ay_n_dist, fm_dist])

    return prior

def get_SSP_prior(
        D_range: Iterable,
        logMs_range: Iterable,
        Age_range: Optional[Iterable] = [0.1, 12],
        Z_range: Optional[Iterable] = [-2.25,0.25],
        device: Optional[str] = 'cpu'
    )-> torch.distributions.distribution.Distribution:
    
    unif_bounds_tensor = torch.tensor([D_range,logMs_range, Age_range, Z_range]).to(device)
    unif_dist = BoxUniform(unif_bounds_tensor[:,0], unif_bounds_tensor[:,1], device= device)
    return unif_dist