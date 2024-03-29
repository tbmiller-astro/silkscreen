
import torch
from tqdm.auto import tqdm
from sbi import utils as sbi_utils
from sbi.inference import posteriors
from sbi.utils import BoxUniform
from sbi.utils import process_prior
from typing import Iterable, Optional
import gc
from astropy.io import fits
import mpire

import torch.nn as nn
import torch.nn.functional as F
import artpop
import astropy.units as u
import numpy as np

default_sersic_dict = {'n':0.7, 'r_eff_as':10, 'theta': 0,'ellip':0,'dx':0,'dy':0}

def run_sims(sim_func, proposal, num, n_jobs = 1, samp_kwargs = {}) -> torch.Tensor:
    theta = proposal.sample((num,), **samp_kwargs).to('cpu')# Always need on cpu
    theta_to_samp = [t for t in theta]

    with mpire.WorkerPool(n_jobs) as pool:
        x = []
        for result in pool.imap(sim_func,theta_to_samp,progress_bar = True, chunk_size = 500, max_tasks_active = 2500):
            x.append(result)
    
    print ('done with worker pool')
    x = torch.stack(x).to('cpu')
    print ('done with stacking')
    return theta,x

def parse_input_file(location, output = 'torch'):
    suffix = location.split('.')[-1]
    assert suffix in ['pt','npy','fits']
    if suffix == 'pt':
        obs_data = torch.load(location)
    elif suffix == 'npy':
        arr =  np.load(location)
        
        if arr.dtype.byteorder == '>': # Ensure correct byte order to transfer to torch
            arr = arr.byteswap().newbyteorder('<')
        obs_data = torch.from_numpy(arr)
    elif suffix == 'fits':
        data = fits.getdata(location)
        if data.dtype.byteorder == '>':
            data = data.byteswap().newbyteorder('<')
        obs_data = torch.from_numpy(data )
    else:
        return 0

    if output == 'numpy':
        return obs_data.numpy()
    return obs_data

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

def parse_torch_sim_file(obj):
    if isinstance(obj, str):
        t_all,x_all = torch.load(obj)
    else:
        for i,f in enumerate(obj):
            t_cur,x_cur = torch.load(f)
            if i == 0:
                per_file = t_cur.shape[0]
                N_param = t_cur.shape[1]
                im_size = tuple(x_cur.shape[1:])
            
                x_all = torch.ones((per_file*len(obj),*im_size) )
                t_all = torch.ones((per_file*len(obj),N_param) )
            
            t_all[i*per_file:(i+1)*per_file] = t_cur
            x_all[i*per_file:(i+1)*per_file] = x_cur           
            del t_cur,x_cur
            
    return t_all,x_all

def get_reddening(coords,filts):
    from dustmaps.sfd import SFDQuery
    import extinction
    
    ebv = SFDQuery()(coords)
    
    tab = artpop.filters.get_filter_properties()
    lam_eff = np.hstack([tab[tab['bandpass'] == filt]['lam_eff'].value for filt in filts] )
    flux = np.ones(len(filts))
    
    return extinction.apply(extinction.calzetti00(lam_eff, 3.1*ebv, 3.1), flux)

