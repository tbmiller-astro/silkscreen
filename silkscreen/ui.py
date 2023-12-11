from silkscreen import fitter
import yaml 
import numpy as np
from torch import save as torch_save
from silkscreen import simmer
from silkscreen import utils as ssu
from silkscreen.DenseNet import DenseNet
from sbi.utils.get_nn_models import posterior_nn

from typing import Union
import artpop 
import astropy.units as u 
from silkscreen import utils

#Might be nice to set up the ability to read all the info from a config file? All of this stuff is out of date but we could re-work it

def initialize_from_config(config):
    if type(config) == dict:
        config_dict = config.copy()
    elif type(config) == str:
        with open(config,'r') as stream:
            config_dict = yaml.safe_load(stream)
    else:
        print ("Must use either dictionary or string with location of yaml file")
        return None
    
    #Copy basic set up parameters
    device = config_dict['training']['training_device']
    sim_dict = config_dict['simulator'].copy()

    #Load PSFs
    psf_arr = []
    for f in sim_dict.pop('psf_files'):
        psf_arr.append( ssu.parse_input_file(f,output = 'numpy') )
    psf_arr = np.stack(psf_arr)

    sim_dict['psf'] = psf_arr

    #Load observed data and file to inject simulated images into
    obs_data = ssu.parse_input_file(config_dict['obs_file']) 
    sim_dict['im_dim'] = [obs_data.shape[-2],obs_data.shape[-1]]
    inject_file = sim_dict.pop('inject_file')

    #load artpop imager
    instr = sim_dict.pop('instrument')
    imager_func = getattr(ssu, f'get_{instr}_imager')
    
    #load artpop simmer
    sim_class = getattr(simmer, sim_dict.pop('type' ) )
    sim_class = sim_class(imager_func(), **sim_dict)

    #Define function to simulate images
    if inject_file is None:
        sim_func =  lambda t: sim_class.get_image(t, output = 'torch')
    else:
        inject_array = ssu.parse_input_file(inject_file, output='numpy')
        def sim_func(t):
            t = t.view(-1)
            im = sim_class.get_image_for_injec(t, output = 'torch')
            im += ssu.get_injec_cutouts(1, sim_class.im_dim, array = inject_array, output = 'torch')
            return im

    #Set up prior
    prior_dict = config_dict['prior'].copy()
    prior_type = prior_dict.pop('type').lower()
    prior_func =  getattr(ssu,f'get_{prior_type}_prior')
    prior = prior_func(**prior_dict, device = device)

    #Set up network
    network_dict = config_dict['network'].copy()
    enet = DenseNet( **network_dict['CNN'] )
    nde = posterior_nn(**network_dict['Flow'], embedding_net = enet)

    #Set up silkscreen fitter class
    fitter_class = SilkScreenFitter(sim_func,nde, obs_data, prior,device = config_dict['training']['training_device'])

    return sim_class,fitter_class,config_dict

def run_sims_from_config(config, num,save_loc = None):
    #Simulates image from prior
    _,fitter_class, config_dict = initialize_from_config(config)
    t,x = fitter_class.run_sims(fitter_class.prior, num)
    if save_loc is not None:
        torch_save([t,x],save_loc)
    return t,x


def train_model_from_config(config, save_loc = None):
    #(nitialize fitter class
    _,fitter_class, config_dict = initialize_from_config(config)
    training_dict = config_dict['training']

    #Run training
    posterior = fitter_class.train_model(rounds =training_dict['rounds'], num_sim=training_dict['num_sims'], 
       pre_simulated_file=training_dict['pre_simulated_file'], train_kwargs= training_dict['train_kwargs'], 
       data_device= training_dict['data_device'])
    
    #save and return fitter class
    if save_loc is not None:
        fitter_class.pickle_posterior(save_loc,r = -1)
    return fitter_class