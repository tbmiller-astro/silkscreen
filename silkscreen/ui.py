from silkscreen import fitter
import yaml 
import numpy as np
from torch import save as torch_save
from silkscreen import simmer
from silkscreen import utils as ssu
from silkscreen.DenseNet import DenseNet
from silkscreen.fitter import SilkScreenFitter
from sbi.utils.get_nn_models import posterior_nn

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



######
#Below is Imad's old class
######

class SilkScreen():
    def __init__(self,config_file):
        self.parse_config(config_file)

    def parse_config(self,config_file):
        with open(config_file,'r') as stream:
            config_dict = yaml.safe_load(stream)
        # Parse Simulator
        sim_configs = config_dict['simulator']
        bands = sim_configs['band']
        exptime = sim_configs['exptime']
        im_dim = sim_configs['im_dim']
        pixel_scale = sim_configs['pixel_scale']
        if sim_configs['instrument'] == 'DECam':
            imager = ssu.get_DECam_imager()
        if sim_configs['type'] == 'SersicSSP':
            sim = simmer.SersicSSPSimmer(imager,bands,exptime,im_dim,pixel_scale)
        elif sim_configs['type'] == 'SersicTwoSSP':
            sim = simmer.SersicTwoSSPSimmer(imager,bands,exptime,im_dim,pixel_scale)
        elif sim_configs['type'] == 'SersicOMYSimmer':
            sim = simmer.SersicOMYSimmer(imager,bands,exptime,im_dim,pixel_scale)
        elif sim_configs['type'] == 'Sersic Default':
            sim = simmer.Sersic_Default_Simmer(imager,bands,exptime,im_dim,pixel_scale)
        elif sim_configs['type'] == 'Sersic Default 2pop':
            sim = simmer.SersicDefault2PopSimmer(imager,bands,exptime,im_dim,pixel_scale)
        else:
            raise AssertionError('sim type must be in [SersicSSP,SersicTwoSSP,SersicOMYSimmer,Sersic Default, Sersic Default 2pop].')
        self.sim = sim 
        # Parse priors
        prior_dict = config_dict['prior']
        if prior_dict['type'] == '2pop':
            prior = ssu.get_default_2pop_prior(prior_dict['log_distance'],
                                                prior_dict['log_mass'],
                                                MZR_expand_fac=prior_dict['MZR_expansion_factor'])
        
        # Parse network
        network_configs = config_dict['network']
        enet = ssu.Default_NN(num_filt=3,nout=network_configs['n_channels'],im_size=im_dim,dropout_p=network_configs['dropout'])
        nde = posterior_nn('nsf',embedding_net=enet,z_score_theta='none',z_score_x='none')
        
        #self.fitter = SilkScreenFitter(sim.get_image,nde,x_obs,prior,device='cpu')
        self.training_params = config_dict['training']

    def train_model(self,model_save_path=None):
        self.posterior = self.fitter.train_model(rounds=self.training_params['rounds'],
                                                num_sim = self.training_params['num_sim'])
        if model_save_path is not None:
            self.posterior.pickle_posterior(model_save_path, r = -1)
