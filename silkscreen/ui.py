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
from silkscreen.simmer import SersicSSPSimmer, SersicTwoSSPSimmer,SersicThreeSSPSimmer, DefaultSersicSimmer

#TODO All this stuff needs to be re-worked a little bit with the changes to fitter.py and simmer.py

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
#Imad's  class
######

class SilkscreenFitter():
    """
    Class for initializing user observations
    """
    def __init__(self,
                image_path,
                instrument: str,
                image_bands: list,
                exp_times: list,
                image_dim: tuple,
                ):
        """
        Initialize an observation object. 

        Parameters
        ----------
        image: pt, npy, or fits file
            image to fit
        instrument: str or `artpop.ArtImager`
            telescope / imager of the input observations. Currently supported: 'DECam', 'HSC'
        image_bands: list
            names of the bands for the input observations. 
        """
        self.image = self.parse_input(image_path)
        self.parse_imager(instrument,image_bands)
        self.exp_times = exp_times
        self.image_dim = image_dim

    def parse_input(self,image_path,output='torch'):
        return utils.parse_input_file(image_path,output=output)
    def parse_imager(self,imager,image_bands):
        if isinstance(imager,str):
            if imager not in ['DECam','HSC']:
                raise AssertionError(f'imager not in supported list: DECam, HSC')
            elif imager == 'DECam':
                self.imager = artpop.image.ArtImager('DECam', diameter = 4.0*u.m, read_noise = 7)
                self.pixel_scale = 0.262
                convert_dict = {'g':'DECam_g',
                        'r':'DECam_r',
                        'i':'DECam_i',
                        'z':'DECam_z'}
                self.bands = []
                for i in image_bands:
                    if i in convert_dict.values():
                        self.bands.append(i)
                    elif i.lower() in convert_dict.keys():
                        self.bands.append(convert_dict[i])
                    else:
                        raise AssertionError(f'Band name {i} not recognized for this instrument.')
            elif imager == 'HSC':
                self.imager = artpop.image.ArtImager('HSC', diameter = 8.4*u.m, read_noise = 4.5)
        else:
            self.imager = imager
            self.bands = image_bands
    def select_model(self,model):
        assert model in ['SersicSSP','SersicTwoSSP','SersicThreeSSP','DefaultSersicSSP']
        self._modstr = model
        if model == 'SersicSSP':
            self.simmer = SersicSSPSimmer(self.imager,self.bands,self.exp_times,self.image_dim,self.pixel_scale)
        elif model == 'SersicTwoSSP':
            self.simmer = SersicTwoSSPSimmer(self.imager,self.bands,self.exp_times,self.image_dim,self.pixel_scale)
        elif model == 'SersicThreeSSP':
            self.simmer = SersicThreeSSPSimmer(self.imager,self.bands,self.exp_times,self.image_dim,self.pixel_scale)
        elif model == 'DefaultSersicSimmer':
            self.simmer = DefaultSersicSimmer(self.imager,self.bands,self.exp_times,self.image_dim,self.pixel_scale)

    def initialize_NN(self,nout=16,dropout=0.5,z_score_theta='independent',z_score_x='structured'):
        self.enet = utils.Default_NN(num_filt=len(self.bands),
                                    nout=nout,
                                    im_size=self.image_dim,
                                    dropout=dropout)
        self.nde = posterior_nn('nsf',
                                embedding_net=self.enet,
                                z_score_theta=z_score_theta,
                                z_score_x = z_score_x)
        
    
    def set_priors(self,distance_prior,mass_prior,**kwargs):
        if self._modstr == 'SersicSSP':
            self.prior = utils.get_default_prior(distance_prior,mass_prior,**kwargs)
        elif self._modstr == 'SersicTwoSSP':
            self.prior = utils.get_default_2pop_prior(distance_prior,mass_prior,**kwargs)
        elif self._modstr == 'SersicThreeSSP':
            raise NotImplementedError('Not Yet Implemented')

    def train_NN(self,rounds=3,
                num_simulations = [int(1e4),int(5e3),int(5e3)],
                save_posterior_to = './posterior.pkl',
                device='cpu'):
        self.fitter = Fitter(self.simmer.get_image,self.nde,self.image,self.prior,device=device)
        self.posterior = self.fitter.train_model(rounds=rounds,
                                                num_sim=num_simulations)
        self.fitter.pickle_posterior(save_posterior_to)


####
# Imad's Newer Class
####
from .observation import SilkScreenObservation
from .simmer import ArtpopSimmer
import torch
import copy
import gc
from sbi.inference import SNPE
from .utils import parse_torch_sim_file, run_sims, parse_input_file,get_injec_cutouts
from .observation import SilkScreenObservation
from .simmer import ArtpopSimmer
from typing import Callable, Optional, Union, Iterable
import sbi
from sbi.inference import NeuralInference

class Silkscreen():

    def __init__(self,
                obs_object:SilkScreenObservation,
                simulator: ArtpopSimmer):
        """
        Container Class for running SilkScreen

        Parameters
        ----------
        obs_object: SilkScreenObservation
            the object containing the observation parameters and image data
        simulator: ArtpopSimmer (callable)
            Name of the ArtpopSimmer to use for simulations
        """
        self.obs_object = obs_object  
        self.simmer = simulator(obs_object) 
    
    def list_model_params(self):
        print('The current model parameters are:')
        print(self.simmer.param_descrip)
    
    def simulate_images(self,
                        params: Iterable,
                        n_images: int = 1,
                        ):
        """
        Simulate images using the established simulator.

        Parameters
        ----------
        params: Iterable
            the `theta` or input parameters to the simulator. The required list for a setup is accessible via the `.list_model_params()` method.
        n_images: int, default: 1
            number of independent constructions/images to return
        """
        img_list = []
        for i in range(n_images):
            img_list.append(self.simmer.get_image(params))
        return img_list
            

    def train_and_fit(self, 
                    nde: Callable,
                    prior: Optional[torch.distributions.distribution.Distribution],
                    rounds: int,
                    num_sim: Union[int,Iterable],
                    device: Optional[str] = 'cpu',
                    pre_simulated_file: Optional[str] = None,
                    train_kwargs: Optional[dict] = None,
                    data_device: Optional[str] = 'cpu',
                    inject_image: Optional[str] = None,
                    save_dir: Optional[str] = './silkscreen_results/',
                    save_sims: Optional[bool] = False,
                    save_posterior: Optional[bool] = False,
                    )-> 'NeuralInference':
        """Function used to train SilkScreen Model
        Parameters
        ----------
        nde : 
            posterior_nn network from sbi
            _description_
        prior : Optional[torch.distributions.distribution.Distribution]
            Prior used to draw parameters from
        rounds : int
            Number of training and simulation rounds, more rounds means more targeted inference
        num_sim : Union(list, int)
            Number of simulations to simulate/train per round, can be list or int
        device : Optional[str], optional
            device used to perform training, highly reccomemded to be 'cuda', by default 'cpu'
        pre_simulated_file : Optional[str], optional
            locaiton of file containing pre-simulated parameters and data to be used in the first round of training, by default None
        train_kwargs : Optional[dict], optional
            kwargs passed to `inference.train()` method, by default None and defaults will be used
        data_device : Optional[str], optional
            where to store data, defaults is the 'cpu' to free up more memory on the 'gpu' if used, but this is slightly slower
        inject_image : Optional[str], optional
            location of file containing real data to inject simulated data into. can be 'pt','npt' or 'fits'
        save_dir : Optional[str], optional
            Location to save results, by default './silkscreen_results/'
        save_sims : Optional[bool], optional
            Whether or not to save simulated data, by default False
        save_posterior : Optional[bool], optional
            where or not to pickle posterior, by default False
        Returns
        -------
        NeuralInference
            sbi Neural Inference object containing trained model
        """

        inference =  SNPE(prior = prior, density_estimator = nde, device = device)
        
        default_train_kwargs = {'training_batch_size': 100,'clip_max_norm': 8,'learning_rate':1e-4, 'validation_fraction':0.1,
            'z_score_theta':'independent', 'z_score_x':'structured'}
        default_train_kwargs.update(train_kwargs)
        sim_class = self.simmer
        data_as_tensor = torch.Tensor(sim_class.obs_object.data)

        if inject_image is not None:
            inject_data = parse_input_file(inject_image,output='torch')
            def sim_func(t): 
                im = sim_class.get_image_for_injec(t,output = 'torch') + get_injec_cutouts(1,sim_class.obs_object.im_dim, array = inject_data, output = 'torch') 
                return im[0]
        
        else:
            def sim_func(t): 
                t = t.view(-1)
                im = sim_class.get_image(t)
                return im
        
        if isinstance(num_sim, Iterable): assert len(num_sim) == rounds

        for r in range(rounds):
            if r == 0:
                proposal = prior
            else:
                proposal = posterior.set_default_x(data_as_tensor[None])

            num_r = num_sim[r] if isinstance(num_sim, Iterable) else num_sim

            #Can also use pre-simulated images from file for initial round
            if pre_simulated_file is not None and r == 0:
                theta_cur,x_cur = parse_torch_sim_file(pre_simulated_file)
            else:
                theta_cur,x_cur = run_sims(sim_func, proposal, num_r)
                
            append_sims_kwargs = {'proposal':proposal, 'device':data_device}
            inference.append_simulations(theta_cur,x_cur,**append_sims_kwargs)
            
            if save_sims and not (pre_simulated_file is not None and r == 0):
                torch.save([theta_cur,x_cur],f'{save_dir}sims_round_{r}.pt')

            del theta_cur,x_cur

            density_estimator = inference.train(**default_train_kwargs)
            
            # Return Posterior
            posterior = inference.build_posterior(density_estimator)
            
            if save_posterior:
                post = copy.deepcopy(posterior)
                
                post.set_default_x(data_as_tensor[None])
                post.potential_fn.device = 'cpu'
                if hasattr(post.prior.custom_prior):
                    post.prior.custom_prior.device = 'cpu'
                post._device = 'cpu'

                torch.save(post,f'{save_dir}posterior_round_{r}.pt')
            
            if device == 'cuda': torch.cuda.empty_cache()
            gc.collect() 
        return inference
