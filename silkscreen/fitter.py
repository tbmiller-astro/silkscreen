import torch
import copy
import gc
from tqdm import tqdm
from sbi.inference import SNPE
from .utils import parse_torch_sim_file, run_sims, parse_input_file,get_injec_cutouts
from .observation import SilkScreenObservation
from .simmer import ArtpopSimmer
from typing import Callable, Optional, Union, Iterable
import sbi
from sbi.inference import NeuralInference

def fit_silkscreen_model(
    observation: SilkScreenObservation,
    simmer: type,
    nde: Callable,
    prior: torch.distributions.distribution.Distribution,
    rounds: int,
    num_sim: Union[int,Iterable],
    device: Optional[str] = 'cpu',
    pre_simulated_file: Optional[str] = None,
    train_kwargs: Optional[dict] = {},
    data_device: Optional[str] = 'cpu',
    inject_image: Optional[str] = None,
    save_dir: Optional[str] = './silkscreen_results/',
    save_sims: Optional[bool] = False,
    save_posterior: Optional[bool] = False,
    )-> 'NeuralInference':
    """Function used to train SilkScreen Model

    Parameters
    ----------
    observations :  SilkScreenObservation
        observation object used to specify observations
    sim_class : ArtpopSimmer
        ArtpopSimmer class used to simulate data
    nde : 
        posterior_nn network from sbi, if none
    prior : torch.distributions.distribution.Distribution
        Prior used to draw parameters from, not strictly required if only training one round
    rounds : int
        Number of training and simulation rounds, more rounds means more targeted inference
    num_sim : Union(list, int)
        Number of simulations to simulate/train per round, can be list or int
    device : Optional[str], optional
        device used to perform training, highly reccomemded to be 'cuda' of 'gpu', by default 'cpu'
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

    sim_class = simmer(observation)
    
    inference =  SNPE(prior = prior, density_estimator = nde, device = device)
    
    default_train_kwargs = {'training_batch_size': 128,'clip_max_norm': 8,'learning_rate':1e-4, 'validation_fraction':0.1,'num_atoms':5} #Default training hyperparameters
    default_train_kwargs.update(train_kwargs)

    data_as_tensor = torch.Tensor(observation.data)

    if inject_image is not None:
        inject_data = parse_input_file(inject_image,output='torch')
        def sim_func(t): 
            im = sim_class.get_image_for_injec(t,output = 'torch') + get_injec_cutouts(1,observation.im_dim, array = inject_data, output = 'torch') 
            return im[0]
    
    else:
        def sim_func(t): 
            t = t.view(-1)
            im = sim_class.get_image(t, output = 'torch')
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
            
        append_sims_kwargs = {'proposal':proposal, 'data_device':data_device}
        inference.append_simulations(theta_cur,x_cur,**append_sims_kwargs)
        
        if save_sims and not (pre_simulated_file is not None and r == 0):
            torch.save([theta_cur,x_cur],f'{save_dir}sims_round_{r}.pt')

        del theta_cur,x_cur

        density_estimator = inference.train(**default_train_kwargs)
        
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

class SilkScreenFitter():
    def __init__(self,
        sim_function,
        nde,
        x_obs,
        prior,
        device = 'cpu',
        ):
        self.sim_function = sim_function

        if self.sim_function is None:
            self.has_sim_function = False
        else:
            self.has_sim_function = True

        self.prior = prior
        self.x_obs = x_obs
        self.x_shape = x_obs.shape
        self.device = device
        self.inference = SNPE(prior = prior, density_estimator = nde, device = device)

    def train_model(self,
        rounds = 1,
        num_sim = int(1e4),
        pre_simulated_file = None,
        train_kwargs = {},
        data_device = None,
        ):
        
        #Update any kwargs for infer.train
        def_train_kwargs = {'training_batch_size': 64,'clip_max_norm': 8,'learning_rate':1e-4, 'validation_fraction':0.1}
        def_train_kwargs.update(train_kwargs)

        ##Don't strictly need simulator but need to provide pre-sim and can't do more than one round
        if self.has_sim_function == False:
            assert pre_simulated_file is not None
            assert rounds == 1

        if isinstance(num_sim, Iterable): assert len(num_sim) == rounds

        self.posteriors = []

        for r in range(rounds):
            if r == 0:
                proposal = self.prior
            else:
                proposal = self.posteriors[-1].set_default_x(self.x_obs[None])

            num_r = num_sim[r] if isinstance(num_sim, Iterable) else num_sim

            #Can also use pre-simulated images from file for initial round
            if pre_simulated_file is not None and r == 0:
                theta_cur,x_cur = parse_torch_sim_file(pre_simulated_file)
            else:
                theta_cur,x_cur = self.run_sims(proposal, num_r)
                
            append_sims_kwargs = {'proposal':proposal, 'device':data_device}
            self.inference.append_simulations(theta_cur,x_cur,**append_sims_kwargs)
            del theta_cur,x_cur

            density_estimator = self.inference.train(**def_train_kwargs)
            
            # Return Posterior
            posterior = self.inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
            torch.cuda.empty_cache()
        return posterior
    
    def save_simulations(self,file_name,rounds):
        ###TODO This functions needs to be redone with new version of SBI ####
        if isinstance(rounds,Iterable):
            theta = []
            x = []
            for r in rounds:
                theta.append( self.inference._dataset.datasets[r].tensors[0].to('cpu') )
                x.append (self.inference._dataset.datasets[r].tensors[1].to('cpu') )
            theta = torch.stack(theta)
            x = torch.stack(x)
        else:
            theta = self.inference._dataset.datasets[rounds].tensors[0].to('cpu') 
            x = self.inference._dataset.datasets[rounds].tensors[1].to('cpu') 
        torch.save([theta,x],file_name)
    
    def pickle_posterior(self,file, r = -1):
        assert len(self.posteriors) >= 1
        post = copy.deepcopy(self.posteriors[r])
        
        post.set_default_x(self.x_obs[None])
        post.potential_fn.device = 'cpu'
        if hasattr(post.prior.custom_prior):
            post.prior.custom_prior.device = 'cpu'
        post._device = 'cpu'

        torch.save(post,file)
