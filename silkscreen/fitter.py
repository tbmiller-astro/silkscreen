import torch
from torch import optim
import copy
import gc
from tqdm import tqdm
from sbi.inference import SNPE
from .utils import parse_torch_sim_file, run_sims, parse_input_file,get_injec_cutouts
from .observation import SilkScreenObservation
from .simmer import ArtpopSimmer
from typing import Callable, Optional, Union, Iterable
import sbi
from sbi.utils import RestrictedPrior, get_density_thresholder
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
    lr_flow = 1e-4,
    lr_cnn = 1e-4,
    lr_mod = 0.1,
    )-> NeuralInference:
    """Function used to train SilkScreen Model

    Parameters
    ----------
    observations :  SilkScreenObservation
        observation object used to specify observations
    sim_class : ArtpopSimmer
        ArtpopSimmer class used to simulate data
    nde : 
        posterior_nn network from sbi
    prior : torch.distributions.distribution.Distribution
        Prior used to draw parameters from, not strictly required if only training one round
    rounds : int
        Number of training and simulation rounds, more rounds means more targeted inference
    num_sim : Union(list, int)
        Number of simulations to simulate/train per round, can be list or int
    device : Optional[str], optional
        device used to perform training, highly reccomemded to be 'cuda' or 'gpu', by default 'cpu'
    pre_simulated_file : Optional[str], optional
        location of file containing pre-simulated parameters and data to be used in the first round of training, by default None
    train_kwargs : Optional[dict], optional
        kwargs passed to `inference.train()` method, by default None and defaults will be used
    data_device : Optional[str], optional
        where to store data, defaults is the 'cpu' to free up more memory on the 'gpu' if used, but this is slightly slower
    inject_image : Optional[str], optional
        location of file containing real data to inject simulated data into. can be 'pt','npy' or 'fits'
    save_dir : Optional[str], optional
        Location to save results, by default './silkscreen_results/'
    save_sims : Optional[bool], optional
        Whether or not to save simulated data, by default False
    save_posterior : Optional[bool], optional
        whether or not to pickle posterior, by default False

    Returns
    -------
    NeuralInference
        sbi NeuralInference object containing trained model
    """

    sim_class = simmer(observation)
    
    inference =  SNPE(prior = prior, density_estimator = nde, device = device)
    
    default_train_kwargs = {'training_batch_size':64,'clip_max_norm': 10,'learning_rate':1e-4, 'validation_fraction':0.1,'stop_after_epochs': 25} #Default training hyperparameters
    default_train_kwargs.update(train_kwargs)

    data_as_tensor = torch.Tensor(observation.data)

    if inject_image is not None:
        inject_data = parse_input_file(inject_image,output='torch')
        def sim_func(t):
            t = t.view(-1)
            im = sim_class.get_image_for_injec(t,output = 'torch') + get_injec_cutouts(1,observation.im_dim, array = inject_data, output = 'torch').squeeze()
            return im
    
    else:
        def sim_func(t): 
            t = t.view(-1)
            im = sim_class.get_image(t, output = 'torch')
            return im
    
    if isinstance(num_sim, Iterable): assert len(num_sim) == rounds

    for r in range(rounds):
        if r == 0:
            proposal = prior
            sampling_kwargs = {}
        else:
            #Set up truncated SNPE
            posterior.set_default_x(data_as_tensor[None].to(device))
            
            samples = []
            log_prob = []
            for j in range(int(50000/32) +1):
                samples.append( posterior.sample((32,),show_progress_bars = False, ) )
                log_prob.append(  posterior.log_prob(samples[-1]) )
            log_probs = torch.stack(log_prob).flatten()
            log_prob_threshold = log_probs.sort()[0][int(log_probs.shape[0]*1e-3)]
            
            def accept_reject_fn(theta):
                theta_log_probs = posterior.log_prob(theta)
                predictions = theta_log_probs > log_prob_threshold
                return predictions.bool()
            
            proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection",device=device)
            sampling_kwargs = dict(max_sampling_batch_size = 32,show_progress_bars = False)
        
        num_r = num_sim[r] if isinstance(num_sim, Iterable) else num_sim

        #Can also use pre-simulated images from file for initial round
        if pre_simulated_file is not None and r == 0:
            theta_cur,x_cur = parse_torch_sim_file(pre_simulated_file)
        else:
            theta_cur,x_cur = run_sims(sim_func, proposal, num_r, samp_kwargs = sampling_kwargs )
            
        append_sims_kwargs = {'proposal':proposal, 'data_device':data_device}
        inference.append_simulations(theta_cur,x_cur,**append_sims_kwargs)
        
        if save_sims and not (pre_simulated_file is not None and r == 0):
            torch.save([theta_cur,x_cur],f'{save_dir}sims_round_{r}.pt')

        del theta_cur,x_cur
        
        
        w_decay_cnn = 1e-4
        w_decay_flow = 1e-4
        if r == 0:
            #Trick sbi using 'force_first_round_loss'
            _ = inference.train(max_num_epochs = -1,**default_train_kwargs) # Run to initialize optimizer

            inference.optimizer = optim.Adam([{'params': inference._neural_net._transform.parameters(), 'lr': lr_flow, 'weight_decay':w_decay_flow},
               {'params': inference._neural_net._embedding_net.parameters(), 'lr': lr_cnn, 'weight_decay':w_decay_cnn}], lr=1e-4)        
            density_estimator = inference.train(force_first_round_loss=True, resume_training= True, **default_train_kwargs)
        else:
            inference.optimizer = optim.Adam([{'params': inference._neural_net._transform.parameters(), 'lr': lr_flow*lr_mod, 'weight_decay':w_decay_flow},
               {'params': inference._neural_net._embedding_net.parameters(), 'lr': lr_cnn*lr_mod, 'weight_decay':w_decay_cnn}], lr=1e-4)        
            density_estimator = inference.train(force_first_round_loss=True,**default_train_kwargs)
            
        posterior = inference.build_posterior(density_estimator)
        
        if save_posterior:
            post = copy.deepcopy(posterior)
            
            post.set_default_x(data_as_tensor[None])
            post.potential_fn.device = 'cpu'
            post._device = 'cpu'
            
            if hasattr(post.prior, 'custom_prior'):
                post.prior.custom_prior.device = 'cpu'

            torch.save(post,f'{save_dir}posterior_round_{r}.pt')
        
        if device == 'cuda': torch.cuda.empty_cache()
        gc.collect()
    return inference
