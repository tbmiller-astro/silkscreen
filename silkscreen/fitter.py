import torch
from collections import Iterable
import copy

from sbi.inference import SNPE
from .utils import parse_torch_sim_file

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

    def run_sims(self, proposal, num):
        theta = proposal.sample((num,)).to('cpu')# Always need on cpu
        x = []
        for theta_cur in theta:
            x.append(self.sim_function(theta_cur))

        x = torch.stack(x)
        return theta,x


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
'''
def_train_kwargs.update({'max_num_epochs':30,'stop_after_epochs':30})
self.density_estimator_init = self.inference.train(**def_train_kwargs)
            
# 'Freeze' parameters in CNN
for key, param  in self.inference._neural_net.named_parameters():
    if 'embedding' in key:
        param.requires_grad = False

# Continue training with just Flow
def_train_kwargs.update({'learning_rate':1e-4,'force_first_round_loss':True})
def_train_kwargs.pop('max_num_epochs')
def_train_kwargs.pop('stop_after_epochs')
density_estimator = self.inference.train(**def_train_kwargs) ''' 
