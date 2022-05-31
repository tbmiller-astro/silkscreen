import argparse
import torch
from collections import Iterable
import copy
import _pickle as Cpickle

from sbi import analysis
from sbi import inference
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi


class SilkScreenFitter():
    def __init__(self,
        sim_function,
        nde,
        x_obs,
        prior,
        device = 'cpu',
        sim_function_kwargs = {}
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
        self.sim_function_kwargs = sim_function_kwargs

    def train_model(self,
        rounds = 1,
        num_sim = int(1e4),
        pre_simulated_file = None,
        train_kwargs = {},
        append_sims_kwargs = {},
        ):
        
        #Update any kwargs for infer.train
        def_train_kwargs = {'training_batch_size': 64,'clip_max_norm': 5,'learning_rate':1e-4, 'validation_fraction':0.1}
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

            #Can also use pre-simulated images from file for initial round
            if pre_simulated_file is not None and r == 0:
                theta_cur,x_cur = torch.load(pre_simulated_file)

            else:
                #Draw samples
                if isinstance(num_sim, Iterable):
                    theta_cur = proposal.sample((num_sim[r],)).to('cpu')
                else:
                    theta_cur = proposal.sample((num_sim,)).to('cpu')

                x_cur = []
                for theta in theta_cur:
                    x_cur.append(self.sim_function(theta, **self.sim_function_kwargs)[None])

                x_cur = torch.vstack(x_cur)
                
                ## make sure number of thetas matches if shuffled images.
                if 'num_shuffle' in self.sim_function_kwargs:
                    theta_cur = a[:,None,:] *torch.ones(sim_function_kwargs['num_shuffle' ])[None,:,None]
                    theta_cur = theta_cur.view(x_cur.shape[0],-1)
            
            print (x_cur.shape,theta_cur.shape)
            append_sims_kwargs.update({'proposal':proposal})
            
            self.inference.append_simulations(theta_cur,x_cur,**append_sims_kwargs)
            density_estimator = self.inference.train(**def_train_kwargs)
            # Return Posterior
            posterior = self.inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
            torch.cuda.empty_cache()
        return posterior
    
    def save_simulations(self,file_name,rounds):
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
        torch.save([theta,x],'a')
    
    def pickle_posterior(self,file, r = -1):
        assert len(self.posteriors) >= 1
        post = copy.deepcopy(self.posteriors[r])
    
        post.posterior_estimator.to('cpu')
        post.prior.custom_prior.device = 'cpu'
        post.prior.support.base_constraint.lower_bound = post.prior.support.base_constraint.lower_bound.to('cpu')
        post.prior.support.base_constraint.upper_bound = post.prior.support.base_constraint.upper_bound.to('cpu')
        with open(file, "wb") as pkl_file:
            Cpickle.dump(post, pkl_file)
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
