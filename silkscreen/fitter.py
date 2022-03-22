import argparse
import torch
from collections import Iterable

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
        train_kwargs = {'training_batch_size': 64,'clip_max_norm': 5,'learning_rate':1e-4, 'validation_fraction':0.1}
        ):

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
                proposal = self.posteriors[-1].set_default_x(self.x_obs.view(1,*self.x_shape ))

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

            self.inference = self.inference.append_simulations(theta_cur,x_cur)

            #Work around for no when sbi v0.18 to make sure not too much mem used on gpu
            for j in range( len(self.inference._x_roundwise) ):
                self.inference._x_roundwise[j] = self.inference._x_roundwise[j].to('cpu')
                self.inference._theta_roundwise[j] = self.inference._theta_roundwise[j].to('cpu')

            density_estimator = self.inference.train(**train_kwargs)
            posterior = self.inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
        return posterior
