##What Tim's basic workflow has looked like
import torch
from silkscreen import utils
from silkscreen import simmer
from silkscreen.fitter import SilkScreenFitter
from sbi.utils.get_nn_models import posterior_nn

# Need to re-do this but shouldn't be too hard

#Load data
x_obs = torch.load('SOME/PATH/TO/data.pt')

#initialize simmer
simmer = simmer.SersicSSPSimmer( utils.get_DECam_imager(),['DECam_g','DECam_r','DECam_z'], [30,30,60],(101,101), 0.262)

#initialize NN
enet = utils.Default_NN(num_filt = 3,nout = 16, im_size = (101,101),dropout_p = 0.)
nde = posterior_nn('nsf',embedding_net=enet, z_score_theta='none', z_score_x='none')
prior = utils.get_default_prior([1,2],[6,8])

#Put it all together into the fitter class
fitter = SilkScreenFitter(simmer.get_image, nde, x_obs, prior, device = 'cpu')

#Sim models and Train
posterior = fitter.train_model(rounds = 3, num_sim = [int(1e4),int(5e3),int(5e3)])

##Then save the final posterior
fitter.pickle_posterior('PATH/TO/SAVE/posterior.pkl')