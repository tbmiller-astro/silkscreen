##What Tim's basic workflow has looked like
import torch
import silkscreen
from silkscreen.neural_nets import build_default_NN

device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU is needed for training

#Load data
x_obs = torch.load('SOME/PATH/TO/CUTOUT.pt')
psfs =  torch.load('SOME/PATH/TO/PSFs.pt')

ser_dict = torch.load('SOME/PATH/TO/sersic_param_dict.pt') # Dictionary with parameters specifying morphology

# Basic setup for DECaLs images
iso_kwargs = dict(mag_limit=27, mag_limit_band='DECam_r') #magnitude limit for artpop to resolve individual stars
obs = silkscreen.SilkScreenObservation(data = x_obs, imager = 'DECam', filters = ['DECam_g','DECam_r', 'DECam_z'], sky_sb = [ 22.04, 20.91, 18.46],
exp_time = [87*2,67*2,100*2], pixel_scale = 0.262, zpt = 22.5, psf = psfs, distribution= 'sersic', distribution_kwargs=ser_dict, iso_kwargs=iso_kwargs,)

prior = silkscreen.priors.get_new_dwarf_fixed_age_prior(
    D_range=[1.5, 9.],
    logMs_range=[6., 8.],
    Fy_range=[0.0, 0.1],
    Fm_range=[0., 0.2],
    device=device,
) # Example of prior used in the initial paper

simmer_use = silkscreen.simmer.ContYoungDwarfSimmer # Simulator used in paper


#Specify training setup
nde = build_default_NN(num_filter=3, num_summary = 16)
num_sim = [25_000]*4
train_kwargs = {'retrain_from_scratch':False, 'discard_prior_samples':False, 'training_batch_size': 256}


infer = silkscreen.fit_silkscreen_model(obs,simmer_use , nde, prior, 4, num_sim , inject_image= 'for_inj.pt', # file containing large patch of sky to inject simulated images into
    device = device, data_device = 'cpu',n_jobs = 5, save_dir = './silkscreen_results/', save_posterior = True,lr_cnn = 5e-6, lr_flow = 2e-4, lr_mod = 1., train_kwargs = train_kwargs, norm_func = torch.asinh)

torch.save(infer.summary,'./silkscreen_results/training_summary.pt')
