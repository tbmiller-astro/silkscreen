import yaml 
from silkscreen import simmer
from silkscreen import utils
from silkscreen.fitter import SilkScreenFitter
from sbi.utils.get_nn_models import posterior_nn

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
            imager = utils.get_DECam_imager()
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
            prior = utils.get_default_2pop_prior(prior_dict['log_distance'],
                                                prior_dict['log_mass'],
                                                MZR_expand_fac=prior_dict['MZR_expansion_factor'])
        
        # Parse network
        network_configs = config_dict['network']
        enet = utils.Default_NN(num_filt=3,nout=network_configs['n_channels'],im_size=im_dim,dropout_p=network_configs['dropout'])
        nde = posterior_nn('nsf',embedding_net=enet,z_score_theta='none',z_score_x='none')
        self.fitter = SilkScreenFitter(sim.get_image,nde,x_obs,prior,device='cpu')
        self.training_params = config_dict['training']

    def train_model(self,model_save_path=None):
        self.posterior = self.fitter.train_model(rounds=self.training_params['rounds'],
                                                num_sim = self.training_params['num_sim'])
        if model_save_path is not None:
            self.posterior.pickle_posterior(model_save_path, r = -1)
