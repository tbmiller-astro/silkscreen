from sbi import utils as sbi_utils
from sbi.inference import posteriors

import torch
import torch.nn as nn
import torch.nn.functional as F
import artpop
import astropy.units as u

default_sersic_dict = {'n':0.5, 'r_eff_as':10, 'theta': 0,'ellip':0,'dx':0,'dy':0}

def block_mean(x, num_block):
    r1 = x.shape[1]%num_block
    r2 = x.shape[2]%num_block
    x = x[:,:-r1,:-r2]
    x = x.reshape(x.shape[0],int(x.shape[1]/num_block), num_block,int(x.shape[2]/num_block), num_block)
    return x.mean(axis = (2,4) )

#Basic function to reutrn common imagres
def get_DECam_imager():
    return artpop.image.ArtImager('DECam', diameter = 4.0*u.m, read_noise = 7)

def get_HSC_imager():
    return artpop.image.ArtImager('HSC', diameter = 8.4*u.m, read_noise = 4.5)

def get_injec_cutouts(num, size,files = None, array = None ,output = 'numpy', pad = 0):
    #function to create cutouts to inject real images into

    if files is not None:
        obs_ims = []
        for f in files:
            obs_ims.append(fits.getdata(f) )
        obs_ims = np.asarray(obs_ims)
    elif array is not None:
        obs_ims = np.asarray(array)
    else:
        print("Must specify files or array")

    x_max = obs_ims.shape[1]
    y_max = obs_ims.shape[2]

    #add padding to not deal with edges
    x = np.arange(0,size[0])
    y = np.arange(0,size[1])
    X,Y = np.meshgrid(x,y)

    inds_X = X + np.random.randint(low = pad,high = x_max - size[0]-pad, size = num)[:,None,None]
    inds_Y = Y + np.random.randint(low = pad,high = y_max - size[1]-pad, size = num)[:,None,None]

    #Extract cutouts
    cutouts = obs_ims[:,inds_X,inds_Y]

    cutouts = np.moveaxis(cutouts,0,1)

    if output == 'torch': cutouts = torch.from_numpy(cutouts).type(torch.float)
    return cutouts

def load_post(prior, enet, state_dict, im_shape, flow = 'maf'):
    ## Function to load post from state_dict
    ## this is super finicky

    #Need example data
    t_start = prior.sample((2,))
    x_start = torch.ones((2,*im_shape))

    #initialize model
    nde = sbi_utils.posterior_nn(model=flow, embedding_net=enet,z_score_theta = False,z_score_x = False)
    net = nde(t_start,x_start)

    #Load trained parameters
    net.load_state_dict(state_dict )

    #Return sbi object
    return posteriors.direct_posterior.DirectPosterior(net, prior, x_shape = (1,*im_shape) )

class New_NN(nn.Module):
    def __init__(self, num_filt = 3,nout = 16, im_size = (151,151),dropout_p = 0.5,hidden_fc_features = 4096):
        super().__init__()
        self.num_filt = int(num_filt)
        self.nout = int(nout)
        self.shape = im_size
        self.hidden_fc_features = hidden_fc_features

        #define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.num_filt, out_channels=16, kernel_size=10,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,stride = 2)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride = 2,padding = 0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride = 2,padding = 0)
        #Combine all layers
        self.conv_layers = nn.Sequential(self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.pool1,self.conv3,nn.ReLU(),self.conv4,nn.ReLU(),self.pool2)

        #Test convolutional layers to determine shape of output
        test_tensor = torch.ones( (1,self.num_filt, self.shape[0],self.shape[1]) )
        test_conv = self.conv_layers(test_tensor )
        self.num_fc_input = test_conv.numel()

        #Define FC layers
        self.fc1 = nn.Linear(in_features=self.num_fc_input, out_features=self.hidden_fc_features)
        self.fc2 = nn.Linear(in_features=self.hidden_fc_features, out_features=self.hidden_fc_features)
        self.fc3 = nn.Linear(in_features=self.hidden_fc_features, out_features=self.nout)
        self.dout = nn.Dropout(p=dropout_p)
        #Combine FC layers
        self.fc_layers = nn.Sequential(self.fc1, self.dout,nn.ReLU(), self.fc2, self.dout,nn.ReLU(),self.fc3)

    def forward(self, x):
        x = x.view(-1, int(self.num_filt), self.shape[0],self.shape[1])
        x = torch.asinh(x)
        x = self.conv_layers(x)
        x = x.view(-1,self.num_fc_input)
        x = self.fc_layers(x)
        return x

class Default_NN(nn.Module):
    #Default NN we have been using, can specify image size and number of filters
    def __init__(self, num_filt = 3,nout = 16, im_size = (151,151),dropout_p = 0.5):
        super().__init__()
        self.num_filt = int(num_filt)
        self.nout = int(nout)
        self.shape = im_size

        #define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.num_filt, out_channels=32, kernel_size=10,padding = 1,stride = 3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride = 2,padding = 0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding = 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride = 4,padding = 0)
        #Combine all layers
        self.conv_layers = nn.Sequential(self.conv1,nn.ReLU(),self.pool1,self.conv2,self.conv3,nn.ReLU(),self.pool2)

        #Test convolutional layers to determine shape of output
        test_tensor = torch.ones( (1,self.num_filt, self.shape[0],self.shape[1]) )
        test_conv = self.conv_layers(test_tensor )
        self.num_fc_input = test_conv.numel()

        #Define FC layers
        self.fc1 = nn.Linear(in_features=self.num_fc_input, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=self.nout)
        self.dout = nn.Dropout(p=dropout_p)
        #Combine FC layers
        self.fc_layers = nn.Sequential(self.fc1, self.dout,nn.ReLU(), self.fc2, self.dout,nn.ReLU(),self.fc3)

    def forward(self, x):
        x = x.view(-1, int(self.num_filt), self.shape[0],self.shape[1])
        x = torch.asinh(x)
        x = self.conv_layers(x)
        x = x.view(-1,self.num_fc_input)
        x = self.fc_layers(x)
        return x
