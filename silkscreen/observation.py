
from typing import Optional, Union, Iterable
import artpop
from . import utils
import numpy as np

class SilkScreenObservation(object):
    '''Class used to store observationial details
    #TODO Update docs below
    Parameters
    ----------
    imager: Artpop Imager object
    filters: list
        list of filters for images
    exp_time: float
        exposure time to use (seconds)
    im_im: int
        image dimensions
    pixel_scale: float
        pixel scale for the instrument
    mag_limit: float, optional (default: None)
        Magnitude limit for which to resolve individual stars.
        Stars below this magnitude are not individually sampled and are estimated by a
        smooth component. By default, all stars are simulated.
    mag_limit_band: str, optional (default: None)
        Band in which the mag limit is calculated / applied.
    sky_sb: float, optional (default: 21)
        sky surface brightness to use, in mag/arcsec^2
    psf: artpop psf object, optional (default: None)
        PSF to use, if using multiple bands, the length of the first axis must be equal to the number of bands
    iso_kwargs: dict, optional (default :None)
        Additional keyword arguments to pass to ArtpopIsoLoader   
    '''
    def __init__(self,
            data: np.array,
            imager: Union[str, artpop.ArtImager],
            filters: Iterable,
            exp_time: Union[float, Iterable],
            pixel_scale: float,
            zpt: Union[float, Iterable],
            distribution: str,
            distribution_kwargs: dict,
            psf: np.array,
            sky_sb: Optional[Union[float, Iterable]] = 21.,
            iso_kwargs: Optional[dict] = {},
            extinction_reddening: Optional[Iterable]= None):
        
        self.data = data
        if len(data.shape) == 2:
            self.num_filt = 1
            self.im_dim = data.shape
        elif len(data.shape) == 3:
            self.num_filt = data.shape[0]
            self.im_dim = data.shape[1:]
        else:
            print ('Data must be either 2 or 3 dimensions')
            raise AttributeError
        
        if type(imager) is str:
            assert imager in ['DECam', 'HSC']
            self.imager = getattr(utils, f'get_{imager}_imager')()
        else:
            assert  isinstance(imager, artpop.ArtImager),'imager argument must be an artpop.ArtImager object'
            self.imager = imager
        
        self.pixel_scale = pixel_scale

        assert len(filters) == self.num_filt, f'Mismatch between data and filters, data implies {filters} filters but specified {self.num_filt}'
         
        self.filters = filters # Check filters in artpop

        self.exp_time = exp_time
        self.sky_sb = sky_sb
        self.zpt = zpt
        self.psf = psf
        if self.num_filt == 1:
            assert len(self.psf.shape) == 2, "For a single band PSF must be 2 dimensional"
        else:
            assert len(self.psf.shape) == 3 and self.psf.shape[0] == self.num_filt, "For multiple bands, the PSF array must be 3 dimensional with the first axis representing each bandß"
        
        self.iso_kwargs = iso_kwargs
        self.extinction_reddening = extinction_reddening
        
        assert distribution.lower() in ['sersic', 'plummer']
        self.distribution = distribution
        self.distribution_kwargs = distribution_kwargs

        if distribution.lower() == 'sersic':
            assert ('r_eff_as' in self.distribution_kwargs and 'n' in self.distribution_kwargs),"When using a sersic distribution must specify at least 'r_eff_as' and 'n' in 'distribtuion_kwargs'"

        if self.distribution.lower() == 'plummer':
            assert 'scale_radius_as' in self.distribution_kwargs, "When using a plummer distribution must specify at least scale_radius_as' in 'distribtuion_kwargs'"
