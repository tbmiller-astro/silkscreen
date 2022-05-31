
import artpop
import astropy.units as u
import numpy as np
import glob
import copy
from collections import Iterable
import torch
import os
from astropy.table import Table

class ArtpopIsoLoader():
    def __init__(self,
                phot_system,
                v_over_vcrit=0.4,
                ab_or_vega='ab',
                mag_limit=None,
                mag_limit_band=None,
                imf = 'kroupa',
                im_dim = 151,
                pixel_scale = 0.168):
        """
        Class to load all isochrones into memory, wrapping Artpop

        Parameters
        ----------
        phot_system: str
            photometric system as used by artpop -- must be in the dictionary defined by artpop.filters.get_filter_names()
        v_over_vcrit: float, optional (default: 0.4)
            Parameter needed for MIST isochrones.
        ab_or_vega: str, optional (default: 'ab')
            whether to use ab or vega magnitudes. Options are 'ab' or 'vega'.
        mag_limit: float, optional (default: None)
            Magnitude limit for which to resolve individual stars.
            Stars below this magnitude are not individually sampled and are estimated by a
            smooth component. By default, all stars are simulated.
        mag_limit_band: str, optional (default: None)
            Band in which the mag limit is calculated / applied.
        imf: str, optional (default: 'kroupa')
            IMF to sample stars from. Options include 'kroupa', 'chabrier'.
        im_dim: int, optional (default: 151)
            dimensions of the generated images. Images must be square.
        pixel_scale: float, optional (default: 0.168)
            pixel scale to apply to the image, in arcseconds per pixel. Set to depreciate
            (and be set automatically).
        """
        #Set up all variables
        self.phot_system = phot_system
        self.v_over_vcrit = v_over_vcrit
        self.ab_or_vega = ab_or_vega
        filter_dict = artpop.filters.get_filter_names()
        self.filters = filter_dict[phot_system].copy()
        self.zpt_convert = artpop.filters.load_zero_point_converter()

        self.imf = imf
        self.mag_limit = mag_limit
        self.mag_limit_band = mag_limit_band

        self.im_dim = im_dim
        self.pixel_scale = pixel_scale

        self.feh_grid = np.concatenate([np.arange(-3.0, -2., 0.5),
                                np.arange(-2.0, 0.75, 0.25)])

        #Download mist iso's if needed
        artpop.util.fetch_mist_grid_if_needed(phot_system, v_over_vcrit=v_over_vcrit)

        #Load all Isochrone objects into memory
        self.iso_list = []
        for feh in self.feh_grid:
            v = f'{v_over_vcrit:.1f}'
            ver = 'v1.2'
            p = artpop.stars.isochrones.phot_str_helper[phot_system.lower()]
            path = os.path.join(artpop.MIST_PATH, 'MIST_' + ver + f'_vvcrit{v}_' + p)
            sign = 'm' if feh < 0 else 'p'
            fn = f'MIST_{ver}_feh_{sign}{abs(feh):.2f}_afe_p0.0_vvcrit{v}_{p}.iso.cmd'
            fn = os.path.join(path, fn)
            iso_cmd = artpop.stars._read_mist_models.IsoCmdReader(fn, verbose=False)
            self.iso_list.append(iso_cmd)

    def fetch_iso(self, log_age, feh):
        #Use pre-looaded isolists
        ## Copied from artpop essentially but using pre-loaded files

        i_feh = self.feh_grid.searchsorted(feh)
        feh_lo, feh_hi = self.feh_grid[i_feh - 1: i_feh + 1]

        iso_0 = self.iso_list[i_feh-1]
        mist_0 = iso_0.isocmds[iso_0.age_index(log_age)]

        iso_1 = self.iso_list[i_feh+1]
        mist_1 = iso_1.isocmds[iso_1.age_index(log_age)]


        y0, y1 = np.array(mist_0.tolist()), np.array(mist_1.tolist())

        x = feh
        x0, x1 = feh_lo, feh_hi
        weight = (x - x0) / (x1 - x0)

        len_0, len_1 = len(y0), len(y1)

        # if necessary, extrapolate using trend of the longer array
        if (len_0 < len_1):
            delta = y1[len_0:] - y1[len_0 - 1]
            y0 = np.append(y0, y0[-1] + delta, axis=0)
        elif (len_0 > len_1):
            delta = y0[len_1:] - y0[len_1 - 1]
            y1 = np.append(y1, y1[-1] + delta, axis=0)

        y = y0 * (1 - weight) + y1 * weight
        iso = np.core.records.fromarrays(y.transpose(), dtype=mist_0.dtype)

        for filt in self.filters:
            converter = getattr(self.zpt_convert, f'to_{self.ab_or_vega.lower()}')
            try:
                m_convert = converter(filt)
            except AttributeError:
                m_convert = 0.0
                logger.warning(f'No AB / Vega conversion found for {filt}.')
            iso[filt] = iso[filt] + m_convert

        return artpop.isochrones.Isochrone(
            mini = iso['initial_mass'],
            mact = iso['star_mass'],
            mags = Table(iso[self.filters]),
            eep = iso['EEP'],
            log_L = iso['log_L'],
            log_Teff = iso['log_Teff'],)

    def build_ssp(self, log_Ms, dist,  feh, log_age):

        iso_cur = self.fetch_iso(log_age,feh)

        #Generate artpop SSP
        ssp_cur = artpop.populations.SSP(iso_cur,
                                        total_mass=10**log_Ms,
                                        distance=dist,
                                        mag_limit=self.mag_limit,
                                        mag_limit_band=self.mag_limit_band,
                                        imf=self.imf,
                                        add_remnants=True,
                                        random_state=None)
        return ssp_cur

    def build_sersic_ssp(self, log_Ms, dist,  feh, log_age,sersic_params = None):
        #Copied from artpop

        ssp_cur = self.build_ssp(log_Ms, dist,  feh, log_age)

        ser_param = {'n':0.5, 'r_eff_as':10, 'theta': 0,'ellip':0,'dx':0,'dy':0}
        ser_param.update(sersic_params)

        ser_param['r_eff_kpc'] = ser_param['r_eff_as']*np.pi/(180*3600) * dist*1e3

        #Generate artpop SSP

        source_cur = artpop.source.SersicSP(ssp_cur,
                                            ser_param['r_eff_kpc'],
                                            ser_param['n'],
                                            ser_param['theta'],
                                            ser_param['ellip'],
                                            self.im_dim,
                                            self.pixel_scale,
                                            num_r_eff=10,
                                            dx=ser_param['dx'],
                                            dy=ser_param['dy'],
                                            labels=None)
        return source_cur

class ArtpopSimmer(ArtpopIsoLoader):
    "Base Class to produce many artpop simulations"
    def __init__(self,
                imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                mag_limit = None,
                mag_limit_band = None,
                sky_sb = 21,
                zpt = 27,
                psf =  None,
                iso_kwargs = {},
                extinction_reddening = None):
        '''
        Initialize Class.

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
            PSF to use. If none provided, a Moffat psf with a 0.7 arcsecond FWHM is used.
        iso_kwargs: dict, optional (default :None)
            Additional keyword arguments to pass to ArtpopIsoLoader
        '''
        self.imager = imager
        super().__init__(imager.phot_system,
                        mag_limit=mag_limit,
                        mag_limit_band=mag_limit_band,
                        im_dim = im_dim,
                        pixel_scale = pixel_scale,
                        **iso_kwargs)
        self.num_filters = len(filters)
        self.filters = filters

        ##All these can be dynamically overwritten

        #Can pass constant or list with len == num_filters
        if isinstance(sky_sb,Iterable): assert len(sky_sb) == self.num_filters
        self.sky_sb = sky_sb

        if isinstance(zpt,Iterable): assert len(zpt) == self.num_filters
        self.zpt = zpt

        #Can pass constant or list with len == num_filters
        if isinstance(exp_time,Iterable): assert len(exp_time) == self.num_filters
        self.exp_time = exp_time

        #Can pass 2D array for all filters or 3D array with separate for each filter
        if psf is None:
            self.psf = artpop.moffat_psf(fwhm=0.7*u.arcsec,pixel_scale = pixel_scale)
        else:
            self.psf = psf
        
        self.extinction_reddening = extinction_reddening

    def build_source(x):
        ##Function which takes in array of values representing the free parameters
        ## and return ArtPop Src object
        return "Not Implemented yet"

    def get_image(self,
                    x,
                    num_shuffle = 1,
                    output = 'numpy'):

        src_cur = self.build_source(x)

        xy_to_shuffle = src_cur.xy.copy()

        img_list = []
        for i in range(num_shuffle):
            cur_shuf_list = []
            np.random.shuffle(xy_to_shuffle)
            src_cur.xy = xy_to_shuffle
            for j,filt_cur in enumerate(self.filters):

                cur_exp_time = self.exp_time[j] if isinstance(self.exp_time,Iterable) else self.exp_time
                cur_sky_sb =  self.sky_sb[j] if isinstance(self.sky_sb,Iterable) else self.sky_sb
                cur_psf = self.psf[j] if self.psf.ndim>2 else self.psf
                cur_zpt = self.zpt[j] if isinstance(self.zpt,Iterable) else self.zpt

                im_cur = self.imager.observe(src_cur,filt_cur, cur_exp_time*u.second, psf = cur_psf, sky_sb = cur_sky_sb)
                cur_shuf_list.append(im_cur.image)

            img_list.append(cur_shuf_list)
        img_list = np.asarray(img_list).squeeze()
        if self.extinction_reddening is not None:
            img_list *= self.extinction_reddening[:,None,None]
        
        if output == 'torch': img_list = torch.from_numpy(img_list).type(torch.float)
        return img_list

    def get_image_for_injec(self,
                            x,
                            num_shuffle = 1,
                            output = 'numpy'):
        # Same as above but us ideal imager instead
        # This isn't truly correcy and will underestimate noise if galaxy counts ~ sky counts
        # But for most of our setups this shouldn't be the case although important to check
        # Could maybe add a option to use Poisson noise if neeeded

        imager = artpop.IdealImager()

        src_cur = self.build_source(x)

        xy_to_shuffle = src_cur.xy.copy()

        img_list = []
        for i in range(num_shuffle):
            cur_shuf_list = []
            np.random.shuffle(xy_to_shuffle)
            src_cur.xy = xy_to_shuffle
            for j,filt_cur in enumerate(self.filters):

                cur_psf = self.psf[j] if self.psf.ndim>2 else self.psf
                cur_zpt = self.zpt[j] if isinstance(self.zpt,Iterable) else self.zpt

                im_cur = imager.observe(src_cur,filt_cur, psf = cur_psf,zpt = cur_zpt)
                cur_shuf_list.append(im_cur.image)

            img_list.append(cur_shuf_list)
        
        if self.extinction_reddening is not None:
            img_list *= self.extinction_reddening[:,None,None]
        img_list = np.asarray(img_list).squeeze()
        if output == 'torch': img_list = torch.from_numpy(img_list).type(torch.float)
        return img_list


class SersicSSPSimmer(ArtpopSimmer):
    "Class to simulate simple ssp with a sersic profile"
    def __init__(self,
                imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                sersic_params,
                mag_limit = None,
                mag_limit_band = None,
                sky_sb = 22,
                zpt = 27,
                psf = None,
                iso_kwargs = {}):

        super().__init__(imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                mag_limit = mag_limit,
                mag_limit_band = mag_limit_band,
                sky_sb = sky_sb,
                zpt = zpt,
                psf = psf,
                iso_kwargs = iso_kwargs)

        self.sersic_params = sersic_params
        # log Ms, Dist, Z and log age
        self.N_free = 4
        self.param_descrip = ['log Ms/Msun','D (Mpc)', 'Z','log Age (Gyr)']
    def build_source(self, x):
        "x is array with logMs,D,Z,logAge"
        logMs, D, Z, logAge = x.tolist()
        return self.build_sersic_ssp(logMs, D, Z, logAge,sersic_params = self.sersic_params)

class SersicTwoSSPSimmer(ArtpopSimmer):
    "Class to simulate a Two component ssp with a sersic profile"
    def __init__(self,
                imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                sersic_params,
                mag_limit = None,
                mag_limit_band = None,
                sky_sb = 22,
                zpt = 27,
                psf = None):

        super().__init__(imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                mag_limit,
                mag_limit_band,
                sky_sb,
                zpt,
                psf)

        self.sersic_params = sersic_params
        # log Ms, Dist, Z and log age
        self.N_free = 7
        self.param_descrip = ['log Ms/Msun','D (Mpc)', 'F_1', 'Z_1','log Age_1 (Gyr)', 'Z_2','log Age_2 (Gyr)']

    def build_source(self, x):
        logMs, D,F_1, Z_1, logAge_1,Z_2, logAge_2 = x.tolist()
        F_2 = 1.-F_1
        src_1 = self.build_sersic_ssp(logMs+np.log10(F_1), D, Z_1, logAge_1,sersic_params = self.sersic_params)
        src_2 = self.build_sersic_ssp(logMs+np.log10(F_2), D, Z_2, logAge_2,sersic_params = self.sersic_params)
        return src_1 + src_2

class SersicOMYSimmer(ArtpopSimmer):
    "Class to simulate a three component ssp with a sersic profile"
    def __init__(self,
                imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                sersic_params,
                mag_limit = None,
                mag_limit_band = None,
                sky_sb = 22,
                zpt = 27,
                psf = None):

        super().__init__(imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                mag_limit,
                mag_limit_band,
                sky_sb,
                zpt,
                psf)

        self.sersic_params = sersic_params
        # log Ms, Dist, Z and log age
        self.N_free = 7
        self.param_descrip = ['D (Mpc)', 'logM_y','F_m', 'Z_y','log Age_y (Gyr)','logM_m', 'Z_m','log Age_m (Gyr)', 'logM_o','Z_o','log Age_o (Gyr)']

    def build_source(self, x):
        D,logM_y, Z_y, logAge_y,logM_m, Z_m, logAge_m,logM_o,Z_o, logAge_o = x.tolist()
        
        src_y = self.build_sersic_ssp(logM_y, D, Z_y, logAge_y,sersic_params = self.sersic_params)
        src_m = self.build_sersic_ssp(logM_m, D, Z_m, logAge_m,sersic_params = self.sersic_params)
        src_o = self.build_sersic_ssp(logM_o, D, Z_o, logAge_o,sersic_params = self.sersic_params)

        return src_y + src_m + src_o

class Sersic_Default_Simmer(ArtpopSimmer):
    "Class to simulate a three component ssp with a sersic profile"
    def __init__(self,
                imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                sersic_params,
                mag_limit = None,
                mag_limit_band = None,
                sky_sb = 22,
                zpt = 27,
                psf = None,
                extinction_reddening = None):

        super().__init__(imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                mag_limit,
                mag_limit_band,
                sky_sb,
                zpt,
                psf,
                extinction_reddening = extinction_reddening)

        self.sersic_params = sersic_params
        self.N_free = 7
        self.param_descrip = ['D (Mpc)', 'logMs','F_y', 'F_m','log Age_y (Gyr)','log Age_m (Gyr)', 'Z']

    def build_source(self, x):
        D,logM, f_y,f_m, logAge_y,logAge_m, Z = x.tolist()
        
        f_o = 1. - f_m
        logAge_o = 10.
        
        src_y = self.build_sersic_ssp(logM + np.log10(f_y + 1e-6), D, Z, logAge_y, sersic_params = self.sersic_params)
        src_m = self.build_sersic_ssp(logM + np.log10(f_m + 1e-6), D, Z, logAge_m, sersic_params = self.sersic_params)
        src_o = self.build_sersic_ssp(logM + np.log10(f_o + 1e-6), D, Z, logAge_o, sersic_params = self.sersic_params)

        return src_o + src_m + src_y
    
class SersicDefault2PopSimmer(ArtpopSimmer):
    "Class to simulate a three component ssp with a sersic profile"
    def __init__(self,
                imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                sersic_params,
                mag_limit = None,
                mag_limit_band = None,
                sky_sb = 22,
                zpt = 27,
                psf = None,
                extinction_reddening = None):

        super().__init__(imager,
                filters,
                exp_time,
                im_dim,
                pixel_scale,
                mag_limit,
                mag_limit_band,
                sky_sb,
                zpt,
                psf,
                extinction_reddening = extinction_reddening)

        self.sersic_params = sersic_params
        self.N_free = 5
        self.param_descrip = ['D (Mpc)', 'logMs','F_y', 'log Age_y (Gyr)', 'Z']

    def build_source(self, x):
        D,logM, f_y, logAge_y, Z = x.tolist()
        
        f_o = 1. - f_y
        logAge_o = 10.
        
        src_y = self.build_sersic_ssp(logM + np.log10(f_y + 1e-6), D, Z, logAge_y, sersic_params = self.sersic_params)
        src_o = self.build_sersic_ssp(logM + np.log10(f_o + 1e-6), D, Z, logAge_o, sersic_params = self.sersic_params)

        return src_o +  src_y