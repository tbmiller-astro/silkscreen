
import artpop
import astropy.units as u
import numpy as np
import glob
from collections import Iterable
import torch
import os

class ArtpopSrcLoader():
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

    def build_sersic_ssp(self, log_Ms, dist,  feh, log_age,sersic_params = None):
        #Copied from artpop

        #Calc Isochrone
        iso_cur = self.fetch_iso(log_age,feh)

        if sersic_params is not None:
            ser_param = sersic_params.copy()
        else:
            ser_param = {'n':0.5, 'r_eff_as':10, 'theta': 0,'ellip':0}

        ser_param['r_eff_kpc'] = ser_param['r_eff_as']*np.pi/(180*3600) * dist*1e3

        #Generate artpop SSP
        ssp_cur = artpop.populations.SSP(iso_cur, 
                                        total_mass=10**log_Ms,
                                        distance=dist, 
                                        mag_limit=self.mag_limit, 
                                        mag_limit_band=self.mag_limit_band,
                                        imf=self.imf, 
                                        add_remnants=True, 
                                        random_state=None)

        source_cur = artpop.source.SersicSP(ssp_cur, 
                                            ser_param['r_eff_kpc'], 
                                            ser_param['n'], 
                                            ser_param['theta'],  
                                            ser_param['ellip'], 
                                            self.im_dim, 
                                            self.pixel_scale,
                                            num_r_eff=10, 
                                            dx=0, 
                                            dy=0, 
                                            labels=None)
        return source_cur

class ArtpopSimmer(ArtpopSrcLoader):
    "Class to produce many artpop simulations"
    def __init__(self, 
                imager,
                filters,
                exp_time,
                im_dim, 
                pixel_scale, 
                mag_limit = None,
                mag_limit_band = None,
                sky_sb = 22,
                psf = None):
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
        sky_sb: float, optional (default: 22)
            sky surface brightness to use, in mag/arcsec^2 
        psf: artpop psf object, optional (default: None)
            PSF to use. If none provided, a Moffat psf with a 0.7 arcsecond FWHM is used.
        '''
        self.imager = imager
        super().__init__(imager.phot_system, 
                        mag_limit=mag_limit, 
                        mag_limit_band=mag_limit_band, 
                        im_dim = im_dim, 
                        pixel_scale = pixel_scale)
        self.num_filters = len(filters)
        self.filters = filters

        ##All these can be dynamically overwritten

        #Can pass constant or list with len == num_filters
        if isinstance(sky_sb,Iterable): assert len(sky_sb) == self.num_filters
        self.sky_sb = sky_sb

        #Can pass constant or list with len == num_filters
        if isinstance(exp_time,Iterable): assert len(exp_time) == self.num_filters
        self.exp_time = exp_time

        #Can pass 2D array for all filters or 3D array with separate for each filter
        if psf is None:
            self.psf = artpop.moffat_psf(fwhm=0.7*u.arcsec,pixel_scale = pixel_scale)
        else:
            self.psf = psf

    def image_sersic_ssp(self, 
                        log_Ms,
                        dist,
                        feh,
                        log_age,
                        sersic_params = None,
                        num_shuffle = 1,
                        output = 'numpy'):
        src_cur = self.build_sersic_ssp(log_Ms, dist, feh, log_age, sersic_params = sersic_params)

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

                im_cur = self.imager.observe(src_cur,filt_cur, cur_exp_time*u.second, psf = cur_psf, sky_sb = cur_sky_sb)
                cur_shuf_list.append(im_cur.image)

            img_list.append(cur_shuf_list)
        img_list = np.asarray(img_list).squeeze()
        if output == 'torch': img_list = torch.from_numpy(img_list).type(torch.float)
        return img_list
