import os
from typing import Iterable, Optional

import artpop
import astropy.units as u
import numpy as np
import torch
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
from scipy.stats import norm
from .observation import SilkScreenObservation

class ArtpopIsoLoader:
    def __init__(
        self,
        phot_system,
        v_over_vcrit=0.4,
        ab_or_vega="ab",
        mag_limit=None,
        mag_limit_band=None,
        imf="kroupa",
    ):
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
        """
        # Set up all variables
        self.phot_system = phot_system
        self.v_over_vcrit = v_over_vcrit
        self.ab_or_vega = ab_or_vega
        filter_dict = artpop.filters.get_filter_names()
        self.filters = filter_dict[phot_system].copy()
        self.zpt_convert = artpop.filters.load_zero_point_converter()

        self.imf = imf
        self.mag_limit = mag_limit
        self.mag_limit_band = mag_limit_band

        self.feh_grid = np.concatenate(
            [np.arange(-3.0, -2.0, 0.5), np.arange(-2.0, 0.75, 0.25)]
        )

        # Download mist iso's if needed
        artpop.util.fetch_mist_grid_if_needed(phot_system, v_over_vcrit=v_over_vcrit)

        # Load all Isochrone objects into memory
        self.iso_list = []
        for feh in self.feh_grid:
            v = f"{v_over_vcrit:.1f}"
            ver = "v1.2"
            p = artpop.stars.isochrones.phot_str_helper[phot_system.lower()]
            path = os.path.join(artpop.MIST_PATH, "MIST_" + ver + f"_vvcrit{v}_" + p)
            sign = "m" if feh < 0 else "p"
            fn = f"MIST_{ver}_feh_{sign}{abs(feh):.2f}_afe_p0.0_vvcrit{v}_{p}.iso.cmd"
            fn = os.path.join(path, fn)
            iso_cmd = artpop.stars._read_mist_models.IsoCmdReader(fn, verbose=False)
            self.iso_list.append(iso_cmd)

    def fetch_iso(self, log_age, feh):
        # Use pre-looaded isolists
        ## Copied from artpop essentially but using pre-loaded files

        i_feh = self.feh_grid.searchsorted(feh)
        feh_lo, feh_hi = self.feh_grid[i_feh - 1 : i_feh + 1]

        iso_0 = self.iso_list[i_feh - 1]
        mist_0 = iso_0.isocmds[iso_0.age_index(log_age)]

        iso_1 = self.iso_list[i_feh + 1]
        mist_1 = iso_1.isocmds[iso_1.age_index(log_age)]

        y0, y1 = np.array(mist_0.tolist()), np.array(mist_1.tolist())

        x = feh
        x0, x1 = feh_lo, feh_hi
        weight = (x - x0) / (x1 - x0)

        len_0, len_1 = len(y0), len(y1)

        # if necessary, extrapolate using trend of the longer array
        if len_0 < len_1:
            delta = y1[len_0:] - y1[len_0 - 1]
            y0 = np.append(y0, y0[-1] + delta, axis=0)
        elif len_0 > len_1:
            delta = y0[len_1:] - y0[len_1 - 1]
            y1 = np.append(y1, y1[-1] + delta, axis=0)

        y = y0 * (1 - weight) + y1 * weight
        iso = np.core.records.fromarrays(y.transpose(), dtype=mist_0.dtype)

        for filt in self.filters:
            converter = getattr(self.zpt_convert, f"to_{self.ab_or_vega.lower()}")
            try:
                m_convert = converter(filt)
            except AttributeError:
                m_convert = 0.0
                artpop.logger.warning(f"No AB / Vega conversion found for {filt}.")
            iso[filt] = iso[filt] + m_convert

        return artpop.isochrones.Isochrone(
            mini=iso["initial_mass"],
            mact=iso["star_mass"],
            mags=Table(iso[self.filters]),
            eep=iso["EEP"],
            log_L=iso["log_L"],
            log_Teff=iso["log_Teff"],
        )

    def build_ssp(self, log_Ms, dist, feh, log_age):
        iso_cur = self.fetch_iso(log_age, feh)

        # Generate artpop SSP
        ssp_cur = artpop.populations.SSP(
            iso_cur,
            total_mass=10**log_Ms,
            distance=dist,
            mag_limit=self.mag_limit,
            mag_limit_band=self.mag_limit_band,
            imf=self.imf,
            add_remnants=True,
            random_state=None,
        )
        return ssp_cur


class ArtpopSimmer(ArtpopIsoLoader):
    "Base Class to produce many artpop simulations based on SilkScreen Observations Object"

    def __init__(self, obs_object: SilkScreenObservation):
        self.obs_object = obs_object
        super().__init__(
            self.obs_object.imager.phot_system, **self.obs_object.iso_kwargs
        )
        self.ideal_imager = artpop.IdealImager()
        self.ideal_psf = artpop.gaussian_psf(fwhm=1., pixel_scale=1., mode = 'oversample', factor=25)

    def build_sp(self, x):
        raise NotImplementedError

    def build_source(self, x: torch.Tensor, dist_kwargs = {}):
        sp_use = self.build_sp(x)
        defaults_kwargs = {"theta": 0, "ellip": 0, "dx": 0, "dy": 0}
        if self.obs_object.distribution.lower() == "sersic":
            ser_param = {**defaults_kwargs, **self.obs_object.distribution_kwargs}
            ser_param.update(dist_kwargs)# Update if provided, if not default to what was built in
            ser_param["r_eff_kpc"] = (
                ser_param["r_eff_as"]
                * np.pi
                / (180 * 3600)
                * sp_use.distance.to(u.kpc).value
            )
            source_cur = artpop.source.SersicSP(
                sp_use,
                ser_param["r_eff_kpc"],
                ser_param["n"],
                ser_param["theta"],
                ser_param["ellip"],
                self.obs_object.im_dim,
                self.obs_object.pixel_scale,
                num_r_eff=5,
                dx=ser_param["dx"],
                dy=ser_param["dy"],
                labels=None,
            )

        elif self.obs_object.distribution.lower() == "plummer":
            plummer_param = {**defaults_kwargs, **self.obs_object.distribution_kwargs}
            ser_param.update(dist_kwargs) # Update if provided, if not default to what was built in
            plummer_param["scale_radius_kpc"] = (
                plummer_param["scale_radius_as"]
                * np.pi
                / (180 * 3600)
                * sp_use.distance.to(u.kpc).value
            )
            source_cur = artpop.source.PlummerSP(
                sp_use,
                plummer_param["scale_radius_kpc"],
                self.obs_object.im_dim,
                self.obs_object.pixel_scale,
                dx=plummer_param["dx"],
                dy=plummer_param["dy"],
                labels=None,
            )
        return source_cur

    def get_artpop_obs(self, src, filt_ind):
        filt_cur = self.obs_object.filters[filt_ind]
        ## Check to get the right properties
        cur_exp_time = (
            self.obs_object.exp_time[filt_ind]
            if isinstance(self.obs_object.exp_time, Iterable)
            else self.obs_object.exp_time
        )
        cur_sky_sb = (
            self.obs_object.sky_sb[filt_ind]
            if isinstance(self.obs_object.sky_sb, Iterable)
            else self.obs_object.sky_sb
        )
        cur_psf = (
            self.obs_object.psf[filt_ind]
            if self.obs_object.psf.ndim > 2
            else self.obs_object.psf
        )
        cur_zpt = (
            self.obs_object.zpt[filt_ind]
            if isinstance(self.obs_object.zpt, Iterable)
            else self.obs_object.zpt
        )

        im_cur = self.obs_object.imager.observe(
            src,
            filt_cur,
            cur_exp_time * u.second,
            psf=cur_psf,
            sky_sb=cur_sky_sb,
            zpt=cur_zpt,
        )
        return im_cur
    
    def get_artpop_ideal_obs(self,src, filt_ind):
        filt_cur = self.obs_object.filters[filt_ind]
        ## Check to get the right properties
        cur_exp_time = (
            self.obs_object.exp_time[filt_ind]
            if isinstance(self.obs_object.exp_time, Iterable)
            else self.obs_object.exp_time
        )

        cur_zpt = (
            self.obs_object.zpt[filt_ind]
            if isinstance(self.obs_object.zpt, Iterable)
            else self.obs_object.zpt
        )

        im_cur = self.ideal_imager.observe(
            src,
            filt_cur,
            psf=self.ideal_psf,
            zpt=cur_zpt,
        )
        return im_cur

    def get_image(
        self, x: torch.Tensor, num_shuffle: Optional[int] = 1, output="numpy", dist_kwargs = {},
    ):
        src_cur = self.build_source(x, dist_kwargs=dist_kwargs)

        xy_to_shuffle = src_cur.xy.copy()

        img_list = []
        for i in range(num_shuffle):
            cur_shuf_list = []
            np.random.shuffle(xy_to_shuffle)
            src_cur.xy = xy_to_shuffle
            for j, filt_cur in enumerate(self.obs_object.filters):
                im_cur = self.get_artpop_obs(src_cur, j)
                cur_shuf_list.append(im_cur.image)

            img_list.append(cur_shuf_list)
        img_list = np.asarray(img_list).squeeze()
        if self.obs_object.extinction_reddening is not None:
            img_list *= self.obs_object.extinction_reddening[:, None, None]

        if output == "torch":
            img_list = torch.from_numpy(img_list.astype(np.float32)).type(torch.float)
        return img_list

    def get_image_for_injec(self, x, num_shuffle=1, output="numpy", dist_kwargs = {}, return_ideal = False):
        # Same as above but us ideal imager instead
        # This isn't truly correct and will underestimate noise if galaxy counts ~ sky counts
        # But for most of our setups this shouldn't be the case although important to check
        # Could maybe add a option to use Poisson noise if needed

        src_cur = self.build_source(x, dist_kwargs=dist_kwargs)

        xy_to_shuffle = src_cur.xy.copy()

        img_list = []
        ideal_list = []
        for i in range(num_shuffle):
            cur_shuf_list = []
            ideal_shuf_list = []
            np.random.shuffle(xy_to_shuffle)
            src_cur.xy = xy_to_shuffle
            for j, filt_cur in enumerate(self.obs_object.filters):
                obs_cur = self.get_artpop_obs(src_cur, j)
                im_cur = obs_cur.image
                cur_shuf_list.append(im_cur)
                if return_ideal:
                    im_ideal = self.get_artpop_ideal_obs(src_cur, j)
                    ideal_shuf_list.append(im_ideal.image)
            img_list.append(cur_shuf_list)
            ideal_list.append(ideal_shuf_list)

        if self.obs_object.extinction_reddening is not None:
            img_list *= self.obs_object.extinction_reddening[:, None, None]
            if return_ideal:
                ideal_list *= self.obs_object.extinction_reddening[:, None, None]
        
        img_list = np.asarray(img_list).squeeze()
        if return_ideal:
            ideal_list = np.asarray(ideal_list).squeeze()
        
        if output == "torch":
            img_list = torch.from_numpy(img_list.astype(np.float32)).type(torch.float)
            if return_ideal:
                ideal_list = torch.from_numpy(ideal_list.astype(np.float32)).type(torch.float)
       
        if return_ideal:
            return img_list,ideal_list
        
        else:
            return img_list



## Both types of simmers used in the original silkscreen paper
class SSPSimmer(ArtpopSimmer):
    "Class to simulate simple ssp"

    def __init__(self, obs_object: SilkScreenObservation):
        super().__init__(obs_object)
        # log Ms, Dist, Z and log age
        self.N_free = 4
        self.param_descrip = ["log Ms/Msun", "D (Mpc)", "Z", "Age (Gyr)"]

    def build_sp(self, x):
        "x is array with logMs,D,Z,logAge"
        logMs, D, Z, Age = x.tolist()
        logAge = np.log10(Age) + 9.0
        return self.build_ssp(logMs, D, Z, logAge)


class DefaultDwarfFixedAgeSimmer(ArtpopSimmer):
    "Class to simulate the default dwarf model with 3 components with fixed ages, with shared metallicity"

    def __init__(self, obs_object: SilkScreenObservation):
        super().__init__(obs_object)

        self.N_free = 6
        self.param_descrip = [
            "D (Mpc)",
            "logMs",
            "Z",
            "F_y",
            "age_y (/100 Myr)",
            "F_m",
        ]

    def build_sp(self, x):
        D, logM, Z, f_y, age_y_norm, f_m = x.tolist()

        # Fixed ages for the old and medium components component
        logAge_y = np.log10(age_y_norm) + 8.0
        logAge_m = np.log10(2.0) + 9.0
        logAge_o = np.log10(12.5) + 9.0

        f_o = 1.0 - f_m - f_y

        ssp_y = self.build_ssp(logM + np.log10(f_y + 1e-6), D, Z, logAge_y)
        ssp_m = self.build_ssp(logM + np.log10(f_m + 1e-6), D, Z, logAge_m)
        ssp_o = self.build_ssp(logM + np.log10(f_o + 1e-6), D, Z, logAge_o)
        sp_comb = ssp_y + ssp_m + ssp_o
        return sp_comb


class NewDwarfFixedAgeSimmer(ArtpopSimmer):
    "Class to simulate the default dwarf model with 3 components with fixed ages, with shared metallicity"

    def __init__(self, obs_object: SilkScreenObservation, age_y_100myr: float = 1., Z_offset: float = 0.):
        super().__init__(obs_object)

        self.age_y_100myr = age_y_100myr
        self.Z_offset = Z_offset
        self.N_free = 5
        self.param_descrip = [
            "D (Mpc)",
            "logMs",
            "Z",
            "F_y",
            "F_m",
        ]

    def build_sp(self, x):
        D, logM, Z, f_y, f_m = x.tolist()

        # Fixed ages for the old, medium, and young components
        logAge_y = np.log10(self.age_y_100myr) + 8.0  # Young component with specific age
        logAge_m = np.log10(1.5) + 9.0  # 1.5 Gyr component
        logAge_o = np.log10(10.0) + 9.0  # 10 Gyr component

        f_o = 1.0 - f_m - f_y

        ssp_y = self.build_ssp(logM + np.log10(f_y + 1e-6), D, Z + self.Z_offset, logAge_y )
        ssp_m = self.build_ssp(logM + np.log10(f_m + 1e-6), D, Z + self.Z_offset, logAge_m )
        ssp_o = self.build_ssp(logM + np.log10(f_o + 1e-6), D, Z, logAge_o)
        sp_comb = ssp_y + ssp_m + ssp_o
        return sp_comb

class ContYoungDwarfSimmer(ArtpopSimmer):
    "Class to simulate the default dwarf model with 3 components with fixed ages, with shared metallicity"

    def __init__(self, obs_object: SilkScreenObservation, age_range_100myr: Iterable = (0.5,2.5), N_cont: int = 12, Z_offset: float = 0.):
        super().__init__(obs_object)

        self.age_range_100myr = age_range_100myr
        self.N_cont = N_cont
        self.age_list = np.linspace(self.age_range_100myr[0], self.age_range_100myr[1], num = N_cont)
        self.Z_offset = Z_offset
        self.N_free = 5
        self.param_descrip = [
            "D (Mpc)",
            "logMs",
            "Z",
            "F_y",
            "F_m",
        ]

    def build_sp(self, x):
        D, logM, Z, f_y, f_m = x.tolist()

        # Fixed ages for the old, medium, and young components
        logAge_m = np.log10(1.5) + 9.0  # 1.5 Gyr component
        logAge_o = np.log10(10.0) + 9.0  # 10 Gyr component

        f_o = 1.0 - f_m - f_y

        ssp_m = self.build_ssp(logM + np.log10(f_m + 1e-7), D, Z + self.Z_offset, logAge_m)
        ssp_o = self.build_ssp(logM + np.log10(f_o + 1e-7), D, Z, logAge_o)

        sp_comb = ssp_m + ssp_o

        f_y_each = f_y/float(self.N_cont)
        for age in self.age_list:
            logAge_cur = np.log10(age) + 8.
            ssp_cur = self.build_ssp(logM + np.log10(f_y_each + 1e-7), D, Z + self.Z_offset, logAge_cur )
            sp_comb = sp_comb + ssp_cur
        return sp_comb

class DwarfFourPopFixedAgeSimmer(ArtpopSimmer):
    "Class to simulate the default dwarf model with 3 components with fixed ages, with shared metallicity"

    def __init__(self, obs_object: SilkScreenObservation):
        super().__init__(obs_object)

        self.N_free = 6
        self.param_descrip = [
            "D (Mpc)",
            "logMs",
            "Z",
            "F_vy"
            "F_y",
            "F_m",
        ]

    def build_sp(self, x):
        D, logM, Z, f_vy,f_y, f_m = x.tolist()

        # Fixed ages for the old, medium, and young components
        logAge_vy = np.log10(.75) + 8.0  #  75 Myr old component
        logAge_y = np.log10(2.5) + 8.0  # 250 Myr old component
        logAge_m = np.log10(1.5) + 9.0  # 1.5 Gyr component
        logAge_o = np.log10(10.0) + 9.0  # 10 Gyr component

        f_o = 1.0 - f_m - f_y - f_vy

        ssp_vy = self.build_ssp(logM + np.log10(f_vy + 1e-6), D, Z, logAge_vy)
        ssp_y = self.build_ssp(logM + np.log10(f_y + 1e-6), D, Z, logAge_y)
        ssp_m = self.build_ssp(logM + np.log10(f_m + 1e-6), D, Z, logAge_m)
        ssp_o = self.build_ssp(logM + np.log10(f_o + 1e-6), D, Z, logAge_o)
        sp_comb = ssp_y + ssp_m + ssp_o + ssp_vy
        return sp_comb

### Other options not used in the paper but allow some more freedom.
class DefaultDwarfThreePopSimmer(ArtpopSimmer):
    "Class to simulate the default dwarf model with 3 components, with shared metallicity"

    def __init__(self, obs_object: SilkScreenObservation):
        super().__init__(obs_object)

        self.N_free = 7
        self.param_descrip = [
            "D (Mpc)",
            "logMs",
            "Z",
            "F_y",
            "Age_Y (Gyr)",
            "F_m",
            "Age_M (Gyr)",
        ]  # , ] (TYPO??)

    def build_sp(self, x):
        D, logM, Z, f_y, age_y, f_m, age_m = x.tolist()

        # Fixed ages for the old component
        logAge_o = 10.08
        logAge_m = np.log10(age_m) + 9.0
        logAge_y = np.log10(age_y) + 9.0

        f_o = 1.0 - f_m - f_y

        ssp_y = self.build_ssp(logM + np.log10(f_y + 1e-6), D, Z, logAge_y)
        ssp_m = self.build_ssp(logM + np.log10(f_m + 1e-6), D, Z, logAge_m)
        ssp_o = self.build_ssp(logM + np.log10(f_o + 1e-6), D, Z, logAge_o)
        sp_comb = ssp_y + ssp_m + ssp_o
        return sp_comb


class DefaultDwarfTwoPopSimmer(ArtpopSimmer):
    "Class to simulate the default dwarf model with 2 components with shared metallicity"

    def __init__(self, obs_object: SilkScreenObservation):
        super().__init__(obs_object)

        self.N_free = 5
        self.param_descrip = ["D (Mpc)", "logMs", "Z", "F_y", "Age_Y (Gyr)"]

    def build_sp(self, x):
        D, logM, Z, f_y, age_y = x.tolist()

        # Fixed ages for the old component
        logAge_o = 10.08  # 12 gyr
        logAge_y = np.log10(age_y) + 9.0

        f_o = 1.0 - f_y

        ssp_y = self.build_ssp(logM + np.log10(f_y + 1e-6), D, Z, logAge_y)
        ssp_o = self.build_ssp(logM + np.log10(f_o + 1e-6), D, Z, logAge_o)
        sp_comb = ssp_y + ssp_o
        return sp_comb

def sigmoid(x):
    return 1./( np.exp(-x) + 1)

class DwarfSigmoidLinearSimmer(ArtpopSimmer):
    "Class to simulate the default dwarf model with 2 components with shared metallicity"
    def __init__(self, obs_object: SilkScreenObservation, time_first = 12, time_last = 0.05, N_ssps = 20):
        super().__init__(obs_object)
        self.time_first = time_first
        self.time_last = time_last
        self.N_ssps = N_ssps
        self.time_bins = np.linspace(self.time_first, self.time_last, num = self.N_ssps
                                     )

        self.N_free = 5
        self.param_descrip = ["D (Mpc)", "logMs", "Z", "SFH_a", "SFH_b"]

    def SFH_func(self, t, a,b, t0 = 8):
        return sigmoid( b - a*(t-t0) )
    
    def build_sp(self, x):
        D, logM, Z, sfh_a, sfh_b = x.tolist()
        
        #Build oldest age component
        F_first = self.SFH_func(self.time_first, sfh_a, sfh_b)
        sp_comb = self.build_ssp(logM + np.log10(F_first + 1e-7), D, Z, np.log10(self.time_first)+9)
        
        #Loop through time bins and build each ssp
        for j in range(1,len(self.time_bins)):
            age = self.time_bins[j]
            age_last = self.time_bins[j-1]
            med_age = (age + age_last)/2
            F_cur = self.SFH_func(age, sfh_a, sfh_b) - self.SFH_func(age_last, sfh_a, sfh_b)
            logAge_cur = np.log10(med_age) + 9.
            ssp_cur = self.build_ssp(logM + np.log10(F_cur + 1e-7), D, Z, logAge_cur )
            sp_comb = sp_comb + ssp_cur
        
        #Final young component is 25 Myr old
        F_cur = 1. - self.SFH_func(age_last, sfh_a, sfh_b)
        logAge_cur = np.log10(0.025) + 9.
        ssp_cur = self.build_ssp(logM + np.log10(F_cur + 1e-7), D, Z, logAge_cur )
        sp_comb = sp_comb + ssp_cur
        
        return sp_comb

def tmax_tau_to_mu_sig(t_max, tau):
    # tau from the FWHM equation: T_FWHM = t_max * 2 * sinh(tau * sqrt(2*ln2))
    # => tau = arcsinh(T_FWHM / (2 * t_max)) / sqrt(2 * ln2)
    sig = np.arcsinh(tau / (2.0 * t_max)) / np.sqrt(2*np.log(2))

    # T0 from the mode equation: t_max = exp(T0 - tau^2)
    # => T0 = ln(t_max) + tau^2
    mu = np.log(t_max) + sig**2
 
    return mu, sig

AGE_UNIVERSE_GYR = cosmo.age(0).value

class LogNormalSimmer(ArtpopSimmer):
    "Class to simulate the default dwarf model with 2 components with shared metallicity"
    def __init__(self, obs_object: SilkScreenObservation, time_first = 13, time_last = 0.05, N_ssps = 20):
        super().__init__(obs_object)
        self.time_first = time_first
        self.time_last = time_last
        self.N_ssps = N_ssps
        self.time_bins = np.linspace(self.time_first, self.time_last, num = self.N_ssps)

        self.N_free = 5
        self.param_descrip = ["D (Mpc)", "logMs", "Z", "T_max", "tau"]

    def SFH_func(self, age, Tmax,tau):
        mu,sig = tmax_tau_to_mu_sig(Tmax, tau)
        t_uni = AGE_UNIVERSE_GYR - age
        x = (np.log(t_uni)-mu)/sig
        return norm().cdf(x)
    
    def build_sp(self, x):
        D, logM, Z, T_max, tau = x.tolist()
        
        #Build oldest age component
        F_first = self.SFH_func(self.time_first, T_max, tau)
        sp_comb = self.build_ssp(logM + np.log10(F_first + 1e-7), D, Z, np.log10(self.time_first)+9)
        
        #Loop through time bins and build each ssp
        for j in range(1,len(self.time_bins)):
            age = self.time_bins[j]
            age_last = self.time_bins[j-1]
            med_age = (age + age_last)/2
            F_cur = self.SFH_func(age, T_max, tau) - self.SFH_func(age_last, T_max, tau)
            logAge_cur = np.log10(med_age) + 9.
            ssp_cur = self.build_ssp(logM + np.log10(F_cur + 1e-7), D, Z, logAge_cur )
            sp_comb = sp_comb + ssp_cur
        
        #Final young component is 25 Myr old
        F_young = 1. - self.SFH_func(self.time_last, T_max, tau)
        age_young = [0.05,0.045,0.04,0.035,0.03,0.025,0.2,0.015,0.01,0.005]
        F_young_each = F_young/len(age_young)
        for age_young_cur in age_young:
            logAge_cur = np.log10(age_young_cur) + 9.
            ssp_cur = self.build_ssp(logM + np.log10(F_young_each + 1e-7), D, Z, logAge_cur )
            sp_comb = sp_comb + ssp_cur
        
        return sp_comb