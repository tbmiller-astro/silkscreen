import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
# import h5py
# import asdf
import artpop
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb

import torch #for CPU unpickling solution -- see CPU_Unpickler
import io

## define style choices for plotting
from albumpl.palette import set_default
from albumpl.cmap import LondonCalling, Clampdown
set_default('LondonCalling')
matplotlib.rcParams['font.size'] = 20


class CPU_Unpickler(pickle.Unpickler):
	"""
	Reads in GPU posterior on CPU computer.

	From https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219.

	With all credit to Matthew McDermott (@mmcdermott).

	"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# The specifics of these commands will probably change a bit, since the posterior sample file I'm using is from ~April 2022
## can't install cuda-supported pytorch to look at the posterior distribution with M1 Mac, so not sure if this is something
## we want to be concious of for users, too? The CPU_Unpickler class should fix the issue for now.
## The rest of this should work, though, even if it needs to be cleaned up for use.
class AnalyzePosterior():
	def __init__(self, posterior):
		"""
		Initializes posterior analysis object from posterior output file.

		Parameters:
			posterior (str): Path and name of posterior file to be loaded.
		"""
		self.posterior = posterior

	def load_posterior(self, nsamps = 10**5, nparam = 7):

		"""
		Loads SilkScreen posterior for analysis.

		Parameters:
			nsamps (int): Number of samples to draw from posterior distribution.
			nparam (int): Number of parameters in model -- 7 for DefaultDwarfSimmer, 4 for SSPSimmer, ...

		Returns:
			postdict (dict): Contains sampled posteriors.

		"""
		suff = self.posterior.partition('.')[2]

		if (suff == 'pkl') or (suff == 'pt') or (suff == 'pth'):
			with open(posterior, 'rb') as f:
			    post = CPU_Unpickler(f).load()

		# if suff == 'asdf': #added as template for having additional file types

		# if suff == 'hdf5':

		x_o = torch.ones(1, 3, 205, 205, device = torch.device("cpu"))
		out = post.posterior_estimator.sample_and_log_prob(nsamps, x_o)

		postdict = {}
		for k in range(ndim):
			postdict[str(k)] = [] 
		for i in range(nsamps):
		    out_ = out[0][0][i]
		    for k in range(ndim):
			    postdict[str(k)].append(out_[k].item())

		return postdict


	def infer_vals(self):
		"""
		Returns mean value of posterior distributions.
		"""
		postdict = self.load_posterior()

		postparams = []
		for k in postdict.keys():
			postparams.append(np.mean(postdict[k]))

		return postparams


	def show_posterior(self, savepath = None, **kwargs):
		"""
		Shows posterior distribution.
		
		Parameters:
			savepath (optional, str): If a path is given, posterior corner plot will be saved.
		"""

		postparams = self.infer_vals()
		postdict = self.load_posterior()

		if len(postparams) == 7:
			mosaic = """
					A......
					BC.....
					DEF....
					GHIJ...
					KLMNO..
					PQRSTU.
					VWXYZab
					 """
			paramlabels = ['D (Mpc)', 'logMs','Z','F_y','Age_Y (Gyr)', 'F_m','Age_M (Gyr)']
			#we should probably change these to be consistent with the style in the paper

		if len(postparams) == 4:
			mosaic = """
					A...
					BC..
					DEF.
					GHIJ
					 """
			paramlabels = ['log Ms/Msun','D (Mpc)', 'Z','Age (Gyr)']

		axd = plt.figure(figsize = (10, 10)).subplot_mosaic(mosaic = mosaic, 
			gridspec_kw = {'wspace':0, 'hspace':0}, sharex = 'column')

		axd['A'].hist(postdict[0], bins = 100)
		axd['A'].axis('off')
		axd['A'].axvline(postparams[0])
		axd['B'].hist2d(postdict[0], postdict[1], bins = 100, cmap = LondonCalling('reverse'))
		axd['B'].scatter(postparams[0], postparms[1], marker = 'x')
		axd['C'].hist(postdict[1], bins = 100)
		axd['C'].axis('off')
		axd['C'].axvline(postparams[1])
		axd['D'].hist2d(postdict[0], postdict[2], bins = 100, cmap = LondonCalling('reverse'))
		axd['D'].scatter(postparams[0], postparams[2], marker = 'x')
		axd['E'].hist2d(postdict[1], postdict[2], bins = 100, cmap = LondonCalling('reverse'))
		axd['E'].scatter(postparams[1], postparams[2], marker = 'x')
		axd['F'].hist(postdict[2], bins = 100)
		axd['F'].axis('off')
		axd['G'].hist2d(postdict[0], postdict[3], bins = 100, cmap = LondonCalling('reverse'))
		axd['G'].scatter(postparams[0], postparams[3], marker = 'x')
		axd['H'].hist2d(postdict[1], postdict[3], bins = 100, cmap = LondonCalling('reverse'))
		axd['H'].scatter(postparams[1], postparams[3], marker = 'x')
		axd['I'].hist2d(postdict[2], postdict[3], bins = 100, cmap = LondonCalling('reverse'))
		axd['I'].scatter(postparams[2], postparams[3], marker = 'x')
		axd['J'].hist(postdict[3], bins = 100)
		axd['J'].axis('off')

		if len(postparams) == 4:
			axd['G'].set_xlabel(paramlabels[0])
			axd['H'].set_xlabel(paramlabels[1])
			axd['I'].set_xlabel(paramlabels[2])
			axd['J'].set_xlabel(paramlabels[3])

		if len(postparams) == 7:
			axd['K'].hist2d(postdict[0], postdict[4], bins = 100, cmap = LondonCalling('reverse'))
			axd['K'].scatter(postparams[0], postparams[4], marker = 'x')
			axd['L'].hist2d(postdict[1], postdict[4], bins = 100, cmap = LondonCalling('reverse'))
			axd['L'].scatter(postparams[1], postparams[4], marker = 'x')
			axd['M'].hist2d(postdict[2], postdict[4], bins = 100, cmap = LondonCalling('reverse'))
			axd['M'].scatter(postparams[2], postparams[4], marker = 'x')
			axd['N'].hist2d(postdict[3], postdict[4], bins = 100, cmap = LondonCalling('reverse'))
			axd['N'].scatter(postparams[3], postparams[4], marker = 'x')
			axd['O'].hist(postdict[4], bins = 100)
			axd['O'].axis('off')
			axd['P'].hist2d(postdict[0], postdict[5], bins = 100, cmap = LondonCalling('reverse'))
			axd['P'].scatter(postparams[0], postparams[5], marker = 'x')
			axd['Q'].hist2d(postdict[1], postdict[5], bins = 100, cmap = LondonCalling('reverse'))
			axd['Q'].scatter(postparams[1], postparams[5], marker = 'x')
			axd['R'].hist2d(postdict[2], postdict[5], bins = 100, cmap = LondonCalling('reverse'))
			axd['R'].scatter(postparams[2], postparams[5], marker = 'x')
			axd['S'].hist2d(postdict[3], postdict[5], bins = 100, cmap = LondonCalling('reverse'))
			axd['S'].scatter(postparams[3], postparams[5], marker = 'x')
			axd['T'].hist2d(postdict[4], postdict[5], bins = 100, cmap = LondonCalling('reverse'))
			axd['T'].scatter(postparams[4], postparams[5], marker = 'x')
			axd['U'].hist(postdict[5], bins = 100)
			axd['U'].axis('off')
			axd['V'].hist2d(postdict[0], postdict[6], bins = 100, cmap = LondonCalling('reverse'))
			axd['V'].scatter(postparams[0], postparams[6], marker = 'x')
			axd['W'].hist2d(postdict[1], postdict[6], bins = 100, cmap = LondonCalling('reverse'))
			axd['W'].scatter(postparams[1], postparams[6], marker = 'x')
			axd['X'].hist2d(postdict[2], postdict[6], bins = 100, cmap = LondonCalling('reverse'))
			axd['X'].scatter(postparams[2], postparams[6], marker = 'x')
			axd['Y'].hist2d(postdict[3], postdict[6], bins = 100, cmap = LondonCalling('reverse'))
			axd['Y'].scatter(postparams[3], postparams[6], marker = 'x')
			axd['Z'].hist2d(postdict[4], postdict[6], bins = 100, cmap = LondonCalling('reverse'))
			axd['Z'].scatter(postparams[4], postparams[6], marker = 'x')
			axd['a'].hist2d(postdict[5], postdict[6], bins = 100, cmap = LondonCalling('reverse'))
			axd['a'].scatter(postparams[5], postparams[6], marker = 'x')
			axd['b'].hist(postdict[6], bins = 100)
			axd['b'].axis('off')

			axd['V'].set_xlabel(paramlabels[0])
			axd['W'].set_xlabel(paramlabels[1])
			axd['X'].set_xlabel(paramlabels[2])
			axd['Y'].set_xlabel(paramlabels[3])
			axd['Z'].set_xlabel(paramlabels[4])
			axd['a'].set_xlabel(paramlabels[5])
			axd['b'].set_xlabel(paramlabels[6])

			axd['K'].set_ylabel(paramlabels[4])
			axd['P'].set_ylabel(paramlabels[5])
			axd['V'].set_ylabel(paramlabels[6])


		axd['A'].set_ylabel(paramlabels[0])
		axd['B'].set_ylabel(paramlabels[1])
		axd['D'].set_ylabel(paramlabels[2])
		axd['G'].set_ylabel(paramlabels[3])

		if savepath:
			fig.savefig(savepath, bbox_inches = 'tight', dpi = 300)

		plt.show()


	def show_model(self, photsys, bands = None, zpt = [22.5]*3, psf = None, weights = [1]*3, stretch = 0.1, Q = 5,
					 show_cmd = True, flux_limit = 24, col_errors = 0.5, mag_errors = 0.5, savepath = None, **kwargs):
		"""
		Show output model based on SilkScreen posterior.


		"""
		postparams = self.infer_vals()

		if len(postparams) == 7:

			dist = postparams[0]
			logm = postparams[1]
			feh = postparams[2]
			fy = postparams[3]
			fm = postparams[5]
			fo = 1 - (fy + fm)

			ypop = artpop.MISTSSP(
			    log_age = postparams[4] + 9,       # log of age in years
			    feh = feh,           # metallicity [Fe/H]
			    distance = dist*u.Mpc,
			    phot_system = photsys, # photometric system(s)
			    total_mass = fy*10**logm,
			    imf = 'kroupa'
			)

			mpop = artpop.MISTSSP(
			    log_age = postparams[6] + 9,       # log of age in years
			    feh = feh,           # metallicity [Fe/H]
			    distance = dist*u.Mpc,
			    phot_system = photsys, # photometric system(s)
			    total_mass = fm*10**logm,
			    imf = 'kroupa'
			)

			opop = artpop.MISTSSP(
			    log_age = 10.08,       # log of age in years
			    feh = feh,           # metallicity [Fe/H]
			    distance = dist*u.Mpc,
			    phot_system = photsys, # photometric system(s)
			    total_mass = fo*10**logm,
			    imf = 'kroupa'
			)

			modpop = ypop + mpop + opop

			phot = {}
			for b in bands:
				phot[b] = modpop.star_mags[b]

		if len(postparams) == 4:
			logm = postparams[0]
			dist = postparams[1]
			feh = postparams[2]

			modpop = artpop.MISTSSP(
			    log_age = postparams[3] + 9,       # log of age in years
			    feh = feh,           # metallicity [Fe/H]
			    distance = dist*u.Mpc,
			    phot_system = photsys, # photometric system(s)
			    total_mass = 10**logm,
			    imf = 'kroupa'
			)

			phot = {}
			for b in bands:
				phot[b] = modpop.star_mags[b]

			xy = artpop.sersic_xy( ## ALL OF THESE PHYSICAL THINGS NEED TO BE UPDATED TO DEFAULTS FOR MODEL
			## OR USER INPUTS IF WE'RE GOING THAT DIRECTION
		        num_stars = len(modpop.abs_mag_table),     
		        r_eff = reff*u.kpc,         # effective radius (kpc)
		        n = n,             # Sersic index
		        theta = theta*u.deg,          # position angle (deg)
		        ellip = ellip,         # ellipticity
		        distance = dist*u.Mpc,       # distance to system (Mpc)
		        xy_dim = xy_dim, # xy dimensions of image
		        pixel_scale = pixel_scale
		    )
    
    # SPECIFICALLY NEED TO UPDATE reff, n, theta, ellip, xy_dim, pixel_scale !!
		    pop_full = artpop.source.Source(xy, mags = phot, xy_dim = xy_dim, pixel_scale = pixel_scale)

		    img = {}
		    for i in range(len(bands)):
		        img[bands[i]] = imager.observe(pop_full, bandpass = bands[i], psf = psf[i], zpt = zpt[i]).image

		    pop_rgb = make_lupton_rgb(weights[0]*img[bands[0]], weights[1]*bands[1], weights[2]*bands[2], 
		    	stretch = stretch, Q = Q)


	    if show_cmd:

			col = phot[bands[0]] - phot[bands[2]] #assumes bluest - reddest
			mag = phot[bands[2]]

			fig, (ax1, ax1) = plt.subplots(1, 2, figsize = (10, 5))

			if col_errors:
				col = col + np.random.normal(scale = col_errors, size = col.shape)
			if mag_errors:
				mag = mag + np.random.normal(scale = mag_errors, size = mag.shape)

			pop_cmd = {'col':col, 'mag':mag}

			fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
			if flux_limit:
				ax2.scatter(col[mag <= flux_limit], mag[mag <= flux_limit])
				ax2.set_xlabel('%s - %s'%(bands[0], bands[2]))
				ax2.set_ylabel(bands[2])
				ax2.invert_yaxis()

			pop = (pop_rgb, pop_cmd)

		if not show_cmd:
			fig, ax1 = plt.subplots(1, 1, figsize = (5, 5))

			pop = pop_rgb

		ax1.imshow(pop_rgb)
		ax1.axis('off')

		if savepath:
			fig.savefig(savepath, bbox_inches = 'tight', dpi = 300)

		plt.show()

		return pop



### This is an idea for later potentially -- if we venture into training on space-based photometry, too ###
### would make for potentially easier comparison with *all* available data even in semi-resolved regime ###
### The RGB comparison might be useful now, though, too
class DirectCompareData():
	def __init__(self, rgb, cmd = None):
		"""
		Initializes data comparison object from observations and SilkScreen model output.

		Parameters:
			rgb: make_lupton_rgb() from real data
			cmd (optional): database/table/dictionary where indices correspond to colors, color_err, mags, mag_err
		"""
		self.rgb = rgb
		if cmd:
			self.cmd = cmd


	def perturb_mags(modelcmd):
	    """
	    Make model CMD more realistic by 1. matching the limiting magnitude and 2. perturbing ArtPop stellar magnitudes by the errors on the input CMD.
	    
	    PARAMETERS:
	        modelcmd (dict): output from AnalyzePosterior().show_model()
	    
	    
	    RETURNS: 
	        color (perturbed model colors, list), color errors, mag (perturbed model magnitudes, list) , mag errors
	    """

	    cmd = self.cmd
	    input_mags = cmd[2]
	    input_mag_err = cmd[3]

	    mag = np.array(modelcmd['mag'])
	    col = np.array(modelcmd['col'])

	    magcut = mag[mag <= (max(input_mags) + max(input_mag_err))]
	    colcut = col[mag <= (max(input_mags) + max(input_mag_err))]

	    pop_err = [] 
	    pop_color_err = []
	    new_pop = []
	    new_popcolor = []

	    for i in range(len(pop)):
	        pop_err_mod = input_mag_err[np.argmin(abs(input_mags - pop[i]))]
	        pop_err.append(pop_err_mod)
	        new_pop.append(pop[i] + gauss(0, pop_err_mod**2))

	        pop_color_err_mod = input_mag_err[np.argmin(abs(input_mags - pop[i]))]
	        pop_color_err.append(pop_color_err_mod)
	        new_popcolor.append(pop_color[i] + gauss(0, pop_color_err_mod**2))

	    modcmd = {'col':new_popcolor, 'colerr':pop_color_err, 'mag':new_pop, 'magerr':pop_err}
	        
	    return modcmd
	    

	def compare_model(self, posterior, bands = None, show_cmd = True, flux_limit = 24, col_errors = 0.5, mag_errors = 0.5, 
		savepath = None):
		"""

		"""

		pop = AnalyzePosterior(posterior).show_model(bands, show_cmd, flux_limit, col_errors, mag_errors)

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
		ax1.imshow(self.rgb, origin = 'lower')
		ax1.axis('off')
		ax1.set_title('Observation')

		ax2.imshow(pop[0], origin = 'lower')
		ax2.axis('off')
		ax2.axis('Model')

		if savepath:
			fig.savefig(savepath, bbox_inches = 'tight', dpi = 300)

		plt.show()


	def compare_cmd(self, col_lims = None, mag_lims = None, bins = [20]*2, sub = True, 
		labels = ['color', 'mag'], savepath = None):
		"""
		Shows normalized 2D histograms for the data and mock CMDs + (optionally) residuals.

		Parameters:
			col_lims (optional, list): Lower and upper limit on color for CMD histograms.
			mag_lims (optional, list): Lower and upper limit on magnitude for CMD histograms.
			bins (optional, list): Number of bins in color and magnitude, respectively.
			sub (bool): If True, will show residual histogram of data and model.
			labels (optional, list): List of strings, x- and y-axis labels. Swap for bands!
			savepath (str): If savepath will save comparison figure.

		"""
		cmd = self.cmd
		modpop = self.compare_model()
		modcmd = self.perturb_mags(modpop[1])

		if not sub:
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10), sharey = True)

		if sub:
			fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (32, 10), sharey = True)

		h, xe, ye, _ = ax1.hist2d(cmd[0], cmd[2], bins = bins, cmap = LondonCalling('reverse'), weights = 1/len(cmd[0]))
		ax1.set_title('Observation')
		ax1.invert_yaxis()
		hm, xem, yem, _ = ax2.hist2d(modcmd['col'], modcmd['mag'], bins = bins, 
			cmap = LondonCalling('reverse'), weights = 1/len(new_popcolor[0]))
		ax2.set_title('Model')

		if col_lims:
			ax1.set_xlim(col_lims)
			ax2.set_xlim(col_lims)

		if mag_lims:
			ax1.set_ylim(mag_lims)
			ax2.set_ylim(mag_lims)

		ax1.set_xlabel(labels[0])
		ax2.set_xlabel(labels[0])

		ax1.set_ylabel(labels[1])
		ax2.set_ylabel(labels[1])

		if sub:
			cbar = ax3.pcolormesh((h - hm).T, cmap = Clampdown())
			ax3.set_title('Observation - Model')
			plt.colorbar(cbar, ax = ax3, pad = 0.1, vmin = -1, vmax = 1)

			if col_lims:
				ax3.set_xlim(col_lims)
			if mag_lims:
				ax3.set_ylim(mag_lims)

			ax3.set_xlabel(labels[0])
			ax3.set_ylabel(labels[1])

		if savepath:
			fig.savefig(savepath, bbox_inches = 'tight', dpi = 300)
