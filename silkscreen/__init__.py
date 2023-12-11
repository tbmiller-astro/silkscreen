""" Silkscreen """
__version__ = "0.01"
import artpop.log
artpop.log.logger.disabled = True
from .simmer import DefaultDwarfTwoPopSimmer,  DefaultDwarfThreePopSimmer, SSPSimmer
from .fitter import fit_silkscreen_model
from .observation import SilkScreenObservation
from .priors import get_SSP_prior, get_default_dwarf_3pop_prior, get_default_dwarf_2pop_prior

