from typing import Iterable, Optional

import numpy as np
import torch
from sbi.utils import BoxUniform, MultipleIndependent, process_prior
from sbi.utils.user_input_checks_utils import CustomPriorWrapper, MultipleIndependent
from scipy.special import erf
from scipy.stats import truncnorm
from torch import tensor
from torch.distributions import Uniform


class MZRPrior:
    def __init__(self, logM_bounds, frac_expand=1.5, device="cpu"):
        self.logM_bounds = logM_bounds
        self.logM_dist = build_uniform_dist(logM_bounds, device)
        self.MZR_sig = 0.17 * frac_expand
        self.Z_bounds = [-2.25, 0.5]
        self.device = device

    # def MZR(self, logM): ## COMMENTED OUT AS A SUGGESTED CHANGE

    #     if logM <= 8.7:
    #         feh = -1.69 + 0.3*(logM - 6.) #Kirby+2013 https://ui.adsabs.harvard.edu/abs/2013ApJ...779..102K/abstract
    #         #scatter is ~0.2 dex

    #     else:
    #         logmass_ = [8.91, 9.11, 9.31, 9.51, 9.72, 9.91, 10.11, 10.31, 10.51, 10.72, 10.91, 11.11, 11.31, 11.51, 11.72, 11.91]
    #         feh_ = [-0.60, -0.61, -0.65, -0.61, -0.52, -0.41, -0.23, -0.11, -0.01, 0.04, 0.07, 0.1, 0.12, 0.13, 0.14, 0.15]
    #         mzr_fit = np.polyfit(logmass_, feh_, 6)
    #         feh = np.poly1d(mzr_fit)(logM) #Gallazzi+2005 https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G/abstract
    #         #scatter is ~0.3 dex

    #     return feh

    @staticmethod
    def KirbyMZR(logM):
        return -1.69 + 0.3 * (logM - 6.0)

    def _log_prob_Z(self, Z, logM):
        Z_mean = self.KirbyMZR(logM).cpu().numpy()
        a = (self.Z_bounds[0] - Z_mean) / self.MZR_sig
        b = (self.Z_bounds[1] - Z_mean) / self.MZR_sig
        return torch.Tensor(truncnorm.logpdf(Z.cpu(), a, b, Z_mean, self.MZR_sig)).to(
            self.device
        )

    def _sample_Z(self, logM, sample_shape):
        Z_mean = self.KirbyMZR(logM.cpu().view(sample_shape)).cpu()
        a = (self.Z_bounds[0] - Z_mean) / self.MZR_sig
        b = (self.Z_bounds[1] - Z_mean) / self.MZR_sig
        Z_samps = truncnorm.ppf(
            np.random.uniform(size=sample_shape), a, b, Z_mean, self.MZR_sig
        )
        if not isinstance(Z_samps, Iterable):
            return torch.tensor(Z_samps, dtype=torch.float32).to(self.device)
        return torch.Tensor(Z_samps).to(self.device)

    def log_prob(self, x):
        log_prob = self.logM_dist.log_prob(x[:, 0])
        log_prob += self._log_prob_Z(x[:, 1], x[:, 0])
        return log_prob

    def sample(self, sample_shape=torch.Size([])):
        logM_samps = self.logM_dist.sample(sample_shape=sample_shape).view(sample_shape)
        Z_samps = self._sample_Z(logM_samps, sample_shape)
        return torch.stack([logM_samps, Z_samps]).T


class MyTruncNorm:
    def __init__(self, loc, sig, bounds, device="cpu"):
        self.loc = loc
        self.sig = sig
        self.a = (bounds[0] - self.loc) / self.sig
        self.b = (bounds[1] - self.loc) / self.sig
        self.Z = 0.5 * (erf(self.b / np.sqrt(2)) - erf(self.a / np.sqrt(2)))
        self.phi_a = 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * self.a**2)
        self.phi_b = 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * self.b**2)
        self.device = device

    @property
    def mean(self):
        return self.loc + (self.phi_a - self.phi_b) / self.Z * self.sig

    @property
    def variance(self):
        var_frac = (
            1.0
            - (self.b * self.phi_b - self.a * self.phi_a) / self.Z
            - (self.phi_a - self.phi_b) ** 2 / self.Z**2
        )
        return self.sig**2 * var_frac

    def log_prob(self, x):
        x_cpu = x.cpu().numpy()
        return torch.Tensor(
            truncnorm.logpdf(x_cpu, self.a, self.b, self.loc, self.sig)
        ).to(self.device)

    def sample(self, sample_shape=torch.Size([])):
        samps = truncnorm.ppf(
            np.random.uniform(size=sample_shape), self.a, self.b, self.loc, self.sig
        )
        if not isinstance(samps, Iterable):
            return torch.tensor(
                [
                    samps,
                ],
                dtype=torch.float32,
            ).to(self.device)
        return torch.tensor(samps).to(self.device).reshape(-1, 1)


def build_uniform_dist(bounds, device):
    return Uniform(
        tensor([bounds[0]], dtype=torch.float32).to(device),
        tensor([bounds[1]], dtype=torch.float32).to(device),
    )


def build_truncnorm_dist(loc, scale, bounds, device):
    custom_dist = MyTruncNorm(loc, scale, bounds, device=device)

    lb = torch.tensor([bounds[0]]).to(device)
    ub = torch.tensor([bounds[1]]).to(device)

    return CustomPriorWrapper(
        custom_dist, event_shape=torch.Size([1]), lower_bound=lb, upper_bound=ub
    )


def build_mzr_dist(logMs_range, device):
    custom_dist = MZRPrior(logMs_range, device=device)
    lb = torch.tensor([logMs_range[0], -2.25]).to(device)
    ub = torch.tensor([logMs_range[1], 0.5]).to(device)
    return CustomPriorWrapper(
        custom_dist, event_shape=torch.Size([2]), lower_bound=lb, upper_bound=ub
    )


def get_default_dwarf_fixed_age_prior(
    D_range: Iterable,
    logMs_range: Iterable,
    Ayoung_range: Iterable = [0.5, 3],
    device: Optional[str] = "cpu",
) -> torch.distributions.distribution.Distribution:
    D_dist = build_uniform_dist(D_range, device)
    M_and_Z_dist = build_mzr_dist(logMs_range, device)
    fy_dist = build_truncnorm_dist(0, 0.05, [0.0, 0.2], device)
    ay_n_dist = build_uniform_dist(Ayoung_range, device)

    fm_dist = build_truncnorm_dist(0.4, 0.2, [0.0, 0.8], device)
    prior = MultipleIndependent([D_dist, M_and_Z_dist, fy_dist, ay_n_dist, fm_dist])

    return prior


def get_new_dwarf_fixed_age_prior(
    D_range: Iterable,
    logMs_range: Iterable,
    Fy_range: Iterable = [0.0, 0.05],
    Fm_range: Iterable = [0.15, 0.2],
    device: Optional[str] = "cpu",
) -> torch.distributions.distribution.Distribution:
    D_dist = build_uniform_dist(D_range, device)
    M_and_Z_dist = build_mzr_dist(logMs_range, device)
    fy_dist = build_uniform_dist(
        Fy_range, device
    )  # sample f_y uniform between 0 and 5% of mass
    fm_dist = build_uniform_dist(
        Fm_range, device
    )  # sample f_m uniform between 10 and 20%
    # F_old will be 1 - f_y - f_m
    # no longer sample Ay; all ages will be fixed

    prior = MultipleIndependent([D_dist, M_and_Z_dist, fy_dist, fm_dist])

    return prior

def get_dwarf_four_pop_fixed_age_prior(
    D_range: Iterable,
    logMs_range: Iterable,
    Fvy_range: Iterable = [0.0, 0.05],
    Fy_range: Iterable = [0.0, 0.05],
    Fm_range: Iterable = [0.15, 0.2],
    device: Optional[str] = "cpu",
) -> torch.distributions.distribution.Distribution:
    D_dist = build_uniform_dist(D_range, device)
    M_and_Z_dist = build_mzr_dist(logMs_range, device)
    fvy_dist = build_uniform_dist(
        Fvy_range, device
    )  # sample f_y uniform between 0 and 5% of mass
    fy_dist = build_uniform_dist(
        Fy_range, device
    )  # sample f_y uniform between 0 and 5% of mass
    fm_dist = build_uniform_dist(
        Fm_range, device
    )  # sample f_m uniform between 10 and 20%
    # F_old will be 1 - f_y - f_m
    # no longer sample Ay; all ages will be fixed

    prior = MultipleIndependent([D_dist, M_and_Z_dist, fvy_dist, fy_dist, fm_dist])

    return prior

def get_SSP_prior(
    D_range: Iterable,
    logMs_range: Iterable,
    Age_range: Optional[Iterable] = [0.1, 12],
    Z_range: Optional[Iterable] = [-2.25, 0.25],
    device: Optional[str] = "cpu",
) -> torch.distributions.distribution.Distribution:
    unif_bounds_tensor = torch.tensor([D_range, logMs_range, Age_range, Z_range]).to(
        device
    )
    unif_dist = BoxUniform(
        unif_bounds_tensor[:, 0], unif_bounds_tensor[:, 1], device=device
    )
    return unif_dist


##? Attempts at doing MZR prior but maybe don't need?
class MSSP_Prior:
    def __init__(self, Dlims, Mlims, logAgelims, expand_fac=1.0, device="cpu"):
        self.Dlims = torch.tensor(Dlims)
        self.D_dist = BoxUniform(
            low=[
                Dlims[0],
            ],
            high=[
                Dlims[1],
            ],
        )
        self.Mlims = torch.tensor(Mlims)
        self.logAgelims = torch.tensor(logAgelims)
        self.M_dist_list = []
        self.A_dist_list = []

        if self.Mlims.ndim > 1:
            self.n_pop = self.Mlims.shape[0]
            assert self.Mlims.shape[0] == self.logAgelims.shape[0]
            for i in range(self.n_pop):
                self.M_dist_list.append(
                    BoxUniform(
                        [
                            self.Mlims[i, 0],
                        ],
                        [
                            self.Mlims[i, 1],
                        ],
                    )
                )
                self.A_dist_list.append(
                    BoxUniform(
                        [
                            self.logAgelims[i, 0],
                        ],
                        [
                            self.logAgelims[i, 1],
                        ],
                    )
                )
        else:
            self.n_pop = 1
            assert self.logAgelims.ndim == 1
            self.M_dist_list.append(
                BoxUniform(
                    [
                        self.Mlims[0],
                    ],
                    [
                        self.Mlims[1],
                    ],
                )
            )
            self.A_dist_list.append(
                BoxUniform(
                    [
                        self.logAgelims[0],
                    ],
                    [
                        self.logAgelims[1],
                    ],
                )
            )

        self.MZR_sig = 0.17 * expand_fac
        self.Z_min = -2.25
        self.Z_max = 0.25
        self.device = device

    def MZR(self, logM):
        return -1.69 + 0.3 * (logM - 6.0)

    def sample_Z_SSP(self, M_tot, sample_shape):
        Z_mean = self.MZR(M_tot)
        a = (self.Z_min - Z_mean) / self.MZR_sig
        b = (self.Z_max - Z_mean) / self.MZR_sig

        Z_samps = truncnorm.ppf(
            np.random.uniform(size=sample_shape), a, b, Z_mean, self.MZR_sig
        )
        return Z_samps

    def log_prob_Z_SSP(self, Z, M_tot):
        Z_mean = self.MZR(M_tot)
        a = (self.Z_min - Z_mean) / self.MZR_sig
        b = (self.Z_max - Z_mean) / self.MZR_sig
        return truncnorm.logpdf(Z, a, b, Z_mean, self.MZR_sig)

    def sample(self, sample_shape=torch.Size([])):
        M_samps = []
        A_samps = []
        for i in range(self.n_pop):
            M_samps.append(self.M_dist_list[i].sample(sample_shape))
            A_samps.append(self.A_dist_list[i].sample(sample_shape))

        M_samps = torch.stack(M_samps)
        A_samps = torch.stack(A_samps)
        M_tot = torch.log10((10 ** M_samps.squeeze()).sum(axis=0))

        Z_samps = []
        for i in range(self.n_pop):
            Z_samps.append(
                torch.Tensor(np.array(self.sample_Z_SSP(M_tot, sample_shape)))
            )

        Z_samps = torch.stack(Z_samps)

        samps = []
        samps.append(self.D_dist.sample(sample_shape))
        for i in range(self.n_pop):
            samps.append(M_samps[i])
            samps.append(Z_samps[i].view(-1, *sample_shape).T)
            samps.append(A_samps[i])

        return torch.stack(samps).to(torch.float).T[0].to(self.device)

    def log_prob(self, values):
        values = values.to("cpu")

        log_prob = torch.zeros(values.shape[0])

        log_prob += self.D_dist.log_prob(values[:, 0])

        M_tot = (10 ** values[:, 1::3]).sum(axis=1).log10()

        for i in range(self.n_pop):
            log_prob += self.M_dist_list[i].log_prob(values[:, 3 * i + 1])
            log_prob += self.log_prob_Z_SSP(values[:, 3 * i + 2], M_tot)
            log_prob += self.A_dist_list[i].log_prob(values[:, 3 * i + 3])

        return log_prob.to(self.device)


def _get_mssp_prior_old(Dlims, Mlims, logAgelims, device="cpu"):
    custom_prior = MSSP_Prior(Dlims, Mlims, logAgelims, device=device)

    lower_bounds = []
    upper_bounds = []

    lower_bounds.append(Dlims[0])
    upper_bounds.append(Dlims[1])

    if custom_prior.n_pop == 1:
        lower_bounds.append(Mlims[0])
        upper_bounds.append(Mlims[1])

        lower_bounds.append(-2.25)
        upper_bounds.append(0.25)

        lower_bounds.append(logAgelims[0])
        upper_bounds.append(logAgelims[1])

    else:
        for i in range(custom_prior.n_pop):
            lower_bounds.append(Mlims[i][0])
            upper_bounds.append(Mlims[i][1])

            lower_bounds.append(-2.25)
            upper_bounds.append(0.25)

            lower_bounds.append(logAgelims[i][0])
            upper_bounds.append(logAgelims[i][1])

    prior, _, _ = process_prior(
        custom_prior,
        custom_prior_wrapper_kwargs=dict(
            lower_bound=torch.Tensor(lower_bounds).to(device),
            upper_bound=torch.Tensor(upper_bounds).to(device),
        ),
    )

    return prior


class Default_Prior:
    def __init__(
        self,
        Dlims,
        Mlims,
        MZR_expand_fac=1.0,
        f_Y_max=0.1,
        f_Y_sig=0.02,
        f_M_max=0.75,
        f_M_mean=0.2,
        f_M_sig=0.2,
        device="cpu",
    ):
        self.Dlims = torch.tensor(Dlims).to("cpu")
        self.D_dist = BoxUniform(
            low=[
                Dlims[0],
            ],
            high=[
                Dlims[1],
            ],
            device="cpu",
        )
        self.Mlims = torch.tensor(Mlims)
        self.M_dist = BoxUniform(
            low=[
                Mlims[0],
            ],
            high=[
                Mlims[1],
            ],
            device="cpu",
        )

        self.f_Y_max = f_Y_max
        self.f_Y_sig = f_Y_sig

        self.f_M_max = f_M_max
        self.f_M_sig = f_M_sig
        self.f_M_mean = f_M_mean

        assert self.f_M_max + self.f_Y_max < 1.0

        self.age_M_dist = BoxUniform(
            low=[
                8.5,
            ],
            high=[
                9.5,
            ],
            device="cpu",
        )

        self.MZR_sig = 0.17 * MZR_expand_fac
        self.Z_min = -2.25
        self.Z_max = 0.25
        self.device = device

    def MZR(self, logM):
        # Kirby+13 MZR
        return -1.69 + 0.3 * (logM - 6.0)

    def sample_Z_SSP(self, M_tot, sample_shape):
        Z_mean = self.MZR(M_tot).to("cpu").numpy()
        a = (self.Z_min - Z_mean) / self.MZR_sig
        b = (self.Z_max - Z_mean) / self.MZR_sig
        Z_samps = truncnorm.ppf(
            np.random.uniform(size=sample_shape), a, b, Z_mean, self.MZR_sig
        )
        if not isinstance(Z_samps, Iterable):
            Z_samps = [
                Z_samps,
            ]
        return torch.Tensor(Z_samps)

    def log_prob_Z_SSP(self, Z, M_tot):
        Z_mean = self.MZR(M_tot).to("cpu").numpy()
        a = (self.Z_min - Z_mean) / self.MZR_sig
        b = (self.Z_max - Z_mean) / self.MZR_sig
        return torch.Tensor(truncnorm.logpdf(Z, a, b, Z_mean, self.MZR_sig))

    def sample_f_M(self, sample_shape):
        a = (0 - self.f_M_mean) / self.f_M_sig
        b = (self.f_M_max - self.f_M_mean) / self.f_M_sig
        f_M_samps = truncnorm.ppf(
            np.random.uniform(size=sample_shape), a, b, self.f_M_mean, self.f_M_sig
        )
        if not isinstance(f_M_samps, Iterable):
            f_M_samps = [
                f_M_samps,
            ]
        return torch.Tensor(f_M_samps)

    def log_prob_f_M(self, f_M):
        a = (0 - self.f_M_mean) / self.f_M_sig
        b = (self.f_M_max - self.f_M_mean) / self.f_M_sig
        return torch.Tensor(truncnorm.logpdf(f_M, a, b, self.f_M_mean, self.f_M_sig))

    def sample_f_Y(self, sample_shape):
        a = 0
        b = (self.f_Y_max) / self.f_Y_sig
        f_Y_samps = truncnorm.ppf(
            np.random.uniform(size=sample_shape), a, b, 0, self.f_Y_sig
        )
        if not isinstance(f_Y_samps, Iterable):
            f_Y_samps = [
                f_Y_samps,
            ]
        return torch.Tensor(f_Y_samps)

    def log_prob_f_Y(self, f_Y):
        a = 0
        b = (self.f_Y_max) / self.f_Y_sig
        return torch.Tensor(truncnorm.logpdf(f_Y, a, b, 0, self.f_Y_sig))

    def sample(self, sample_shape=torch.Size([])):
        samps = []
        samps.append(
            self.D_dist.sample(sample_shape).view(sample_shape).to(self.device)
        )
        samps.append(
            self.M_dist.sample(sample_shape).view(sample_shape).to(self.device)
        )

        samps.append(self.sample_f_Y(sample_shape).view(sample_shape).to(self.device))
        samps.append(self.sample_f_M(sample_shape).view(sample_shape).to(self.device))

        samps.append(
            self.age_M_dist.sample(sample_shape).view(sample_shape).to(self.device)
        )

        samps.append(
            self.sample_Z_SSP(samps[1], sample_shape).view(sample_shape).to(self.device)
        )

        return torch.stack(samps).to(torch.float).T

    def log_prob(self, values):
        if values.ndim == 1:
            values = values.view(-1, values.shape[0])
        values = values.to("cpu")
        log_prob = torch.zeros(values.shape[0]).to(self.device)

        log_prob += self.D_dist.log_prob(values[:, 0]).to(self.device)
        log_prob += self.M_dist.log_prob(values[:, 1]).to(self.device)
        log_prob += self.log_prob_f_Y(values[:, 2]).to(self.device)
        log_prob += self.log_prob_f_M(values[:, 3]).to(self.device)
        log_prob += self.age_M_dist.log_prob(values[:, 4]).to(self.device)
        log_prob += self.log_prob_Z_SSP(values[:, 5], values[:, 1]).to(self.device)

        return log_prob


def _get_default_prior_old(
    Dlims,
    Mlims,
    MZR_expand_fac=1.0,
    f_Y_max=0.1,
    f_Y_sig=0.02,
    f_M_max=0.75,
    f_M_mean=0.2,
    f_M_sig=0.2,
    device="cpu",
):
    custom_prior = Default_Prior(
        Dlims,
        Mlims,
        MZR_expand_fac=MZR_expand_fac,
        f_Y_max=f_Y_max,
        f_Y_sig=f_Y_sig,
        f_M_max=f_M_max,
        f_M_mean=f_M_mean,
        f_M_sig=f_M_sig,
        device=device,
    )

    lower_bounds = []
    upper_bounds = []

    # D
    lower_bounds.append(Dlims[0])
    upper_bounds.append(Dlims[1])

    # logM
    lower_bounds.append(Mlims[0])
    upper_bounds.append(Mlims[1])

    # f_Y
    lower_bounds.append(0)
    upper_bounds.append(f_Y_max)

    # f_M
    lower_bounds.append(0)
    upper_bounds.append(f_M_max)

    # log_Age_M
    lower_bounds.append(
        float(custom_prior.age_M_dist.support.base_constraint.lower_bound[0])
    )
    upper_bounds.append(
        float(custom_prior.age_M_dist.support.base_constraint.upper_bound[0])
    )

    # Z
    lower_bounds.append(-2.25)
    upper_bounds.append(0.25)

    prior, _, _ = process_prior(
        custom_prior,
        custom_prior_wrapper_kwargs=dict(
            lower_bound=torch.Tensor(lower_bounds).to(device),
            upper_bound=torch.Tensor(upper_bounds).to(device),
        ),
    )

    return prior


class Default2PopPrior:
    def __init__(
        self, Dlims, Mlims, MZR_expand_fac=1.0, f_Y_max=1.0, f_Y_sig=0.2, device="cpu"
    ):
        self.Dlims = torch.tensor(Dlims)
        self.D_dist = BoxUniform(
            low=[
                Dlims[0],
            ],
            high=[
                Dlims[1],
            ],
        )
        self.Mlims = torch.tensor(Mlims)
        self.M_dist = BoxUniform(
            low=[
                Mlims[0],
            ],
            high=[
                Mlims[1],
            ],
        )

        self.f_Y_max = f_Y_max
        self.f_Y_sig = f_Y_sig
        self.f_M_dist = BoxUniform(
            low=[
                0,
            ],
            high=[
                1,
            ],
        )

        self.age_Y_dist = BoxUniform(
            low=[
                8,
            ],
            high=[
                9.5,
            ],
        )

        self.MZR_sig = 0.17 * MZR_expand_fac
        self.Z_min = -2.25
        self.Z_max = 0.25
        self.device = device

    def MZR(self, logM):
        # Kirby+13 MZR
        return -1.69 + 0.3 * (logM - 6.0)

    def sample_Z_SSP(self, M_tot, sample_shape):
        Z_mean = self.MZR(M_tot).numpy()
        a = (self.Z_min - Z_mean) / self.MZR_sig
        b = (self.Z_max - Z_mean) / self.MZR_sig
        Z_samps = truncnorm.ppf(
            np.random.uniform(size=sample_shape), a, b, Z_mean, self.MZR_sig
        )
        if not isinstance(Z_samps, Iterable):
            Z_samps = [
                Z_samps,
            ]
        return torch.Tensor(Z_samps)

    def log_prob_Z_SSP(self, Z, M_tot):
        Z_mean = self.MZR(M_tot).numpy()
        a = (self.Z_min - Z_mean) / self.MZR_sig
        b = (self.Z_max - Z_mean) / self.MZR_sig
        return truncnorm.logpdf(Z, a, b, Z_mean, self.MZR_sig)

    def sample_f_Y(self, sample_shape):
        a = 0
        b = (self.f_Y_max) / self.f_Y_sig
        f_Y_samps = truncnorm.ppf(
            np.random.uniform(size=sample_shape), a, b, 0, self.f_Y_sig
        )
        if not isinstance(f_Y_samps, Iterable):
            f_Y_samps = [
                f_Y_samps,
            ]
        return torch.Tensor(f_Y_samps)

    def log_prob_f_Y(self, f_Y):
        a = 0
        b = (self.f_Y_max) / self.f_Y_sig
        return truncnorm.logpdf(f_Y, a, b, 0, self.f_Y_sig)

    def sample(self, sample_shape=torch.Size([])):
        samps = []
        samps.append(self.D_dist.sample(sample_shape).view(sample_shape))
        samps.append(self.M_dist.sample(sample_shape).view(sample_shape))

        samps.append(self.sample_f_Y(sample_shape).view(sample_shape))

        samps.append(self.age_Y_dist.sample(sample_shape).view(sample_shape))

        samps.append(self.sample_Z_SSP(samps[1], sample_shape).view(sample_shape))

        return torch.stack(samps).to(torch.float).T.to(self.device)

    def log_prob(self, values):
        values = values.to("cpu")
        if values.ndim == 1:
            values = values.view(-1, values.shape[0])

        log_prob = torch.zeros(values.shape[0])

        log_prob += self.D_dist.log_prob(values[:, 0])
        log_prob += self.M_dist.log_prob(values[:, 1])

        log_prob += self.log_prob_f_Y(values[:, 2])

        log_prob += self.age_Y_dist.log_prob(values[:, 3])

        log_prob += self.log_prob_Z_SSP(values[:, 4], values[:, 1])

        return log_prob.to(self.device)


def _get_default_2pop_prior_old(
    Dlims, Mlims, MZR_expand_fac=1.0, f_Y_max=1.0, f_Y_sig=0.2, device="cpu"
):
    custom_prior = Default2PopPrior(
        Dlims,
        Mlims,
        MZR_expand_fac=MZR_expand_fac,
        f_Y_max=f_Y_max,
        f_Y_sig=f_Y_sig,
        device=device,
    )

    lower_bounds = []
    upper_bounds = []

    # D
    lower_bounds.append(Dlims[0])
    upper_bounds.append(Dlims[1])

    # logM
    lower_bounds.append(Mlims[0])
    upper_bounds.append(Mlims[1])

    # f_Y
    lower_bounds.append(0)
    upper_bounds.append(f_Y_max)

    # log_age_Y
    lower_bounds.append(
        float(custom_prior.age_Y_dist.support.base_constraint.lower_bound[0])
    )
    upper_bounds.append(
        float(custom_prior.age_Y_dist.support.base_constraint.upper_bound[0])
    )

    # Z
    lower_bounds.append(-2.25)
    upper_bounds.append(0.25)

    prior, _, _ = process_prior(
        custom_prior,
        custom_prior_wrapper_kwargs=dict(
            lower_bound=torch.Tensor(lower_bounds).to(device),
            upper_bound=torch.Tensor(upper_bounds).to(device),
        ),
    )

    return prior


import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(
            batch_shape, validate_args=validate_args
        )
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any(
            (self.a >= self.b)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


### Taken from https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
### Tried to Install from github but it didn't work, not sure how to give propr credit
class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return (
            super(TruncatedNormal, self).log_prob(self._to_std_rv(value))
            - self._log_scale
        )
