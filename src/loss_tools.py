"""
Simple numpy tools to play with the curved-sky J3 loss on CAR Q/U maps.

Same operations as losses_car.CarLossJ3 (used in training), but on plain numpy
maps in muK, one function per step, each one timed. Typical use:

    import loss_tools as lt
    ctx = lt.setup()                                   # geometry, spectra, beam, transforms
    Q, U, mask, sigma2 = lt.load_valid_map(ctx, 0)     # one validation map (observed, beamed + noise)
    ell, cl = lt.spectrum(ctx, Q, U, mask)             # pseudo-C_l EE, BB
    t1 = lt.term_data(ctx, Q, U, Qp, Up, mask, sigma2) # likelihood term for a prediction (Qp, Up)
    t2 = lt.term_prior(ctx, Qp, Up)                    # prior term
    lt.print_times(ctx)

Units: the training loader multiplies the maps by map_rescale_factor f and the
loss divides by f^2 (sigma^2 and C_l are in muK^2), so f cancels in both terms:
these functions work directly in muK and give the same numbers as CarLossJ3.
The loader also subtracts the mean of each observed map (subtract_mean below).
"""

import os
import time

import numpy as np
import healpy as hp
import torch
from pixell import enmap, curvedsky


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLUSTER = os.path.join(REPO, 'results', 'task2', 'cluster')       # local copy of the cluster validation set
TAG = 'nCMB_r034_beam_car9.6_cut1120x320_maskSO_inho_hitsSO'


class Context(dict):
    """dict with attribute access: ctx.sht, ctx.mask, ctx.clee, ..."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


# --------------------------------------------------------------------------- #
# set-up and data
# --------------------------------------------------------------------------- #

def setup(lmax=1124, method='2d', nthread=8, fwhm_arcmin=23.0, folder=CLUSTER, tag=TAG):
    """
    Geometry (from the CAR mask FITS), spectra, beam and transform settings.

    method: pixell map2alm/alm2map method. '2d' (default) is the exact transform of
    notebook 02 (pads the cut-out to the full fejer1 sky); training uses 'cyl' (only the
    rows of the cut-out, ~2x faster), which gives the same spectra on these maps.

    Returns ctx with: shape, wcs, mask (CAR, 0/1), sigma2 (CAR, muK^2), clee, clbb,
    bl (B_l up to lmax), lmax, method, nthread, npix, times (dict of accumulated timings).
    """
    mask = enmap.read_map(f'{folder}/aux/mask_car_{tag}.fits')
    sigma2 = enmap.read_map(f'{folder}/aux/variance_car_{tag}.fits')
    cltt, clee, clbb = np.load(f'{folder}/aux/cls_signal_r0.034.npy')
    ctx = Context(shape=mask.shape, wcs=mask.wcs, mask=mask, sigma2=sigma2, clee=clee, clbb=clbb,
                  lmax=lmax, folder=folder, tag=tag,
                  bl=hp.gauss_beam(np.radians(fwhm_arcmin / 60), lmax=lmax),
                  method=method, nthread=nthread,
                  npix=mask.shape[0] * mask.shape[1], times={})
    return ctx


def load_valid_map(ctx, j):
    """Validation map j of the cluster dataset: Q_obs, U_obs [muK], mask, sigma^2 [muK^2] (numpy, ny x nx)."""
    if 'X_valid' not in ctx:
        ctx.X_valid = torch.load(f'{ctx.folder}/data/valid_{ctx.tag}.pt', mmap=True)
    x = ctx.X_valid[j].numpy().astype(np.float64)
    return x[0], x[1], x[2], x[3]


def true_signal(ctx, j, beam=False, lmax_signal=1535):
    """
    True CMB Q, U of validation map j [muK] (CMB seed 1000 + j, as make_dataset), without
    beam (the Wiener-filter target) or with it (beam=True). lmax_signal < 1535 cuts the
    same realization at that multipole.
    """
    seeds = np.load(f'{ctx.folder}/aux/seeds_{ctx.tag}.npz')
    ps = np.zeros((3, 3, 1536))
    ps[1, 1], ps[2, 2] = ctx.clee[:1536], ctx.clbb[:1536]
    ps[:, :, lmax_signal + 1:] = 0
    alm = curvedsky.rand_alm(ps, lmax=1535, seed=int(seeds['cmb_seeds_valid'][j]))
    if beam:
        bl = hp.gauss_beam(np.radians(23.0 / 60), lmax=1535)
        alm[1], alm[2] = curvedsky.almxfl(alm[1], bl), curvedsky.almxfl(alm[2], bl)
    qu = enmap.zeros((2,) + tuple(ctx.shape), ctx.wcs)
    curvedsky.alm2map(alm[1:], qu, spin=2)
    return np.asarray(qu[0]), np.asarray(qu[1])


def subtract_mean(Q, U):
    """Mean subtraction of the training loader (over the whole image, zeros included)."""
    return Q - Q.mean(), U - U.mean()


# --------------------------------------------------------------------------- #
# transforms (timed)
# --------------------------------------------------------------------------- #

def _tic(ctx, name, t0):
    ctx.times.setdefault(name, [0.0, 0])
    ctx.times[name][0] += time.perf_counter() - t0
    ctx.times[name][1] += 1


def qu_to_eb(ctx, Q, U):
    """Q, U map -> a_lm^E, a_lm^B (l <= lmax). This is the operator A of the loss."""
    t0 = time.perf_counter()
    qu = enmap.enmap(np.stack([Q, U]).astype(np.float64), ctx.wcs)
    alm = curvedsky.map2alm(qu, lmax=ctx.lmax, spin=2, method=ctx.method, nthread=ctx.nthread)
    _tic(ctx, 'A: Q,U -> E,B', t0)
    return alm[0], alm[1]


def eb_to_qu(ctx, almE, almB):
    """a_lm^E, a_lm^B -> Q, U on the cut-out. This is the operator Y of the loss."""
    t0 = time.perf_counter()
    qu = enmap.zeros((2,) + tuple(ctx.shape), ctx.wcs)
    curvedsky.alm2map(np.stack([almE, almB]), qu, spin=2, method=ctx.method, nthread=ctx.nthread)
    _tic(ctx, 'Y: E,B -> Q,U', t0)
    return qu[0], qu[1]


def apply_beam(ctx, Q, U):
    """Y B_l A (Q, U): the beam as the loss applies it to the prediction."""
    almE, almB = qu_to_eb(ctx, Q, U)
    return eb_to_qu(ctx, curvedsky.almxfl(almE, ctx.bl), curvedsky.almxfl(almB, ctx.bl))


def bandlimit(ctx, Q, U):
    """Y A (Q, U): the part of a map the loss can see (l <= lmax)."""
    almE, almB = qu_to_eb(ctx, Q, U)
    return eb_to_qu(ctx, almE, almB)


# --------------------------------------------------------------------------- #
# spectrum
# --------------------------------------------------------------------------- #

def spectrum(ctx, Q, U, mask=None):
    """
    Pseudo-C_l of (mask * Q, mask * U): alm2cl / w2, with w2 = sum(mask^2 * pixel area) / 4 pi.
    mask=None uses the whole cut-out. Returns ell, cl with cl[0] = EE, cl[1] = BB (muK^2).
    No mode-coupling correction: compare with theory * B_l^2 + noise, not with C_l itself.
    """
    m = np.ones(ctx.shape) if mask is None else np.asarray(mask)
    almE, almB = qu_to_eb(ctx, m * Q, m * U)
    t0 = time.perf_counter()
    w2 = float(np.sum(m**2 * ctx.mask.pixsizemap()) / (4 * np.pi))
    cl = np.array([curvedsky.alm2cl(almE), curvedsky.alm2cl(almB)]) / w2
    _tic(ctx, 'alm2cl', t0)
    return np.arange(ctx.lmax + 1), cl


def bin_spectrum(ell, cl, edges):
    """Mean of cl in bins [edges[i], edges[i+1])."""
    return (np.array([0.5 * (lo + hi) for lo, hi in zip(edges[:-1], edges[1:])]),
            np.array([[c[(ell >= lo) & (ell < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])] for c in np.atleast_2d(cl)]))


# --------------------------------------------------------------------------- #
# loss terms (same as losses_car.CarLossJ3)
# --------------------------------------------------------------------------- #

def chi2_map(ctx, Qobs, Uobs, Qpred, Upred, mask, sigma2):
    """
    Per-pixel likelihood chi^2: mask * (d - Y B A pred)^2 / sigma^2, for Q and U (2, ny, nx).
    Useful to see where (in the map) the data term is large.
    """
    Qb, Ub = apply_beam(ctx, Qpred, Upred)
    t0 = time.perf_counter()
    inv = np.zeros_like(sigma2)
    ok = (mask > 0) & (sigma2 > 0)
    inv[ok] = 1.0 / sigma2[ok]
    out = np.array([mask * (Qobs - Qb)**2 * inv, mask * (Uobs - Ub)**2 * inv])
    _tic(ctx, 'data term (pixels)', t0)
    return out


def term_data(ctx, Qobs, Uobs, Qpred, Upred, mask, sigma2):
    """Likelihood term of J3: (1/N_pix) sum_i m_i (d_i - (Y B A pred)_i)^2 / sigma_i^2, Q + U."""
    c = chi2_map(ctx, Qobs, Uobs, Qpred, Upred, mask, sigma2)
    return float(c.sum() / ctx.npix)


def prior_per_ell(ctx, Qpred, Upred):
    """sum_m w_m |a_lm|^2 / C_l for each l (l >= 2), E and B: array (2, lmax + 1). Shows which l dominate."""
    almE, almB = qu_to_eb(ctx, Qpred, Upred)
    t0 = time.perf_counter()
    ell = np.arange(ctx.lmax + 1)
    out = np.zeros((2, ctx.lmax + 1))
    for j, (alm, C) in enumerate(((almE, ctx.clee), (almB, ctx.clbb))):
        out[j, 2:] = (curvedsky.alm2cl(alm) * (2 * ell + 1))[2:] / C[2:ctx.lmax + 1]    # sum_m w_m |a_lm|^2 = (2l+1) C_l^hat
    _tic(ctx, 'prior term (alm)', t0)
    return out


def term_prior(ctx, Qpred, Upred, split=False):
    """Prior term of J3: (1/N_pix) sum_{l>=2,m} w_m (|a^E_lm|^2/C^EE_l + |a^B_lm|^2/C^BB_l). split=True: (E, B)."""
    p = prior_per_ell(ctx, Qpred, Upred).sum(axis=1) / ctx.npix
    return (float(p[0]), float(p[1])) if split else float(p.sum())


def loss_terms(ctx, Qobs, Uobs, Qpred, Upred, mask, sigma2):
    """Both terms and J3 = term_data + term_prior, in a dict."""
    t1 = term_data(ctx, Qobs, Uobs, Qpred, Upred, mask, sigma2)
    pE, pB = term_prior(ctx, Qpred, Upred, split=True)
    return dict(data=t1, prior_E=pE, prior_B=pB, prior=pE + pB, J3=t1 + pE + pB)


# --------------------------------------------------------------------------- #
# reference values and timings
# --------------------------------------------------------------------------- #

def expected_values(ctx, mask):
    """
    Expected terms for a perfect prediction: data = 2 N_obs / N_pix (pure noise residual);
    prior (E or B) = f_sky * sum_{l=2}^{lmax} (2l+1) / N_pix for a band-limited field drawn from C_l.
    """
    fsky = float(np.sum(ctx.mask.pixsizemap()) / (4 * np.pi))
    ell = np.arange(2, ctx.lmax + 1)
    return dict(data=2 * float((np.asarray(mask) > 0).sum()) / ctx.npix,
                prior_E=fsky * np.sum(2 * ell + 1) / ctx.npix, prior_B=fsky * np.sum(2 * ell + 1) / ctx.npix)


def print_times(ctx, reset=True):
    """Accumulated wall time per operation since the last reset."""
    for name, (t, n) in ctx.times.items():
        print(f'{name:22s}: {n:4d} calls, {t:7.3f} s total, {t / n:.4f} s per call')
    if reset:
        ctx.times.clear()
