"""
Power-spectrum and response tools shared by the three pipelines.
"""

import numpy as np
from pixell import enmap, curvedsky


def bin_cl(cl, bins):
    """Unweighted mean of cl over bins[i] <= ell < bins[i+1] (as CMBpipeline utilities.bineado)."""
    ell = np.arange(np.shape(cl)[-1])
    return np.array([np.mean(cl[..., (ell >= lo) & (ell < hi)], -1) for lo, hi in zip(bins[:-1], bins[1:])]).T


def bin_modes(ell_modes, power_modes, bins):
    """Mean of the 2D Fourier power over the modes with bins[i] <= |ell| < bins[i+1] (as PowerSpectrum.El_fid)."""
    idx = np.digitize(ell_modes, bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    s = np.bincount(idx[ok], weights=power_modes[ok], minlength=len(bins) - 1)
    n = np.bincount(idx[ok], minlength=len(bins) - 1)
    return s / np.maximum(n, 1)


def flat_spectrum_A(ps, m, ell_flat, bins, w2):
    """
    Pipeline A estimator: CMBpipeline PowerSpectrum.power (|fft2 * tfac|^2) of
    a plane map, binned on ell_flat (PowerSpectrum.flat_spectrum_xy), / w2.
    """
    return bin_modes(ell_flat, ps.power(m), bins) / w2


def flat_spectrum_enmap(harm, modlmap, bins, w2):
    """Pipeline C estimator: |harm|^2 of an enmap FFT with normalize='phys', binned on modlmap, / w2."""
    return bin_modes(np.asarray(modlmap).ravel(), np.abs(np.asarray(harm)).ravel() ** 2, bins) / w2


def curved_spectrum(alm, bins, w2):
    """Pipeline B estimator: alm2cl of the a_lm of the masked map, binned, / w2."""
    return bin_cl(curvedsky.alm2cl(alm), bins) / w2


def response_stats(ell_centers, response, l0):
    """
    Shape of a mode-coupling response (normalized to unit sum): centroid shift
    <ell> - l0, width = 84th - 16th percentile, fraction outside l0 +- 20.
    """
    r = np.clip(response, 0, None)
    r = r / r.sum()
    cdf = np.cumsum(r)
    p16, p84 = np.interp([0.16, 0.84], cdf, ell_centers)
    return dict(shift=np.sum(r * ell_centers) - l0, width=p84 - p16,
                out20=r[np.abs(ell_centers - l0) > 20].sum())


def paired(a, b):
    """
    Paired comparison over realizations (axis 0): mean difference, its standard
    error, and the significance in units of that error.
    """
    d = np.asarray(a) - np.asarray(b)
    mean, err = d.mean(0), d.std(0, ddof=1) / np.sqrt(d.shape[0])
    return mean, err, mean / err
