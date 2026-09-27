"""
Common input (reference sky) of Task 1: one set of beamed a_lm, drawn with
pixell from CMBpipeline's fiducial spectrum with a fixed seed.

HEALPix and CAR maps are only samplings of the field defined by these a_lm.
"""

import os
import sys

import numpy as np
import healpy as hp
from pixell import curvedsky

CMBPIPELINE_SOURCE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "CMBpipeline", "source"))
if CMBPIPELINE_SOURCE not in sys.path:
    sys.path.append(CMBPIPELINE_SOURCE)         # read-only use of CMBpipeline (appended: no shadowing of src modules)


def get_spectra(r, cache_file=None):
    """
    CMBpipeline fiducial spectra utilities.signal_spectrum(r): (cltt, clee, clbb),
    C_ell in muK^2 (CAMB 'total'). Cached in cache_file (CAMB is slow).
    """
    if cache_file is not None and os.path.isfile(cache_file):
        return np.load(cache_file)
    import utilities
    cls = np.array(utilities.signal_spectrum(r=r))
    if cache_file is not None:
        np.save(cache_file, cls)
    return cls


def beam(fwhm_arcmin, lmax):
    """Gaussian beam B_ell = exp(-l(l+1) sigma^2 / 2), same as CMBpipeline utilities.bl."""
    return hp.gauss_beam(np.radians(fwhm_arcmin / 60.0), lmax=lmax)


def make_alm(cltt, clee, clbb, lmax, fwhm_arcmin, seed, comps="full"):
    """
    Beamed (a_T, a_E, a_B), shape (3, nalm), healpy/pixell m-major layout.
    comps: "full", "E" (a_B = 0) or "B" (a_E = 0). TE = TB = EB = 0 as in
    CMBpipeline (make_dataset._create_spectrum_array). For a given seed the
    E and B parts are the same in the three cases.
    """
    ps = np.zeros((3, 3, lmax + 1))
    ps[0, 0], ps[1, 1], ps[2, 2] = cltt[:lmax + 1], clee[:lmax + 1], clbb[:lmax + 1]
    alm = curvedsky.rand_alm(ps, lmax=lmax, seed=seed)
    bl = beam(fwhm_arcmin, lmax)
    alm[1] = curvedsky.almxfl(alm[1], bl)
    alm[2] = curvedsky.almxfl(alm[2], bl)
    if comps == "E":
        alm[2] = 0
    elif comps == "B":
        alm[1] = 0
    return alm


def shell_alm(lmax, l0, half_width, seed):
    """
    E-only a_lm with power only in l0 - half_width <= l <= l0 + half_width
    (flat spectrum C_l = 1, random phases, no beam): input of the mode-mixing test.
    """
    cl = np.zeros(lmax + 1)
    cl[l0 - half_width:l0 + half_width + 1] = 1.0
    ps = np.zeros((3, 3, lmax + 1))
    ps[1, 1] = cl
    return curvedsky.rand_alm(ps, lmax=lmax, seed=seed)
