"""
pixell helpers for pipelines B (CAR + curved sky) and C (rotated CAR + flat FFT).
Only the pieces that are reused; the transforms themselves are called in the
notebook.
"""

import numpy as np
import healpy as hp
from pixell import enmap, curvedsky, utils


def car_mask(shape, wcs, mask_hp, nside, R=None):
    """
    Transfer the HEALPix mask to a CAR geometry: each CAR pixel centre takes
    the value of the HEALPix pixel that contains it (the rule CMBpipeline uses
    for mask2d). R: rotation matrix of the CAR coordinates (pipeline C);
    the pixel direction in the original frame is R^T n'.
    """
    dec, ra = enmap.posmap(shape, wcs)
    vec = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)
    if R is not None:
        vec = vec @ R
    theta = np.arccos(np.clip(vec[..., 2], -1, 1))
    phi = np.mod(np.arctan2(vec[..., 1], vec[..., 0]), 2 * np.pi)
    return enmap.ndmap(mask_hp[hp.ang2pix(nside, theta, phi)].astype(float), wcs)


def w2_car(mask):
    """<mask^2> over the sphere, area weighted (f_sky normalization of the curved-sky spectrum)."""
    return float(np.sum(np.asarray(mask) ** 2 * mask.pixsizemap()) / (4 * np.pi))


def rotation_to_equator(lonc_deg, latc_deg):
    """
    Euler angles for curvedsky.rotate_alm and the matrix R such that the
    rotated field is f'(R n) = f(n): the patch centre goes to (ra, dec) = (0, 0)
    with local north along +dec. R = Ry(latc) Rz(-lonc); rotate_alm(alm,
    psi=-lonc, theta=latc, phi=0) implements it (checked in the notebook).
    """
    lam, dl = np.radians(lonc_deg), np.radians(latc_deg)
    rz = np.array([[np.cos(-lam), -np.sin(-lam), 0], [np.sin(-lam), np.cos(-lam), 0], [0, 0, 1]])
    ry = np.array([[np.cos(dl), 0, np.sin(dl)], [0, 1, 0], [-np.sin(dl), 0, np.cos(dl)]])
    return (-lam, dl, 0.0), ry @ rz


def patch_geometry(vecs, R, res_arcmin, margin_arcmin):
    """
    CAR patch (slice of the fejer1 full sky) covering the observed pixels
    vecs (rotated by R) plus a margin.
    """
    fshape, fwcs = enmap.fullsky_geometry(res=res_arcmin * utils.arcmin, variant="fejer1")
    v = vecs @ R.T
    dec, ra = np.arcsin(np.clip(v[:, 2], -1, 1)), np.arctan2(v[:, 1], v[:, 0])
    m = margin_arcmin * utils.arcmin
    corners = np.array([[dec.min() - m, dec.max() + m], [ra.min() - m, ra.max() + m]])
    pix = enmap.sky2pix(fshape, fwcs, corners)
    y0, y1 = int(np.floor(pix[0].min())), int(np.ceil(pix[0].max())) + 1
    x0, x1 = int(np.floor(pix[1].min())), int(np.ceil(pix[1].max())) + 1
    return enmap.slice_geometry(fshape, fwcs, (slice(y0, y1), slice(x0, x1)))
