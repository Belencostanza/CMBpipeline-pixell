"""
Pipeline A: the current CMBpipeline (SO case), as plain functions that call
../CMBpipeline/source (read-only).

geometry.build_geometry is not called directly (its config_loader points to
cluster paths and creates folders); so_geometry reproduces its SO call sequence
(geometry.py:81-100) with the same functions.
"""

import numpy as np
import healpy as hp
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay

import importlib.util
import os

import sky_input  # noqa: F401  (puts ../CMBpipeline/source on the path)
import projections as pj
import utilities

# CMBpipeline's make_dataset.py, loaded by path: src/ has its own make_dataset.py (Task 2)
_spec = importlib.util.spec_from_file_location(
    "cmbpipeline_make_dataset", os.path.join(sky_input.CMBPIPELINE_SOURCE, "make_dataset.py"))
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)


def _wrap(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def so_geometry(nside, nx, ny, fact, so_hits_file):
    """
    SO mask on HEALPix, equidistant projection, plane grid, dx/dy and psi
    (same calls as geometry.py:81-100). Returns a dict.
    """
    sky_region = pj.sph_mask(nside=nside, radius=40, so_hits_file=so_hits_file)
    mask, nhits, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = sky_region.make_SO_mask()

    proj = pj.proj_2d(lonc, latc, theta_idx, phi_idx, vec_center, nbins=nx, fact=fact)
    z_eq, _, _ = proj.proj_conventions(vecs, vec_center)
    mask2d, nbins_x, nbins_y = proj.define_region_mask_from_res_margins(z_eq, mask.astype(float), nside)
    assert (nbins_x, nbins_y) == (nx, ny), f"plane grid {nbins_x}x{nbins_y} != {nx}x{ny}"

    dx1, dy1, _ = utilities.calculate_res(z_eq, nx, ny)
    dx, dy, _ = utilities.calculate_res_margin(z_eq, dx1, dy1, nx, ny, fact=fact)

    theta0, phi0 = hp.vec2ang(vec_center)
    theta, phi = hp.vec2ang(vecs)
    psi = pj.rotate_geo(theta, _wrap(phi), theta0, _wrap(phi0)).psi_Q_to_P()

    # plane grid of grid_bins_mask_from_res_margin (projections.py:760-779)
    pix = np.radians(proj.reso_arcmin / 60.0)
    x_vals = z_eq[:, 0].min() - fact * pix + pix * np.arange(nx)
    y_vals = z_eq[:, 1].min() - fact * pix + pix * np.arange(ny)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)

    return dict(mask=mask.astype(float), valid_index=valid_index, vecs=vecs, vec_center=vec_center,
                lonc=lonc, latc=latc, theta0=theta0, phi0=phi0, proj=proj, z_eq=z_eq, mask2d=mask2d,
                dx=dx, dy=dy, psi=psi, pix=pix, x_vals=x_vals, y_vals=y_vals,
                grid_x=grid_x, grid_y=grid_y, nx=nx, ny=ny, nside=nside)


def rotate_qu(q, u, psi):
    """make_dataset.make_rot: (Q + iU)' = exp(-2 i psi) (Q + iU), rotation to the patch-centre frame."""
    return dd.make_dataset(1, 1, 1, False, False).make_rot(q, u, psi)


def qu_to_plane(geo, Q_sph, U_sph, tri=None):
    """
    Mask, psi rotation and Clough-Tocher interpolation onto the plane grid.
    tri=None: CMBpipeline's proj_2d.grid_bins_mask_from_res_margin (griddata,
    builds the Delaunay triangulation every call). tri=Delaunay(z_eq): the same
    interpolant (griddata(method='cubic') == CloughTocher2DInterpolator on
    Delaunay(points)) with the triangulation reused; the output is identical.
    """
    q_rot, u_rot = rotate_qu((geo["mask"] * Q_sph)[geo["valid_index"]],
                             (geo["mask"] * U_sph)[geo["valid_index"]], geo["psi"])
    if tri is None:
        Q_pln = geo["proj"].grid_bins_mask_from_res_margin(geo["z_eq"], q_rot, geo["mask2d"])
        U_pln = geo["proj"].grid_bins_mask_from_res_margin(geo["z_eq"], u_rot, geo["mask2d"])
        return Q_pln, U_pln
    out = []
    for vals in (q_rot, u_rot):
        grid = CloughTocher2DInterpolator(tri, vals, fill_value=0)(geo["grid_x"], geo["grid_y"])
        grid[geo["mask2d"] == 0] = 0
        out.append(grid)
    return out[0], out[1]


def nomask_setup(geo):
    """
    No-mask control: every HEALPix pixel whose projection falls inside the
    plane rectangle, its psi and the Delaunay triangulation of its (x, y).
    """
    nside = geo["nside"]
    vec_all = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    near = np.nonzero(vec_all @ geo["vec_center"] > np.cos(np.radians(80)))[0]
    z, _, _ = geo["proj"].proj_conventions(vec_all[near], geo["vec_center"])
    pix, xv, yv = geo["pix"], geo["x_vals"], geo["y_vals"]
    inside = ((z[:, 0] > xv[0] - pix) & (z[:, 0] < xv[-1] + pix) &
              (z[:, 1] > yv[0] - pix) & (z[:, 1] < yv[-1] + pix))
    index = near[inside]
    theta, phi = hp.vec2ang(vec_all[index])
    psi = pj.rotate_geo(theta, _wrap(phi), geo["theta0"], _wrap(geo["phi0"])).psi_Q_to_P()
    return dict(index=index, psi=psi, tri=Delaunay(z[inside]))


def qu_to_plane_nomask(geo, nm, Q_sph, U_sph):
    """psi rotation and Clough-Tocher interpolation of all pixels in the rectangle (mask2d = 1)."""
    q_rot, u_rot = rotate_qu(Q_sph[nm["index"]], U_sph[nm["index"]], nm["psi"])
    return [CloughTocher2DInterpolator(nm["tri"], v, fill_value=0)(geo["grid_x"], geo["grid_y"])
            for v in (q_rot, u_rot)]


def plane_eb(Q_pln, U_pln, nx, ny, dx, dy):
    """
    Flat-sky Q/U -> E/B maps as in PowerSpectrum.get_nn_outputs:
    utilities.transf_eb2_np, then irfft2 / tfac.
    """
    tfac = np.sqrt((dx * dy) / (nx * ny))
    efft, bfft = utilities.transf_eb2_np(Q_pln, U_pln, nx, dx, ny, dy)
    return np.fft.irfft2(efft, s=(ny, nx)) / tfac, np.fft.irfft2(bfft, s=(ny, nx)) / tfac


def grid_positions(geo):
    """(dec, ra) of the plane grid points: inverse equidistant map (projections.py:569-574)."""
    zc = geo["vec_center"] / np.linalg.norm(geo["vec_center"])
    north = np.array([0., 0., 1.]) - zc[2] * zc
    north /= np.linalg.norm(north)
    west = np.cross(zc, north)
    gx, gy = geo["grid_x"], geo["grid_y"]
    theta, phi = np.hypot(gx, gy), np.arctan2(gx, gy)
    vec = (np.cos(theta)[..., None] * zc
           + np.sin(theta)[..., None] * (np.cos(phi)[..., None] * north + np.sin(phi)[..., None] * west))
    return np.array([np.arcsin(np.clip(vec[..., 2], -1, 1)), np.arctan2(vec[..., 1], vec[..., 0])])
