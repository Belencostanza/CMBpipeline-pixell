import sys, platform, os
import time
import numpy as np

import healpy as hp
import torch
from pixell import enmap, curvedsky, utils


class make_dataset:

    """
    Dataset generator for CMB polarization maps on a CAR cut-out of the sphere
    (pixell). Same role and layout as CMBpipeline's make_dataset, SO "rect"
    case, inhomogeneous noise. The maps are never projected onto a plane:

      a_lm (theory spectrum, seed) --beam--> curvedsky.alm2map (spin 2) on the
      CAR cut-out --> + noise (sigma^2 per CAR pixel from the SO hits) --> x mask

    Network input (N, 4, ny, nx), float32: Q_obs, U_obs, mask, sigma^2, as in
    CMBpipeline (DeepWiener_threechannels).
    """

    NUM_SPECTRUM_COMPONENTS = 3  # (cltt, clee, clbb); TE = TB = EB = 0 as in CMBpipeline

    def __init__(
        self,
        nsims_train: int,
        nsims_valid: int,
        nsims_test: int,
        smooth: bool,
        apo: bool,
        nside: int = 512,
        fwhm: float = 23,
        DEFAULT_NOISE_LEVEL: float = 5e-7,
        lmax_sim: int = 1535,
        res_arcmin: float = 9.6,
        npixels_x: int = 1120,
        npixels_y: int = 320,
    ):

        """
        Args:
            nsims_train, nsims_valid, nsims_test: number of maps of each set
            smooth: whether to apply the beam
            apo: apodized mask (not supported for the CAR dataset)
            nside: HEALPix resolution of the SO mask and hits map
            fwhm: beam full width at half maximum [arcmin]
            DEFAULT_NOISE_LEVEL: white-noise level [muK^2 sr]
            lmax_sim: band limit of the simulated a_lm
            res_arcmin: CAR resolution (fejer1 full sky)
            npixels_x, npixels_y: size of the CAR cut-out (multiples of 32)
        """
        if apo:
            raise NotImplementedError("apo = True is not supported for the CAR dataset")

        self.nsims_train = nsims_train
        self.nsims_valid = nsims_valid
        self.nsims_test = nsims_test
        self.smooth = smooth
        self.apo = apo
        self.nside = nside
        self.fwhm = np.radians(fwhm / 60)  # from arcmin to rad
        self.DEFAULT_NOISE_LEVEL = DEFAULT_NOISE_LEVEL
        self.lmax_sim = lmax_sim
        self.res = res_arcmin * utils.arcmin
        self.nx = npixels_x
        self.ny = npixels_y

    # ----------------------------------------------------------------------- #
    # spectra and signal
    # ----------------------------------------------------------------------- #

    def get_spectra(self, r, cache_file=None):
        """
        CMBpipeline fiducial spectra utilities.signal_spectrum(r) (CAMB 'total',
        C_ell in muK^2). Cached in cache_file.
        """
        if cache_file is not None and os.path.isfile(cache_file):
            cltt, clee, clbb = np.load(cache_file)
        else:
            import utilities   # ../CMBpipeline/source
            cltt, clee, clbb = utilities.signal_spectrum(r=r)
            if cache_file is not None:
                np.save(cache_file, np.array([cltt, clee, clbb]))
        return cltt, clee, clbb

    def _create_spectrum_array(self, cltt, clee, clbb):
        """(3, 3, lmax_sim + 1) spectrum matrix for rand_alm, TE = TB = EB = 0."""
        lmax = self.lmax_sim
        ps = np.zeros((self.NUM_SPECTRUM_COMPONENTS, self.NUM_SPECTRUM_COMPONENTS, lmax + 1))
        ps[0, 0], ps[1, 1], ps[2, 2] = cltt[:lmax + 1], clee[:lmax + 1], clbb[:lmax + 1]
        return ps

    def get_alm(self, ps, cmb_seed):
        """(a_T, a_E, a_B) of one realization; beam applied to a_E, a_B if smooth."""
        alm = curvedsky.rand_alm(ps, lmax=self.lmax_sim, seed=cmb_seed)
        if self.smooth:
            bl = hp.gauss_beam(self.fwhm, lmax=self.lmax_sim)     # = CMBpipeline utilities.bl
            alm[1] = curvedsky.almxfl(alm[1], bl)
            alm[2] = curvedsky.almxfl(alm[2], bl)
        return alm

    def get_QUmaps_car(self, alm, shape, wcs):
        """Q, U on the CAR cut-out: exact spin-2 synthesis at the pixel centres."""
        qu = enmap.zeros((2,) + tuple(shape[-2:]), wcs)
        curvedsky.alm2map(alm[1:], qu, spin=2)
        return qu[0], qu[1]

    # ----------------------------------------------------------------------- #
    # geometry, mask and noise
    # ----------------------------------------------------------------------- #

    def get_sky_region(self, so_hits_file):
        """SO mask and hits on HEALPix with CMBpipeline's sph_mask.make_SO_mask."""
        import projections as pj   # ../CMBpipeline/source
        sky_region = pj.sph_mask(nside=self.nside, so_hits_file=so_hits_file)
        mask, nhits, valid_index, vecs, vec_center, theta_idx, phi_idx, lonc, latc = sky_region.make_SO_mask()
        return mask.astype(float), nhits

    def healpix_to_car(self, m_hp, shape, wcs):
        """Value of the HEALPix pixel containing each CAR pixel centre (rule of CMBpipeline's mask2d)."""
        dec, ra = enmap.posmap(shape, wcs)
        pix = hp.ang2pix(self.nside, np.pi / 2 - dec, np.mod(ra, 2 * np.pi))
        return enmap.ndmap(np.asarray(m_hp)[pix].astype(float), wcs)

    def car_geometry(self, mask_hp):
        """
        CAR cut-out of ny x nx pixels centred on the bounding box of the SO mask,
        sliced from the fejer1 full-sky grid at res (so that curved-sky
        transforms on the cut-out are exact with map2alm(method='2d')).
        """
        fshape, fwcs = enmap.fullsky_geometry(res=self.res, variant="fejer1")
        mask_full = self.healpix_to_car(mask_hp, fshape, fwcs)
        ys, xs = np.nonzero(np.asarray(mask_full) > 0)
        ny0, nx0 = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if ny0 > self.ny or nx0 > self.nx:
            raise ValueError(f"mask bounding box {nx0} x {ny0} does not fit in {self.nx} x {self.ny}")
        y0 = ys.min() - (self.ny - ny0) // 2
        x0 = xs.min() - (self.nx - nx0) // 2
        shape, wcs = enmap.slice_geometry(fshape, fwcs, (slice(y0, y0 + self.ny), slice(x0, x0 + self.nx)))
        return shape, wcs

    def get_inhomogenous_noise(self, mask_hp, nhits_hp, mask_car, hits_car):
        """
        Pixel noise variance on the CAR cut-out.

        CMBpipeline (utilities.nhits_to_sigma2) gives each HEALPix pixel
        sigma^2 = (DEFAULT_NOISE_LEVEL / Omega_HP) (1/hits) / <1/hits>, i.e. a
        white-noise level N(n) = DEFAULT_NOISE_LEVEL (1/hits) / <1/hits> per
        steradian. Here the same N(n) is spread over each CAR pixel:
        sigma^2 = N(n) / Omega_pixel(Dec), with <1/hits> the same HEALPix mean.
        """
        inv_hits_hp = 1.0 / (nhits_hp[mask_hp > 0] + 1e-6)
        norm = np.mean(inv_hits_hp)
        inv_hits = np.zeros(mask_car.shape)
        inside = np.asarray(mask_car) > 0
        inv_hits[inside] = 1.0 / (np.asarray(hits_car)[inside] + 1e-6)
        noise_level = self.DEFAULT_NOISE_LEVEL * inv_hits / norm                  # [muK^2 sr]
        variance_map = noise_level / np.asarray(mask_car.pixsizemap())           # [muK^2]
        return enmap.ndmap(variance_map * np.asarray(mask_car), mask_car.wcs)

    def get_inho_noise(self, variance_map, mask, rng):
        """One white-noise realization with the given variance, only inside the mask."""
        noise_map = np.zeros(variance_map.shape)
        inside = np.asarray(mask) > 0
        noise_map[inside] = rng.normal(loc=0.0, scale=np.sqrt(np.asarray(variance_map)[inside]))
        return noise_map

    def make_inho_maps_car(self, ps, variance_map, mask, shape, wcs, cmb_seed, noise_seed):
        """Q_obs, U_obs = mask * (signal + noise) of one realization on the CAR cut-out."""
        alm = self.get_alm(ps, cmb_seed)
        mapQ, mapU = self.get_QUmaps_car(alm, shape, wcs)

        rng_noise = np.random.default_rng(noise_seed)
        noiseQ = self.get_inho_noise(variance_map, mask, rng_noise)
        noiseU = self.get_inho_noise(variance_map, mask, rng_noise)

        dataQ = np.asarray(mask) * (np.asarray(mapQ) + noiseQ)
        dataU = np.asarray(mask) * (np.asarray(mapU) + noiseU)
        return dataQ, dataU

    # ----------------------------------------------------------------------- #
    # dataset
    # ----------------------------------------------------------------------- #

    def make_train_dataset(
        self,
        filename_train: str,
        filename_valid: str,
        so_hits_file: str,
        r: float = 0.034,
        cmb_seed0: int = 0,
        noise_seed0: int = 100000,
        spectra_file: str = None,
        mask_car_file: str = None,
        hits_car_file: str = None,
        variance_car_file: str = None,
        seeds_file: str = None,
        ):

        """
        Generate and save the training and validation sets, each a float32
        tensor (N, 4, ny, nx) with channels Q_obs, U_obs, mask, sigma^2.
        Map i (train: 0 ... nsims_train-1, valid: nsims_train ...) uses CMB seed
        cmb_seed0 + i and noise seed noise_seed0 + i.

        The fixed channels (mask, sigma^2), the hits, the spectra and the seeds
        are also written (FITS files keep the CAR geometry / WCS).
        """
        nsims = self.nsims_train + self.nsims_valid

        cltt, clee, clbb = self.get_spectra(r, spectra_file)
        ps = self._create_spectrum_array(cltt, clee, clbb)

        mask_hp, nhits_hp = self.get_sky_region(so_hits_file)
        shape, wcs = self.car_geometry(mask_hp)
        mask_car = self.healpix_to_car(mask_hp, shape, wcs)
        hits_car = self.healpix_to_car(nhits_hp, shape, wcs)
        variance_map = self.get_inhomogenous_noise(mask_hp, nhits_hp, mask_car, hits_car)
        print(f'CAR cut-out: nx x ny = {shape[-1]} x {shape[-2]}, observed pixels: {int(mask_car.sum())}')

        for fname, m in ((mask_car_file, mask_car), (hits_car_file, hits_car), (variance_car_file, variance_map)):
            if fname is not None:
                enmap.write_map(fname, m)

        X_train = torch.zeros((self.nsims_train, 4, self.ny, self.nx), dtype=torch.float32)
        X_valid = torch.zeros((self.nsims_valid, 4, self.ny, self.nx), dtype=torch.float32)
        mask_t = torch.from_numpy(np.asarray(mask_car, dtype=np.float32))
        var_t = torch.from_numpy(np.asarray(variance_map, dtype=np.float32))

        cmb_seeds = cmb_seed0 + np.arange(nsims)
        noise_seeds = noise_seed0 + np.arange(nsims)

        t0 = time.time()
        for map_id in range(nsims):

            print('map', map_id)
            gridQ_data, gridU_data = self.make_inho_maps_car(ps, variance_map, mask_car, shape, wcs,
                                                            int(cmb_seeds[map_id]), int(noise_seeds[map_id]))

            X = X_train if map_id < self.nsims_train else X_valid
            i = map_id if map_id < self.nsims_train else map_id - self.nsims_train
            X[i, 0, :, :] = torch.from_numpy(gridQ_data.astype(np.float32))
            X[i, 1, :, :] = torch.from_numpy(gridU_data.astype(np.float32))
            X[i, 2, :, :] = mask_t
            X[i, 3, :, :] = var_t

        print(f'{nsims} maps in {time.time() - t0:.1f} s')

        torch.save(X_train, filename_train)
        torch.save(X_valid, filename_valid)
        if seeds_file is not None:
            np.savez(seeds_file, cmb_seeds_train=cmb_seeds[:self.nsims_train], noise_seeds_train=noise_seeds[:self.nsims_train],
                     cmb_seeds_valid=cmb_seeds[self.nsims_train:], noise_seeds_valid=noise_seeds[self.nsims_train:])
