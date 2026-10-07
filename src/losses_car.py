"""
Curved-sky J3 loss on the CAR cut-out: the counterpart of CMBpipeline's
losses.lossj3_plane_beam_inho (../CMBpipeline/source/losses.py) with the flat
FFTs replaced by spin-2 spherical-harmonic transforms (sht_torch.CarSHT).

With yhat the network output, d the normalized data, f = map_rescale_factor,
a = A yhat (E, B a_lm, l <= lmax) and N_pix = ny * nx:

  J3 = (1/N_pix) [ sum_i m_i (d_i - (Y B_l a)_i)^2 / (sigma_i^2 f^2)
                 + sum_{l>=2, m} w_m ( |a^E_lm|^2 / (C^EE_l f^2) + |a^B_lm|^2 / (C^BB_l f^2) ) ]

w_0 = 1, w_{m>0} = 2. CMBpipeline's code computes the same expression with
planar Fourier modes (docs/04_training_plan.md, Sec. 2); the overall 1/N_pix
scale is kept so that loss values and lr / wd ranges stay comparable.

The loss depends on yhat only through a = A yhat, so the Wiener-filtered map is
defined as the band-limited wf_map(yhat) = Y A yhat.
"""

import numpy as np
import torch


class CarLossJ3:

    def __init__(self, sht, bl, clee, clbb, factor, device, dtype=torch.float32):
        """
        Args:
            sht: sht_torch.CarSHT of the dataset geometry
            bl: beam transfer function B_l (l = 0 ... >= sht.lmax), the one used to make the data
            clee, clbb: C_l^EE, C_l^BB [muK^2] (l = 0 ... >= sht.lmax)
            factor: map_rescale_factor (the data are multiplied by it in the loader)
            dtype: real dtype of the loss tensors (float64 only for gradcheck)
        """
        self.sht = sht
        self.device = device
        ell = sht.ell
        self.bl_lm = torch.tensor(bl[ell], dtype=dtype, device=device)            # (nalm,)
        inv = np.zeros((2, sht.nalm))
        good = ell >= 2
        inv[0, good] = sht.w[good] / (clee[ell[good]] * factor**2)
        inv[1, good] = sht.w[good] / (clbb[ell[good]] * factor**2)
        self.inv_cl_lm = torch.tensor(inv, dtype=dtype, device=device)           # (2, nalm), includes w_m
        self.factor = factor
        self.npix = sht.shape[0] * sht.shape[1]

    def realspace_loss_beam_inho(self, y_true, alm, mask, inho):
        """(1/N_pix) sum_i m_i (d_i - (Y B a)_i)^2 / (sigma_i^2 f^2), Q + U, mean over the batch."""
        pred_beam = self.sht.synthesis(alm * self.bl_lm)                                   # (B, 2, ny, nx)
        mask = mask.to(self.device)
        inho = inho.to(self.device)
        inv_var = torch.zeros_like(inho)
        valid = (mask > 0) & (inho > 0)
        inv_var[valid] = 1.0 / inho[valid]
        diff = mask * (y_true - pred_beam) ** 2
        loss = inv_var * diff / self.factor**2
        return torch.mean(loss[:, 0]) + torch.mean(loss[:, 1])

    def harmonic_prior(self, alm):
        """(1/N_pix) sum_{l>=2,m} w_m |a_lm|^2 / C_l, E + B, mean over the batch."""
        power = alm.real ** 2 + alm.imag ** 2                                              # (B, 2, nalm)
        return torch.mean(torch.sum(power * self.inv_cl_lm, dim=(-2, -1))) / self.npix

    def __call__(self, y_true, y_pred, mask, inho):
        alm = self.sht.analysis(y_pred)
        term1 = self.realspace_loss_beam_inho(y_true, alm, mask, inho)
        term2 = self.harmonic_prior(alm)
        return term1 + term2

    def terms(self, y_true, y_pred, mask, inho):
        """Both terms separately (diagnostics)."""
        alm = self.sht.analysis(y_pred)
        return self.realspace_loss_beam_inho(y_true, alm, mask, inho), self.harmonic_prior(alm)

    def wf_map(self, y_pred):
        """Band-limited Wiener-filtered map Y A yhat (the only part of yhat the loss sees)."""
        return self.sht.synthesis(self.sht.analysis(y_pred))
