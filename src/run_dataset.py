import sys, platform, os
import numpy as np

import time
import make_dataset as dd
from config_loader import load_config

cfg = load_config()

nsims_train         = cfg["nsims_train"]
nsims_valid         = cfg["nsims_valid"]
nsims_test          = cfg["nsims_test"]
smooth              = cfg["smooth"]
apo                 = cfg["apo"]
nside               = cfg["nside"]
fwhm                = cfg["fwhm_arcmin"]
default_noise_level = cfg["DEFAULT_NOISE_LEVEL"]
filename_train      = cfg["name_train"]
filename_valid      = cfg["name_valid"]

make_data = dd.make_dataset(nsims_train, nsims_valid, nsims_test, smooth, apo,
                            nside=nside, fwhm=fwhm, DEFAULT_NOISE_LEVEL=default_noise_level,
                            lmax_sim=cfg["lmax_sim"], res_arcmin=cfg["res_arcmin"],
                            npixels_x=cfg["npixels_x"], npixels_y=cfg["npixels_y"])

t0 = time.time()
make_data.make_train_dataset(filename_train, filename_valid,
                             so_hits_file=cfg["so_hits_path"],          # SO hits map (mask + noise model)
                             r=cfg["r"],
                             cmb_seed0=cfg["cmb_seed0"], noise_seed0=cfg["noise_seed0"],
                             spectra_file=cfg["spectra_path"],          # cached CAMB spectra -> aux_folder
                             mask_car_file=cfg["mask_car_path"],        # fixed channels with their WCS -> aux_folder
                             hits_car_file=cfg["hits_car_path"],
                             variance_car_file=cfg["variance_car_path"],
                             seeds_file=cfg["seeds_path"])
t1 = time.time()

print('Time to create dataset:', t1-t0)
