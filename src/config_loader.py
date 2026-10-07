"""
Single entry point for reading config.dict (same logic as CMBpipeline's
config_loader, restricted to the keys of the pixell dataset).

    from config_loader import load_config
    cfg = load_config()            # config.dict next to this module (or $WF_CONFIG)

Derived keys:
  folders (absolute, trailing "/", created on demand): root_folder, data_folder, aux_folder
  files: name_train, name_valid, so_hits_path, spectra_path, seeds_path,
         mask_car_path, hits_car_path, variance_car_path
  training (if the keys exist): model_path, loss_path, study_db
  misc:  map_rescale_factor (scalar for the active mask_type), fwhm_rad, bin_kwargs
The folder in cfg["cmbpipeline_source"] is appended to sys.path (read-only use of CMBpipeline).
"""

import ast
import math
import os
import re
import sys

REQUIRED_KEYS = [
    "mask_type", "nside", "res_arcmin", "npixels_x", "npixels_y",
    "r", "lmin", "lmax", "lmax_sim",
    "fwhm_arcmin",
    "noise_type", "inho_type", "DEFAULT_NOISE_LEVEL",
    "map_rescale_factor",
    "smooth", "apo",
    "nsims_train", "nsims_valid", "nsims_test", "cmb_seed0", "noise_seed0",
    "root_folder", "data_folder", "aux_folder", "dataset_tag",
    "mask_files", "cmbpipeline_source",
]

OUTPUT_FOLDERS = ["data_folder", "aux_folder"]
TRAINING_FOLDERS = ["study_folder", "model_folder", "loss_folder"]   # optional (training_car.py)


# --------------------------------------------------------------------------- #
# helpers (as in CMBpipeline)
# --------------------------------------------------------------------------- #

def _strip_comments(text):
    # Remove # comments from each line (# never appears inside string values here).
    return '\n'.join(re.sub(r'\s*#.*$', '', line) for line in text.splitlines())


def find_config(path=None):
    """
    Locate config.dict:
      1. explicit `path`
      2. environment variable WF_CONFIG
      3. ./config.dict (current working directory)
      4. config.dict next to this module
    """
    candidates = []
    if path is not None:
        candidates.append(path)
    if os.environ.get("WF_CONFIG"):
        candidates.append(os.environ["WF_CONFIG"])
    candidates.append(os.path.join(os.getcwd(), "config.dict"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.dict"))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"config.dict not found; looked in {candidates}")


def as_folder(root, folder):
    """Absolute folder path with trailing separator; relative paths hang from root."""
    if not os.path.isabs(folder):
        folder = os.path.join(root, folder)
    return os.path.join(os.path.normpath(folder), "")


def as_file(folder, name):
    """Absolute file path: absolute names pass through, otherwise join with folder."""
    return name if os.path.isabs(name) else os.path.join(folder, name)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def load_config(path=None, create_dirs=True):
    """Load and validate config.dict (see module docstring for the derived keys)."""
    cfg_file = find_config(path)
    with open(cfg_file) as f:
        cfg = ast.literal_eval(_strip_comments(f.read()))

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config.dict is missing required keys: {missing}")
    if cfg["mask_type"] != "rect":
        raise ValueError("only mask_type = 'rect' (SO) is supported")
    if cfg["noise_type"] != "inho":
        raise ValueError("only noise_type = 'inho' is supported")
    for key in ("npixels_x", "npixels_y"):
        if cfg[key] % 32:
            raise ValueError(f"{key} = {cfg[key]} must be a multiple of 32 (5 poolings in DeepWiener)")

    # mask-type dependent scalars
    mrf = cfg["map_rescale_factor"]
    cfg["map_rescale_factor"] = mrf[cfg["mask_type"]] if isinstance(mrf, dict) else mrf

    # derived scalars
    cfg["fwhm_rad"] = math.radians(cfg["fwhm_arcmin"] / 60.0)
    b = cfg.get("binning", {})
    cfg["bin_kwargs"] = dict(lmin=cfg["lmin"], lmax=cfg["lmax"], frac=b.get("frac", 0.2),
                             min_width=b.get("min_width", 20), extra_edges=b.get("extra_edges"))

    # folders
    cfg["config_file"] = cfg_file
    root = as_folder(os.path.dirname(cfg_file), cfg["root_folder"])
    cfg["root_folder"] = root
    folders = OUTPUT_FOLDERS + [k for k in TRAINING_FOLDERS if k in cfg]
    for key in folders:
        cfg[key] = as_folder(root, cfg[key])
    if create_dirs:
        for key in folders:
            os.makedirs(cfg[key], exist_ok=True)

    # files
    tag, aux = cfg["dataset_tag"], cfg["aux_folder"]
    cfg["name_train"]        = as_file(cfg["data_folder"], f"train_{tag}.pt")
    cfg["name_valid"]        = as_file(cfg["data_folder"], f"valid_{tag}.pt")
    cfg["so_hits_path"]      = as_file(root, cfg["mask_files"]["so_hits"])
    cfg["spectra_path"]      = as_file(aux, f"cls_signal_r{cfg['r']}.npy")
    cfg["seeds_path"]        = as_file(aux, f"seeds_{tag}.npz")
    cfg["mask_car_path"]     = as_file(aux, f"mask_car_{tag}.fits")
    cfg["hits_car_path"]     = as_file(aux, f"hits_car_{tag}.fits")
    cfg["variance_car_path"] = as_file(aux, f"variance_car_{tag}.fits")

    # training (as CMBpipeline's config_loader)
    if "model_folder" in cfg:
        cfg["model_path"] = cfg["model_folder"]
        cfg["loss_path"]  = cfg["loss_folder"]
    if "study_name" in cfg:
        cfg["study_db"] = "sqlite:///" + as_file(cfg["study_folder"], f"{cfg['study_name']}.db")

    # read-only use of CMBpipeline (appended: modules of this repository keep precedence)
    src = as_folder(root, cfg["cmbpipeline_source"])
    cfg["cmbpipeline_source"] = src
    if src not in sys.path:
        sys.path.append(src)
    return cfg


def bin_edges(cfg):
    """Bin edges shared by every estimator (CMBpipeline utilities.compute_bins_fractional)."""
    import utilities
    return utilities.compute_bins_fractional(**cfg["bin_kwargs"])


if __name__ == "__main__":
    import pprint
    pprint.pprint(load_config(create_dirs=False))
