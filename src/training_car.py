
import numpy as np
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
import math, gc
import optuna
import healpy as hp
from pixell import enmap

from config_loader import load_config
cfg = load_config()

from network_2d import DeepWiener_threechannels    # ../CMBpipeline/source (read-only)
from sht_torch import CarSHT
from losses_car import CarLossJ3

# Same procedure as CMBpipeline's training_opt_beam_changed.py (SO "rect", inho noise).
# Changes (docs/04_training_plan.md):
#  - no plane geometry (dx, dy) and no flat C_l interpolation: the CAR geometry is read from the mask FITS
#  - loss: curved-sky J3 (losses_car.CarLossJ3) instead of losses.lossj3_plane_beam_inho
#  - optional fixed hyperparameters (cfg["fixed_params"]) and dataset subsets (cfg["nuse_train"], cfg["nuse_valid"])
#    for short tests; timing of the transforms is printed every epoch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)


#############################################READ CONFIG#################################################

epochs               = cfg["epochs"]
noise_type           = cfg["noise_type"]
map_rescale_factor_q = cfg["map_rescale_factor"]
fwhm_rad             = cfg["fwhm_rad"]
name_train           = cfg["name_train"]
name_valid           = cfg["name_valid"]
model_path           = cfg["model_path"]
loss_path            = cfg["loss_path"]
batch_size           = cfg["batch_size"]

cltt, clee, clbb = np.load(cfg["spectra_path"])     # written by run_dataset.py (CMBpipeline signal_spectrum(r))

# CAR geometry of the dataset and spin-2 transforms up to sht_lmax
mask_car = enmap.read_map(cfg["mask_car_path"])
sht = CarSHT(mask_car.shape, mask_car.wcs, lmax=cfg["sht_lmax"], method=cfg["sht_method"], nthread=cfg["sht_nthread"])
bl = hp.gauss_beam(fwhm_rad, lmax=cfg["sht_lmax"])  # same beam as the dataset (make_dataset.get_alm)


######################### DATASET #############################################

class CMBDataset(Dataset):
    def __init__(self, inputs, targets):
        self.X = inputs
        self.Y = targets

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
    def __len__(self):
        return len(self.Y)


def create_dataloader(filename, noise_type, shuffle=True, batch_size=1, nuse=None):

    if noise_type != "inho":
        raise ValueError("only noise_type = 'inho' is supported")
    data_array = torch.load(filename, mmap=True)      # memory-mapped: only the maps used are read
    if nuse is not None:
        data_array = data_array[:nuse]
    qobs = data_array[:,0:1,:,:]
    uobs = data_array[:,1:2,:,:]
    mask = data_array[:,2:3,:,:]
    inho = data_array[:,3:4,:,:]

    qobs_norm = (qobs - qobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q
    uobs_norm = (uobs - uobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q
    inho_norm = inho

    del data_array

    obs_norm = torch.cat((qobs_norm, uobs_norm, mask, inho_norm), dim=1)
    target_norm = torch.cat((qobs_norm, uobs_norm), dim=1)

    dataset = CMBDataset(obs_norm, target_norm)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        )

    return dataloader, mask[0], inho[0]


train_loader, mask, inho = create_dataloader(name_train, noise_type, batch_size=batch_size, nuse=cfg.get("nuse_train"))
valid_loader, _, _ = create_dataloader(name_valid, noise_type, batch_size=batch_size, nuse=cfg.get("nuse_valid"))
mask = mask.to(device)
inho = inho.to(device)

loss_car = CarLossJ3(sht, bl, clee, clbb, map_rescale_factor_q, device)

def criterion(y_true, y_pred):
    return loss_car(y_true, y_pred, mask, inho)


################################################ TRAINING #########################################
def hyper(hyperparameters):
    # CMBpipeline's hyper() shifts the labels (filters3 is written as "lr", lr as "wd"); fixed here
    return ("n_filters0_" + str(hyperparameters[0]) + "_n_filters1_" + str(hyperparameters[1]) +
            "_n_filters2_" + str(hyperparameters[2]) + "_n_filters3_" + str(hyperparameters[3]) +
            "_lr_" + "{:.3e}".format(hyperparameters[4]) + "_wd_" + "{:.3e}".format(hyperparameters[5]))


def train(train_loader, model, optimizer, criterion, scheduler):

    train_loss = 0.0
    model.train()

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        pred = model(inputs)

        loss_wf = criterion(targets, pred)
        loss_wf.backward()

        optimizer.step()
        scheduler.step()
        train_loss += loss_wf.item()

    last_loss = train_loss/len(train_loader)
    return last_loss

def eval(valid_loader, model, optimizer, criterion, min_valid_loss, hyperparameters):

    valid_loss = 0.0
    model.eval()
    for data, labels in valid_loader:
        with torch.no_grad():

            data = data.to(device)
            labels = labels.to(device)
            pred = model(data)
            loss_wf = criterion(labels, pred)
            valid_loss += loss_wf.item()

    val_loss = valid_loss/len(valid_loader)

    if val_loss < min_valid_loss:
        min_valid_loss = val_loss
        print('Best model, saving...')
        torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            model_path + hyper(hyperparameters))

    return val_loss, min_valid_loss


def objective(trial):

    filters0 = trial.suggest_int("filters0", 8, 32, step=8)
    filters1 = trial.suggest_int("filters1", filters0, 64, step=8)
    filters2 = trial.suggest_int("filters2", filters1, 128, step=8)
    filters3 = trial.suggest_int("filters3", filters2, 192, step=8)
    filters4 = filters3
    filters5 = filters4
    filters = [filters0, filters1, filters2, filters3, filters4, filters5]
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)

    in_channels = 2
    out_channels = 2
    model = DeepWiener_threechannels(in_channels, out_channels, filters)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*3, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos')

    hyperparameters = [filters0, filters1, filters2, filters3, lr, wd]
    print('hyperparameters:', hyper(hyperparameters), flush=True)

    trainLoss_history = []
    validLoss_history = []
    min_valid_loss = 1e7

    for epoch in range(epochs):

        print('epoch:', epoch)

        sht.reset_timer()
        t0 = time.time()

        train_loss = train(train_loader, model, optimizer, criterion, scheduler)

        t1 = time.time()
        t_sht_train = sht.t_total
        print(f'trained {t1-t0:.1f} s ({len(train_loader)} steps, {(t1-t0)/len(train_loader):.3f} s/step; '
              f'transforms {t_sht_train:.1f} s = {100*t_sht_train/(t1-t0):.0f} %)')

        if(math.isnan(train_loss)):
            return 10000

        sht.reset_timer()
        t2 = time.time()
        valid_loss, min_valid_loss = eval(valid_loader, model, optimizer, criterion, min_valid_loss, hyperparameters)
        t3 = time.time()
        print(f'validated {t3-t2:.1f} s (transforms {sht.t_total:.1f} s)')

        print(f"train loss {train_loss}, valid loss {valid_loss}", flush = True)

        trainLoss_history.append(train_loss)
        validLoss_history.append(valid_loss)


    np.savez(loss_path + hyper(hyperparameters), trainLoss_history, validLoss_history)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return min_valid_loss


if __name__ == '__main__':

    print(f'train: {len(train_loader.dataset)} maps, valid: {len(valid_loader.dataset)} maps, batch {batch_size}, '
          f'epochs {epochs}, sht lmax {sht.lmax} ({sht.method}, nthread {sht.nthread}), factor {map_rescale_factor_q}', flush=True)

    if cfg.get("fixed_params"):
        # short test with fixed hyperparameters (no Optuna study)
        value = objective(optuna.trial.FixedTrial(cfg["fixed_params"]))
        print("min valid loss:", value)
    else:
        study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=cfg["study_db"],
        direction="minimize",
        load_if_exists=True
        )
        # N_TRIALS (environment) overrides cfg["n_trials"]: the study can be split over several jobs
        n_trials = int(os.environ.get("N_TRIALS", cfg["n_trials"]))
        print("Optuna trials in this job:", n_trials, "| trials already in the study:", len(study.trials), flush=True)
        study.optimize(objective, n_trials=n_trials)

        print("Best trial:")
        trial = study.best_trial
        print("value:", trial.value)

        print("Params:")
        for key, value in trial.params.items():
            print("{}:{}".format(key, value))
