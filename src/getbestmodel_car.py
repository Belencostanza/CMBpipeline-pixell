import numpy as np
import torch
import sys
import time
import optuna
from pixell import enmap

from config_loader import load_config
cfg = load_config()                                 # also puts cmbpipeline_source on sys.path

from network_2d import DeepWiener_threechannels    # ../CMBpipeline/source (read-only)
from sht_torch import CarSHT

# Same as CMBpipeline's getbestmodel_beam.py: best trial of the Optuna study, its saved
# model, prediction on the first nmaps validation maps, saved / map_rescale_factor [muK].
# Also saved: the band-limited Wiener-filtered map Y A (prediction), the only part the
# curved-sky loss constrains (losses_car.CarLossJ3.wf_map), and the trial summary.

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)

#############################################READ CONFIG#################################################

map_rescale_factor_q = cfg["map_rescale_factor"]
name_valid           = cfg["name_valid"]
model_path           = cfg["model_path"]
result_path          = cfg["result_path"]
study_name           = cfg["study_name"]
nmaps                = 10

study = optuna.load_study(study_name=study_name, storage=cfg["study_db"])

trial = study.best_trial

print("\nTrial number {}".format(trial.number), flush=True)
print("Value: %.5e"%trial.value, flush=True)
print(" Params: ", flush=True)
for key, value in trial.params.items():
    print("    {}: {}".format(key, value), flush=True)

filters0 = trial.params["filters0"]
filters1 = trial.params["filters1"]
filters2 = trial.params["filters2"]
filters3 = trial.params["filters3"]
filters4 = filters3
filters5 = filters4
lr = trial.params["lr"]
wd = trial.params["wd"]

hyperparameters = [filters0, filters1, filters2, filters3, lr, wd]
filters = [filters0, filters1, filters2, filters3, filters4, filters5]

def hyper(hyperparameters):
    # same names as training_car.hyper (CMBpipeline's utilities.hyper shifts the labels)
    return ("n_filters0_" + str(hyperparameters[0]) + "_n_filters1_" + str(hyperparameters[1]) +
            "_n_filters2_" + str(hyperparameters[2]) + "_n_filters3_" + str(hyperparameters[3]) +
            "_lr_" + "{:.3e}".format(hyperparameters[4]) + "_wd_" + "{:.3e}".format(hyperparameters[5]))

resultsname = hyper(hyperparameters)

######################### DATASET #############################################

test = torch.load(name_valid, mmap=True)[0:nmaps]
qobs = (test[:,0:1,:,:]).to(device)
uobs = (test[:,1:2,:,:]).to(device)
mask = (test[:,2:3,:,:]).to(device)
inho = (test[:,3:4,:,:]).to(device)
inho_norm = inho

qobs_norm = (qobs - qobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q
uobs_norm = (uobs - uobs.mean(dim=(2, 3), keepdim=True))*map_rescale_factor_q

obs_norm = torch.cat((qobs_norm, uobs_norm, mask, inho_norm), dim=1)
print(obs_norm.shape)

####################################### inference ##########################################

in_channels = 2
out_channels = 2
model = DeepWiener_threechannels(in_channels, out_channels, filters)
model.load_state_dict(torch.load(model_path + resultsname, map_location=torch.device('cpu'))['model_state_dict'])
model.to(device)

model.eval()
t1 = time.time()
with torch.no_grad():
    pred = torch.cat([model(obs_norm[i:i+1]) for i in range(nmaps)])
t2 = time.time()
print('inference', t2-t1, 's')

pred = pred.cpu()/map_rescale_factor_q                       # muK, as CMBpipeline
torch.save(pred, result_path)

# band-limited Wiener-filtered map Y A (prediction), l <= sht_lmax
mask_car = enmap.read_map(cfg["mask_car_path"])
sht = CarSHT(mask_car.shape, mask_car.wcs, lmax=cfg["sht_lmax"], method=cfg["sht_method"], nthread=cfg["sht_nthread"])
pred_bl = torch.from_numpy(sht.Y(sht.A(pred.numpy().astype(np.float64))).astype(np.float32))
torch.save(pred_bl, result_path.replace('.pt', '_bandlimited.pt'))

np.savez(result_path.replace('.pt', '_trial.npz'), number=trial.number, value=trial.value,
         filters=np.array(filters), lr=lr, wd=wd, model=resultsname, nmaps=nmaps,
         map_rescale_factor=map_rescale_factor_q)
print('saved', result_path, 'and', result_path.replace('.pt', '_bandlimited.pt'))
