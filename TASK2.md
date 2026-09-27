# Task 2: Training dataset with pixell maps

Goal: generate the 2D training dataset for `DeepWiener_threechannels`
(`../CMBpipeline/network_2d.py`) using Pipeline B from Task 1 (CAR cut-out maps, 1110 x 324, as in 6b Section of `01_pixell_validation.ipynb`), with the SO
"rect" configuration.

- Write `make_dataset.py` and `run_dataset.py` in CMBpipeline-pixell, following
  the structure and style of the equivalent files in `../CMBpipeline/`.
- Write a `config.dict` in the same format as CMBpipeline's. Keep the same
  parameters (number of maps, noise, beam, mask, lmax, etc.) unless Pipeline B
  requires a change; list every difference.
- Check in `network_2d.py` what the three input channels and output are, and
  what map shape and normalization the network expects. Make the dataset match
  exactly.
- Data output path: a new project directory on the IAS cluster
  (bcostanza@ssh.sns.ias.edu), set in `config.dict`. Do not overwrite any
  existing CMBpipeline data.

Steps:
1. Propose the plan (file structure, config parameters, output format, output
   path on the cluster) and STOP.
2. Implement, and run a small local test (a few maps) that checks shapes,
   channels and values against a CMBpipeline map. STOP. 
3. write the readme with the main ideas of TASK1.md and TASK2.md
  
4. commit and push the CMBpipeline-pixell to the github respository with the same name
5. Only after my approval: set up the cluster run.