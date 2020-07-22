# seti-nb-localization
Dataset generation and ML scripts for localization of narrow-band signals, as described in Brzycki et al. 2020 (submitted). 


### Brief descriptions of included files

`create_dataset.py`: Dataset generation script, produces entirely synthetic data frames with ideal chi-squared background noise and constant intensity narrow-band signals. Creates both one and two signal datasets.

`train_cnn.py`: Contains all ML-related code, including model architectures, custom data generators, and training/testing routines. Accepts command-line arguments to facilitate multiple experiments.

`run_training.sh`: Bash script executing `train_cnn.py` for the full set of experiments, as they appear in the paper. 

`Frame_generation.ipynb`: Jupyter notebook illustrating some basic data frame generation using `setigen`.

`RMSE_figures.ipynb`: Jupyter notebook producing the RMSE plots found in the paper, using the output from test predictions via `train_cnn.py`.
