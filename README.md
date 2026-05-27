# AI-Operational_HydroForecast

[! Attention]
This repository is released for scientific reproducibility purposes. It is not intended for use in training, fine-tuning, or evaluation of large language models or other generative AI systems.

***
## 1. Description
This project proposes to forecast discharges in a way that reflects what can be done in a real time operational forecasting framework. It is essentially based on discharge assimilation from either recent updated measures or from other models simulations. 
It evaluates both a very idealistic forecasting approach and an operational one. The first is what is commonly called "Perfect Meteorological Forecast (PMF)", while the second can be termed as "Ensemble Meteorological Forecast (EMF)".
The PMF assumes that meteorological forecasts (rainfall, temperature, PET, ...) are perfectly known from the future. The EMF is more operational and several techniques could be used to prepare these forecast. In this work,
we consider three options: (1) historical meteorological records on matching dates through years, as a poor's man forecast or commonly known as climatology; (2) using the hindcast products provided by the ECMWF platform;
 or (3) the forecast archives. 
In all these cases, for a given date *t0* and a lead time *hp>0*, the vector of the forecast input X is X[t0:hp] or X[t0:t0+hp]. 
The forecast members size is given by the number of the year from the climatology-case, or the provided size of the forecast products.
The added value of various discharge assimilation strategies is assessed for all these approaches through the present study, and include a comparison on two benchmark models without DA strategies. See [Saint Fleur et al. (2025)](https://doi.org/10.5194/egusphere-2025-4244) for more details.
The two concerned benchmark models are [a regional LSTM from Kratzert et al. (2019)](https://doi.org/10.5194/hess-23-5089-2019) and [a basin-wise SAC-SMA model from Newman et al. (2017)](https://doi.org/10.1175/JHM-D-16-0284.1). 
The related models have been slightly adapted to evaluate our experiments without re-calibration or re-training. These adapted models can be found in [lstm_FK_vUGE](https://github.com/bob-saintfleur/ealstm_regional_modeling/tree/hydro_uge) and [SAC-SMA_vUGE](https://github.com/bob-saintfleur/SACSMA-SNOW17/tree/hydro_uge) 
including necessary instructions for the reproducibility of the results.

***
# 2. Access the code

Before any use of this code, make sur you get the [dataset](https://zenodo.org/records/19825677) downloaded first in your system. Then adapt all `Y:\repo_egu24\data_paper` found in this readme accordingly.

## 2.1 Installation
Get the model on your system by cloning the current directory on the branch `main` or download a zip version.

***

## 2.2. Get the LSTM and the SAC-SMA model
Users may get the adapted versions released on Zenodo of the [SACSMA](https://doi.org/10.5281/zenodo.20379006) and [LSTM](https://doi.org/10.5281/zenodo.20379019) or get the git versions
 [LSTM](https://github.com/bob-saintfleur/ealstm_regional_modeling/tree/hydro_uge) and the [SAC-SMA](https://github.com/bob-saintfleur/SACSMA-SNOW17/tree/hydro_uge), or :

````
git clone https://github.com/bob-saintfleur/SACSMA-SNOW17.git -b hydro_uge
````

````
git clone https://github.com/bob-saintfleur/ealstm_regional_modeling.git -b hydro_uge
````

Please follow the instructions provided in their README files, which have been extended to include the adaptation to the present study. For any run from scratch, users should run these models first to feed the actual MLP model. 

---

# 3. Code indication

## 3.1 Folders
For an easy run, make sure the following directories and data are available. Feel free to use the example provided alongside this repository
- `data/` or `data_paper/data/camels[??]/`
  - `all_sim_obs_lstm/*`
    - `basin01.txt`
    - `0123456789.txt`
    - ..
  - `basins_list`
  - `basins_56`
  - `basins_2.txt`

A LOG folder is created at the very first run, it can be cleaned or deleted, but will still be re-created. It stores the log of your runs every hour by default.
The frequency of its creation can be adapted in the utils/logger.py file.

## 3.2 Running options
Four run modes are possible: `search, climatology, hindcast, apply `

The *`search`* is used to train and optimize a specific model following a set of running options. These running options are mainly related to the basin, the lead time (`hp`), number of random 
initialization (nb_seeds), discharge assimilation or not, assimilation of other model etc... 
You shall pass an argument for the data directory using the keyword (`--data_paper_path`) alongside a `--camels_context` argument. Leave both unused for quick local test. If these
two arguments have been used for training, they should be systematically called back during other evaluation

The *`apply`* mode is to re-evaluate a pre-trained model. Its main arguments include the path to the models (`--run_dir`), and a filter to select only a subset of existing models (--discr_model)

The *`climatology`*, *`hindcast`* and `realtime` run modes behave almost the same as `apply` mode, except their repetitive philosophy.
Their main arguments are the path of the model (`--model_dir`), the period [`OPTIONAL`] to consider (`--period`), an argument to speed up the runs (`--n_sub`). 
By default, all trained models found in **model_dir** , filtered on `basins_list` or basins from passed `basins_file basins_56` argument are evaluated sequentially. **`discr_model`** argument can be used to filter on the latter.

In all cases, for contexts that imply the integration of other benchmark models (`--other_model_use STR`), the basins to be used should have their Ensemble-based pre-run available first in a folder named `climato_bm/, hindcast_bm/, or realtime_bm/`. 
Further filtering on models list can be done using the *`--discr_model STR`* argument. E.g `--discr_model _bv01*_hp3`


***
## 3.3 Arguments for training
## *search* run mode (to train and optimize a certain model)
To run the mlp on a search mode.

Required arguments:
- `main.py `: the script that call to train, by default
- `run_dir` : path to store the trained model (only for MLP)
- `nb_seeds `: number of random initialization (seed) to use
- `param_grid`: a grid of options (rather than parameter or hyperparameter), it may include `hp=1,2 use_future=True target_as_input=True`
- `basins_file` : filename for basins-list file other than the default basins_list, `basins_list` can be also used
- `camels_context`: one of `us, fr OR test`, do not use this argument for quick test using the example content in `./data/*`
- `data_paper_path` : Indicate the path of the current data paper, or do not ue
- `other_model_use` : this arguments control the proposed data assimilation approaches (or context), 
the expected values are `(no_in_mlp, lstm_in_mlp, sma_in_mlp, predict_lstm_error, predict_sma_error)`


### General syntax

````
python main.py --nb_seeds VALUE --param_grid hp=INT,INT,INT use_future=True target_as_input=True --run_dir DIR --other_model_use OTHER_MODEL_USE --basins_file FILE_NAME --data_paper_path PATH_TO_DATA_PAPER --train_period START END --test_period START END
````

### For this data paper, do not use basins_file argument, unless expressly needed
`python main.py --camels_context us/fr --nb_seeds 20 --param_grid hp=1,2,3,4,5,6,7 --run_dir RUN_DIR --other_model_use OTHER_MODEL_USE --data_paper_path PATH_TO_DATA_PAPER `

### Examples of training launch
- QUICK TEST: The followings are equivalent

`python main.py --nb_seeds 3 --param_grid hp=1,2 --run_dir test_01 --other_model_use no_in_mlp --basins_file basins_2.txt` 

<!--
`python main.py --nb_seeds 3 --param_grid hp=1,2 --run_dir test_01 --other_model_use no_in_mlp --basins_file basins_2.txt --data_root ./data`
`python main.py --nb_seeds 3 --param_grid hp=1,2 --run_dir test_01 --other_model_use no_in_mlp --basins_file basins_2.txt --data_path ./data/all_sim_obs_lstm`
-->

- On CAMELS-FR

`python main.py --camels_context fr --nb_seeds 3 --param_grid hp=1,2 --run_dir test_fr --other_model_use no_in_mlp --basins_file basins_2.txt --data_paper_path Y:\repo_egu24\data_paper`

- On CAMELS-US

`python main.py --camels_context us --nb_seeds 3 --param_grid hp=1,2 --run_dir test_us --other_model_use no_in_mlp --basins_file basins_2.txt --data_paper_path Y:\repo_egu24\data_paper`


## 3.4 Arguments for Ensemble-based evaluation mode (climatology, hindcast, realtime)
Required arguments are:
- run_multi.py : the script
- run_mode : `climatology, hindcast or realtime`
- camels_context : `us, fr, test`
- model_dir : path to the pre-trained models (must have children names start with **/run_mlp...**)
- data_paper_path: path of the data_paper if camels_context != test
- period[optional] : Evaluation period
- n_sub[optional] : subdivide the period into n_sub subs and run them in parallel according available CPU number
- discr_model[optional] : a string to fiter on model list, very useful to filter on basins list


### 3.4.1 General syntax
- Minimal
````
python multi_run.py --run_mode MODE --model_dir PATH_TO_MODELS --camels_context STR --data_paper_path PATH_TO_DATA_PAPER --period START END 
````

- Advanced (cut the period, filter models). It may fail if data_root does not comply wit expected structure
````
python multi_run.py .... --n_sub INT --discr_model STR
````

- QUICK TEST

`python multi_run.py --model_dir test_0/no_in_mlp --run_mode hindcast --camels_context test --n_sub 10 --period 19901001 19901112
`

- QUICK CAMELS-FR

`python multi_run.py --model_dir test_fr/no_in_mlp --run_mode hindcast --camels_context fr --n_sub 10 --data_paper_path Y:\repo_egu24\data_paper --basins_file basins_56`

- QUICK CAMELS-US

`python multi_run.py --model_dir test_0/no_in_mlp --run_mode hindcast --camels_context us --n_sub 10 --data_paper_path Y:\repo_egu24\data_paper --basins_file basins_56`


### 3.4.2 As in this paper
The Ensemble-based evaluation directly filters basins using the basins_56 file, if user prefers other lists, 
another value for `--basins_file` should be expressly passed
You will need to iterate also over the three modes 

### 3.4.3 Simple evaluation

- Apply mode (Syntax : python `main.py --run_mode apply --run_dir RUN_DIR --discr_model hp1 --camels_context STR --data_paper_path STR` )

`python main.py --run_mode apply --run_dir test_0/no_in_mlp`


- On CAMELS-FR

`python main.py --run_mode apply --run_dir runs_fr/no_in_mlp --camels_context fr --data_paper_path Y:\repo_egu24\data_paper`


- On CAMELS-US

`python main.py --run_mode apply --run_dir runs_us/no_in_mlp --camels_context us --data_paper_path Y:\repo_egu24\data_paper`
***

# 4. CMD platform advices
## 4.1 Via uv
[NOTE] We strongly recommend to use uv for the training an evaluations

First, make sur you have uv installed on your system, then prepend the command with  `uv run python ......`  in an adapted terminal such as "powershell"

````
uv run python main.py ...
````
## 4.2. Via IDE cmd line
Make sur a proper environment is installed, you can see the main used libraries in the requirements.txt file, and install them as follows.
Note that, you may need python >=3.9 for these runs

```` 
pip install -r requirements.txt
````

After activation of your virtual environment, you should be able to launch the runs the above examples in your terminal (..> ).

````
python main.py --nb_seeds 3 ...
````
***
# 5. Running specificity for this paper (check the RUN_ME notebook)
## 5.1 Datasets (CAMELS_US and CAMELS_FR)
Two datasets are used, one from CAMELS-US, one from CAMELS-FR. The overall running options remain the same for both datasets, with the following exceptions:

[ !! CAUTION /!\ ] Please pay attention to these differences in running options
- CAMELS_US
  - run_mode : `search, climatology, hindcast, apply`
  - other_model_use: `no_in_mlp, lstm_in_mlp, sma_in_mlp, predict_lstm_error, predict_sma_error`

- CAMELS_FR
  - run_mode :`search, climatology, hindcast, realtime, apply`
  - other_model_use : `no_in_mlp`

In both cases:
- use `--basins_file basins_2.txt` for quick test, you may need to create your own `basins_2.txt` file
- use `--basins_file basins_56` for ensemble-based evaluation, can be any basins_list file too
- use `--basins_file basins_list` or LEAVE UN-USED for full basins list training

- use `--nb_seeds 20` for identical seed size

***
# 6 . PRE- and POST-PROCESSING
Notebooks for the data preprocessing and results postprocessing are provided in the `./post_pprocess_v2` folder.

## 6.1. Pre-process
Two options:
- Use the data provided in this related data_paper
- Preparing your own data, you may find an example in the *prepare_data_paper.ipynb*, then jump to the models specificity to get them running

## 6.2. Run
Run your models in respects of their respective running philosophy and instructions. You should start with the benchmark models first.
But, for discovering of the MLP, start with the argument (`--other_model_use no_in_mlp`) or leave unused

## 6.3. Post-process
Two options

### 6.3.1 Use the provided runs
The processed runs are stored in `~/data_paper/processed/[fr, us]`
Three notebooks are provided for post-processing:(A) `post_process_v2/postp_outputs_paper_US_FR.ipynb`, and (B) `post_process_v2/make_plots_paper_US_FR.ipynb`
 - A: use the pre-formatted of the models outputs found in `~/data_paper/processed/[fr, us]`, and compute the metrics that will be saved in the path you indicate
 - B: using the pre-computed metrics found in `~/data_paper/metrics_grouped/[fr, us]`,, make the plots provided in this paper

In Both cases, instructions are provided to checking and/or performing the complete process. Refer to the *TODO* flags provided. 
The metrics are computed in two steps: (1) by context or other_model_use, and (2) grouped for all contexts, with the term `_grouped` inserted before the `/[fr or us]`.

[Note]: In the previous version, the runs concerned only CAMELS-US, in the present, CAMELS-FR is in too. Therefore, changes have been integrated in this way.

### 6.3.2. Use your own runs
Before you use the two indicated notebooks above, you should make sure your predictions are available. Some indications are provided below. However, 
be aware that in the current state of the present script, it may take very long to replicate all the runs

When you launch a training (always with the search mode first), your predictions are saved in the indicated folder as
`[Your_RUN_DIR]/[OTHER_MODEL_USE]/run***_bvBBBBBBB_hpX_DATEHHH/`. This folder is replicated for each `basin` and each `hp` and each `OTHER_MODEL_USE`. The evaluation mode `apply`
is directly performed, but the ensemble-based `hindcast, climatology, realtime` modes need to be called afterward. Once done, the outputs for the ensemble-based modes
are stored in a sub-folder with the same mode name `../run**/climatology`,`../run**/hindcast`, `../run**/realtime`, or `../run**/evaluate_on_best` if `apply` was called.
These files are `proc_seeds_[BASIN]_hp[1-7].csv`, except for the default evaluation which is in `prediction_on_[NNN]seeds_test.csv`.

These files need to be reformatted from one-file-per-basin-per-seed-per-hp to:
 - 1. A multi index dataframe with One-file-per-hp for all basin, where multi-index should be `(context, basin, hp, year, seed, Date)` and the column will be `(prediction)`.
  Save it in a `FILE.parquet.gzip` for faster processing, with:
   - context (stands for OTHER_MODEL_USE): `no_in_mlp, lstm_in_mlp, sma_in_mlp, predict_lstm_error, predict_sma_error`
   - basin: 8-digit string ID of basin
   - hp: integer of lead time
   - seed: integer of the number of the seeds [1 to N]
   - year: Number of the member [1 to M], and -1 in the case of the deterministic (perfect) case
   - Date: yyyy-mm-dd date format
   - FILE: 
     - climatology : `[OTHER_MODEL_USE]_hp[1-7]_CLIM56.parquet.gzip`
     - hindcast: `[OTHER_MODEL_USE]_hp[1-7]_HIND56.parquet.gzip`
     - hindcast: `[OTHER_MODEL_USE]_hp[1-7]_REAL56.parquet.gzip`
     - perfect or deterministic: `[OTHER_MODEL_USE]_hp[1-7]_PERF531.parquet.gzip`
   - save like in : `~/data_paper/processed/[us,fr]/FILE.parquet.gzip `

After successful runs, `section 6.3.1` can be addressed
***

## Support
For any technical questions, please contact `bob.saint-fleur@univ-eiffel.fr`, or `besaintfleur@gmail.com`

## Roadmap
As future step in that projects, it is intended to perform a regional training and provide a real time forecasting functionality 

## Authors and acknowledgment
We want to thank the following contributor:
- Gustave Eiffel University (Université Gustave Eiffel) and aQuasys Company for their collaboration which led to the A3P project funded by Bpifrance and the LAbCom AiQua funded by the ANR.,
- GERS-EE (Soon within Telluris UMR) laboratory for hosting the work
- Eric GAUME for supervising this scientific work
- Florian Surmont for contributing to the adaptation of the SAC-SMA code and the post-processing

## License
This code is distributed under the European Union Public License 1.2
[License](https://interoperable-europe.ec.europa.eu/sites/default/files/custom-page/attachment/2020-03/EUPL-1.2%20EN.txt)

Copyright 2025 Gustave Eiffel University, France

This repository is released primarily for scientific reproducibility purpose, it may not be used for automated code interpretation by generative AI systems.

### How to cite

If you use this code scientific results, please cite:
```bibtex
@article{SaintFleur2026,
  author={Bob E. Saint-Fleur, Eric Gaume, Florian Surmont, Nicolas Akil, and Dominique Theriez},
  title={Testing discharge assimilation strategies to enhance short-range AI-based operational rainfall-runoff forecasts},
  year={2026},
  journal={HESS},
  doi={doi.org/10.5194/egusphere-2025-4244}
}
```

For software reuse, please cite:
```bibtex
@software{ai_operational_hydroforecast,
  author={Bob E. Saint Fleur},
  title={AI-based models for operational discharge forecasting accompanying "Testing discharge assimilation strategies 
  to enhance short-range AI-based operational rainfall-runoff forecasts"},
  year={2026},
  doi={doi.org/10.5194/egusphere-2025-4244}
}
```

## Project status
This project was published in its present state as tied to the [paper](doi.org/10.5194/egusphere-2025-4244) built from research experiments. 
We then acknowledge that there may be rooms for coding optimization, some functions may look unnecessary, or simplified, etc...
We have stopped cleaning it to make sure the same results are found when running by any other user.
