#!/bin/env python3

# Copyright 2025 Gustave Eiffel University
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for full license information.

import json
import os
import pickle
from collections import Counter
from pathlib import Path
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm
from typing import Any
import warnings

warnings.filterwarnings("ignore")

col_name_lower_dict = {'dayl(s)': "DL_CM", 'prcp(mm/day)': "P_CM", 'srad(w/m2)': "RAD_CM", 'swe(mm)': "SWE_CM",
                       'tmax(c)': "T_CM_max", 'tmin(c)': "T_CM_min", 'vp(pa)': "VP_CM", 'qobs': "Q_Obs",
                       "pet(mm/day)": "ETP_CM"}


def custom_legend(axe: Any, dict_labels: dict) -> Any:
    """Custom the legend for an axe"""
    line_ = axe.get_legend_handles_labels()[0]
    out_label = [dict_labels[a] for a in axe.get_legend_handles_labels()[1]]
    return line_, out_label


def read_basin_list_from_file(basins_file: str):
    """Read a list of basins from a file """
    return [a.split()[0] for a in open(basins_file, "r").readlines()]


def get_test_period(period_: tuple, df):
    """Get period date from tuple """
    return pd.date_range(period_[0], period_[1], freq=df.index.to_series().diff().mode()[0])


def add_noise_to_force_order(row, alpha: float = None):
    """
    Add a local noise to a ranked feature to force ranking. The value to be added is
    the 90% of the minimum distance divided by the size of the duplicated values.
    The values to be added are shuffled randomly
    """
    if alpha is None:
        alpha = 1e-4
    if row.sum() == 0.:
        new_row = np.linspace(0., 1e-4, len(row))
        np.random.shuffle(new_row)
        row[:] = new_row
    perturbed = row.copy()
    uniq, counts = np.unique(row, return_counts=True)
    duplicated = uniq[counts > 1]
    if len(duplicated) == 0:
        return perturbed
    epsilon_ = max((np.min(np.diff(np.sort(uniq))) * 0.95), alpha)  # Get the minimum allowed distance
    for val in duplicated:
        mask = row == val
        indx = np.where(mask)[0]
        noise_ = np.linspace(0, epsilon_, num=len(indx) + 1, endpoint=True)  # Get a range from zero
        np.random.shuffle(noise_)  # shuffling
        noise_ = np.random.choice(noise_, size=len(indx), replace=False)  # choose needed
        perturbed.iloc[indx] += noise_  # add noise
    return perturbed


def make_rank_for_basin(df_0, period, feats: list = None):
    """Compute rank diagram for features [feats] in dataframe [df_0] using the [period] as TEST period """
    if feats is None:
        feats = df_0.columns
    test_T = get_test_period(period, df_0)
    test_T = test_T[~((test_T.month == 2) & (test_T.day == 29))]
    test_years = list(test_T.year.unique())
    all_years = list(df_0.index.year.unique())
    mbr_years = [y for y in all_years if y not in test_years]
    df_feats = df_0[feats].copy()
    df_class = []
    size_ = None
    for feat_ in feats:
        alpha = 1e-4 if "Q_" in feat_.lower() else 1e-3
        df_base = df_feats.loc[test_T, feat_].to_frame("Obs")
        df_base = df_base[~((df_base.index.month == 2) & (df_base.index.day == 29))]
        size_ = df_base.shape[0]
        mb_x = [df_base]
        for yr in mbr_years:
            new_t = [a.replace(year=yr) for a in test_T]
            try:
                df_new = df_feats.loc[new_t, feat_].to_frame(f"yr_{yr}")
                df_new.index = df_base.index
                mb_x.append(df_new)
            except KeyError as e:
                # print(e)
                continue
        mb_x = pd.concat(mb_x, axis=1)
        mb_x = mb_x.apply(add_noise_to_force_order, alpha=alpha, axis=1)
        stat_ = Counter(mb_x.rank(axis=1, ascending=True).astype(int)["Obs"])
        rk = pd.DataFrame().from_dict(stat_, orient="index").sort_index()
        rk.columns = [f"{feat_}"]
        rk.index.name = "Rank"
        df_class.append(rk)
    return pd.concat(df_class, axis=1) / size_, mb_x


def select_bv_by_class(size: int = 10, how: Any = "last"):
    """
    Select a sub-sample of basin using the pandas.cut() method to make classes. One basin is selected on each class
    by specifying whether it must be the last, the first or a randomly chosen.

    :param size: how many classes to have (equivalent to number of desired basins)
    :param how: either of first (leftmost), last (rightmost), or randomly
    :return: list of selected basins
    """
    with open("all_metrics.p", "rb") as fp:
        metr_lstm = pickle.load(fp)
    lstm_nse = pd.DataFrame().from_dict(metr_lstm["NSE"]["lstm_NSE"], orient="columns")[["ensemble"]]
    lstm_nse.index.name = "index"
    lstm_nse["rank"] = lstm_nse.rank(ascending=False, axis=0, method="first").astype(int)
    lstm_nse["class_rank"] = pd.cut(lstm_nse["rank"].values, bins=size, precision=0)
    grp = lstm_nse.reset_index().sort_values(by="ensemble")[["index", "class_rank"]].groupby("class_rank",
                                                                                             observed=True)
    grp.index = range(1, size + 1)
    selected = list(grp.agg(how).values[:, 0])
    selected.sort()
    return selected


def get_basin_hp_from_string(f: str):
    """
    Get basin and ho from a string in respect to this code naming format.
    _bv and _hp must be present in the said string f
    """
    return f.split("_bv")[1].split("_")[0], int(f.split("_hp")[1].split("_")[0])


def read_prediction(f: str):
    """Read the prediction file with the Date expected in Index"""
    return pd.read_csv(f, sep=";", index_col="Date", parse_dates=["Date"])


def check_eval_period(period: tuple[str, str], hp_max: int = None):
    """
    Check and adapt period for the evaluation. It should cover the whole passed period extended both-ward by the lead time

    :param period: a tuple of start and end date, str format required
    :param hp_max: the lead time max to be considered

    :return: both real period index and adapted one
    """
    if hp_max is None:
        hp_max = 7
    eval_period = pd.date_range(pd.to_datetime(period[0]), pd.to_datetime(period[1]))
    adapted_period = eval_period.append(pd.date_range(eval_period[0], periods=hp_max + 1, freq="-1D")).append(
        pd.date_range(eval_period[-1], periods=hp_max + 1))
    adapted_period = adapted_period.sort_values()
    return eval_period, adapted_period


def make_obs_and_naive(data_dir: str, basins_list: list, hp_list: list[int], period: tuple[str, str]):
    """
    Make observed and naive dataframe from path. Note that three dataframes are returned (observed, naive, obs_naive). The two first
    are in large format, since columns are named with the basins. the naive has an extra column for the lead time (hp). The last
    one is a multi-index joint-dataframe (observed - naive) with index=[basin, hp, Date] instead of [Date] alone.


    :param data_dir: The directory where data are stored like basinID.txt. e.g. 14400000.txt
    :param basins_list: the list of basins to include. e.g. ['14400000', '01250000']
    :param hp_list: the lead times through a list. e.g [1, 3, 7]
    :param period: The evaluation period (or Test period). e.g ("20060131", "20080820")

    :return: df_observed["basin1", "basin2", ...], df_naive["basin1", "basin2", ..., "hp"], df["naive", "observed"]
    """
    # Read data for all specified basins
    df_all = pd.concat(
        [read_prediction(f).Q_Obs.to_frame(a) for f in glob(data_dir + f"/*.txt") for a in basins_list if a in f],
        axis=1)
    _, extended_period = check_eval_period(period=period, hp_max=max(hp_list))
    df_all = df_all.loc[extended_period]
    df_all.index.name = "Date"

    # Make naives
    hp_df = []
    step_dt = df_all.index.to_series().diff().mode()[0]
    for hpx in hp_list:
        temp_hp = df_all.copy()
        temp_hp.index = temp_hp.index + hpx * step_dt
        temp_hp["hp"] = hpx
        hp_df.append(temp_hp)
    hp_df = pd.concat(hp_df, axis=0)

    # Multi-indexing naive dataframe
    naive_long = hp_df.melt(id_vars="hp", var_name="basin", value_name="naive", ignore_index=False)
    naive_long.set_index(["basin", "hp"], append=True, inplace=True)
    naive_long = naive_long.reorder_levels(["basin", "hp", 'Date']).sort_index()

    # Multi-indexing naive dataframe
    obs_long = df_all.melt(var_name="basin", value_name="obs", ignore_index=False)
    obs_long.set_index(["basin"], append=True, inplace=True)
    obs_long = obs_long.reorder_levels(["basin", 'Date']).sort_index()

    # Join the observed and naive on Date and basin levels
    naive_obs_long = naive_long.join(obs_long, on=["basin", "Date"], how="inner").sort_index()
    naive_obs_long = naive_obs_long.loc[~naive_obs_long.index.duplicated(), :]
    naive_obs_long = naive_obs_long[["obs", "naive"]].copy()
    return df_all, hp_df, naive_obs_long


def reformat_clim_predictions(df_: pd.DataFrame, basin: str, hp: int, context: str):
    """
    Reformat the prediction from climatology runs. The dataframe will be multi-indexed in the following order
    [context, basin, hp, Date, year, seed]. The dataframe will be longer with a single column.
    e.g. reformat_clim_predictions(read_prediction(f), *get_basin_hp_from_string(f), context="no_in_mlp")

    :param df_: dataframe with Date as index, and prediction columns name like "yr00_s00"
    :param basin: the basin to be added
    :param hp: the lead time
    :param context: the run context
    :return: transformed dataframe with Index = [context, basin, hp, Date, year, seed] and Columns = [prediction]
    """
    df = df_.filter(like="yr")
    seeds = df.columns.str.extract(r'_s(\d+)', expand=True)[0].astype(np.int8)
    years = df.columns.str.extract(r'yr(\d+)_', expand=True)[0].fillna(-1).astype(np.int8)
    df.columns = pd.MultiIndex.from_arrays([years, seeds], names=["year", "seed"])
    df = df.melt(value_name="prediction", ignore_index=False).reset_index()
    n_vals = len(df)
    df["basin"] = pd.Series([basin] * n_vals)
    df["hp"] = pd.Series([hp] * n_vals)
    df["context"] = pd.Series([context] * n_vals)
    new_ind = ["context", "basin", "hp", "year", "seed", "Date"]
    df = df.set_index(new_ind)
    return df.astype(np.float32)


def get_lstm_predictions(run_dir, which_lstm: str = "lstm", with_static: bool = None, mse: bool = True):
    """
    Get and the predictions from pretrained LSTM.

    :param run_dir: directory for pretrained models. ..../runs
    :param which_lstm: can be one of lstm or ealstm
    :param with_static: indicate if static was used, if False, which_lstm is not used
    :param mse: indicate use_mse or not

    :return : dataframe with (seed, Date) as index, basins in columns for only qsim
    """
    mse_use = 1 if mse is True else 0
    if with_static is None:
        with_static = True
    x_lstm = which_lstm if with_static is True else "lstm_no_static"
    all_p = [f for f in glob(run_dir + f"/*/{x_lstm}_seed*.p") if "hp" not in Path(f).name]
    all_f = []
    for ff in tqdm(all_p):
        case_mse = 1 if json.load(open(str(Path(ff).parent) + "/cfg.json", "r"))["use_mse"] is True else 0
        if case_mse == mse_use:
            with open(ff, "rb") as fp:
                lstm_ = pickle.load(fp)
            pred_i = pd.concat([v.qsim.to_frame(k) for k, v in lstm_.items()], axis=1)
            pred_i.index.name = "Date"
            pred_i["seed"] = int(ff.split("_seed")[-1].split(".p")[0])  # seed_i  #
            pred_i = pred_i.set_index("seed", append=True).reorder_levels(["seed", "Date"])
            all_f.append(pred_i)
    all_f = pd.concat(all_f, axis=0)
    all_f = (all_f.melt(value_name="prediction", ignore_index=False, var_name="basin")
             .set_index("basin", append=True).reset_index("seed").pivot(columns="seed")
             .reorder_levels(["basin", "Date"]).sort_index())
    all_f.columns = all_f.columns.map(lambda x: x[1])
    return all_f


def get_run_sac(file: str):
    """Get the prediction of the sac-sma models from a file

    :param file: file ending with "..._model_output.txt". The column MOD_RUN will be extracted
    """
    bv, seed = Path(file).stem.split("_model")[0].split("_")
    df_ = pd.read_csv(file, sep="\s+")
    df_['Date'] = pd.to_datetime(df_.YR.map(str) + "/" + df_.MNTH.map(str) + "/" + df_.DY.map(str))
    sac_run = df_[["Date", "MOD_RUN"]].set_index("Date")
    sac_run.columns = [seed]
    return sac_run


def get_sma_predictions(run_dir: str, basins_list: list, save: bool = None):
    """
    Get all predictions from the SAC-SMA models. It targets the XXXXXXXX_SS_model_output.txt files where
    XXXXXXXX stands for the 8 digits basin-Id and SS the running number (a.k.a SEEDS). We suggest to set save to
    True for the first runs, since it may take a bit long (~4min) to load.

    :param run_dir: the models dir, it must have the 2 digit regionsID folders (/01 - 18) as child
    :param basins_list: the list of basins to be considered
    :param save: Indicate whether you save the file. Set to False if not desired.
    :return: a dataframe with (basin, Date) as index, [seed] as columns
    """
    model_output_list = [f for f in glob(run_dir + "/*/*model_output.txt") for b in basins_list if b in f]
    all_runs = []
    for basin in tqdm(basins_list):
        seeds_bv_files = [f for f in model_output_list if basin in f]
        seed_pred = pd.concat([get_run_sac(f) for f in seeds_bv_files], axis=1)
        seed_pred["basin"] = f'{basin}'.zfill(8)
        all_runs.append(seed_pred)
    all_runs = (pd.concat(all_runs, axis=0).set_index(["basin"], append=True)
                .reorder_levels(["basin", "Date"]).sort_index())
    if save is not False:
        all_runs.to_parquet("sacsma_outputs.parquet.gzip", compression="GZIP", engine="pyarrow")
    return all_runs


def get_all_pet_from_sac(run_dir_sacsma: str, basins_list: list, save: bool = None):
    """Get the PET from the sac-sma prediction files

    :param run_dir_sacsma: the models dir, it must have the 2 digit regionsID folders (/01 - 18) as children
    :param basins_list: the list of basins to be considered
    :param save: Indicate whether you save the file. Set to False if not desired.
    :return: a dataframe with (Date) as index, [basin] as columns
    """
    if save is None:
        save = False
    model_output_list = [f for f in glob(run_dir_sacsma + "/*/*_05_model_output.txt") for b in basins_list if b in f]
    all_pet = []
    for file in tqdm(model_output_list):
        basin = Path(file).stem.split("_")[0]
        df_ = pd.read_csv(file, sep="\s+")
        df_['Date'] = pd.to_datetime(df_.YR.map(str) + "/" + df_.MNTH.map(str) + "/" + df_.DY.map(str))
        sac_pet = df_[["Date", "PET"]].set_index("Date")
        sac_pet.columns = [basin]
        sac_pet = sac_pet[~sac_pet.index.duplicated(keep="first")]
        all_pet.append(sac_pet)
    all_pet = pd.concat(all_pet, axis=1, join="inner")
    if save is True:
        all_pet.to_csv("pet_sacsma.txt", sep=";", index_label="Date")
    return all_pet


def load_raw_usgs(flow_file):
    """
    Load the observed discharge data from the usgs streamflow

    :param flow_file: file of the flow
    :return: QObs series with date indexed
    """
    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    obs = pd.read_csv(flow_file, sep='\s+', header=None, names=col_names)
    obs['Date'] = pd.to_datetime(obs.Year.map(str) + "/" + obs.Mnth.map(str) + "/" + obs.Day.map(str))
    obs.set_index('Date', inplace=True, drop=True)
    obs = obs[~obs.index.duplicated(keep="first")]
    return obs["QObs"]


def load_raw_forcing_and_area(forcing_file: str):
    """Read the forcing file and return data with formatted date as index, and the corresponding area (m²) """
    df = pd.read_csv(forcing_file, sep='\s+', header=3)
    df.index = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df = df[[c for c in df.columns if c.endswith(")")]]
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")]
    area = float(open(forcing_file, 'r').readlines()[2])
    return df, area


def flow_cfs_to_mmday(flow: float, area_m2: float):
    """Convert flow from cubic feet per second (cfs) into mm a day"""
    return 28316846.592 * flow * 86400 / (area_m2 * 10 ** 6)


def get_flow_and_forcing(flow_and_forcing_dir, forcing_src, basin_id, is_extended: bool = None):
    """
    Get the forcing and the streamflow in the same dataframe

    :param flow_and_forcing_dir: camels_root where usgs_streamflow/ and basin_mean_forcing/ are located
    :param forcing_src: the forcing source to use daymet, nldas, maurer or maurer_extended
    :param basin_id: the 8-digit basin id
    :param is_extended: to ease choice between maurer and maurer_extended

    :return: a full dataframe
    """
    if is_extended is None:
        is_extended = True
    if is_extended is True and "maurer" in forcing_src:
        forcing_src = "maurer_extended"
    forcing_f = glob(flow_and_forcing_dir + f"/basin_mean_forcing/{forcing_src}/*/{basin_id}_*.txt")[0]
    flow_f = glob(flow_and_forcing_dir + f"/usgs_streamflow/*/{basin_id}*.txt")[0]
    forcing_d, area = load_raw_forcing_and_area(forcing_f)
    forcing_d = forcing_d[[c for c in forcing_d.columns if not c.lower().startswith("dayl")]]
    flow_d = load_raw_usgs(flow_f).map(lambda x: flow_cfs_to_mmday(x, area)).to_frame("QObs")

    return pd.concat([forcing_d, flow_d], axis=1, join="inner")


def rename_camels_for_mlp(df: pd.DataFrame, mapper_col: dict = None):
    """Rename the reformatted dataframe for this mlp code"""
    if mapper_col is None:
        mapper_col = col_name_lower_dict
    df_ = df.copy()
    df_.columns = df_.columns.str.lower().map(mapper_col)
    return df_


# Extra-functions
def get_clim_pred_list_mlp(run_dir: str):
    """ Get all climatology runs from orchestrator.

    :param run_dir: Path to the runs, it should have as children /{context}/run_***/ .. where context is like "lstm_in_mlp".
    """
    return glob(run_dir + "/*/*/climatology/proc_s*.csv")


def get_perf_pred_list_mlp(run_dir: str):
    """ Get all climatology runs from orchestrator.

    :param run_dir: Path to the runs, it should have as children /{context}/run_***/ .. where context is like "lstm_in_mlp".
    """
    return glob(run_dir + "/*/*/*test.csv")


def get_context_from_file(f, rund_dir: str):
    """ Get the context run from a file string path"""
    return f.split(rund_dir)[1].strip(os.sep).split(os.sep)[0]


def merge_ground_to_predictions(pred_df_: pd.DataFrame, obs_naive: pd.DataFrame):
    """
    Join the ground data, including the naive values, to the prediction. Join will be done on (basin, hp, Date) index.

    :param pred_df_: the prediction dataframe, multi-indexed and may or not includes hp (lead time) index.
    :param obs_naive: a dataframe that holds the ground values as columns = [observed, naive], hp is required.
    :return: A multi-index dataframe fit to undergo evaluation, columns=[obs, naive, prediction]
    """
    pred_df = pred_df_.copy()
    if "hp" not in pred_df.index.names:
        hp_list = obs_naive.index.get_level_values("hp").unique()
        pred_hp137 = []
        for hp in hp_list:
            temp_pred = pred_df.copy()
            temp_pred["hp"] = hp
            pred_hp137.append(temp_pred)
            del temp_pred
        pred_df = pd.concat(pred_hp137, axis=0).set_index("hp", append=True).reorder_levels(
            ["context", "basin", "hp", "year", "seed", "Date"]).sort_index()
    pred_df = pred_df.join(obs_naive, on=["basin", "hp", "Date"], how="inner")
    pred_df = pred_df[["obs", "naive", "prediction"]].astype(np.float32)
    return pred_df


def make_spread_skill_ratio(df: pd.DataFrame, obs: pd.DataFrame = None, member_names=None):
    """
    Make spread skill ratio. This ratio compares the variance of an ensemble to that of its error using RMSE. Use the
    root of the mean variance instead of the standard deviation itself (see the Jensen principle) to maintain
    consistency for the RMSE comparison. This should bring a better idea of the ensemble calibration error.

    :param df: a multi index dataframe, with prediction, or including obs
    :param obs: the true observed values if not in df
    :param member_names: the members of the multi-index levels that built the ensembles

    :return : spread, skill, spread_skill_ratio
    """
    if member_names is None:
        member_names = ["year", "seed"]
    id_index = [n for n in df.index.names if n not in member_names]

    if obs is None:
        obs = df["obs"].groupby(id_index).mean().to_frame("obs").sort_index()
    obs.columns = ["obs"]

    var_ens = df["prediction"].groupby(id_index).var().to_frame("var").sort_index()
    spread_ = np.sqrt(var_ens.groupby(["context", "hp", "basin"]).mean())  # root of the mean (on date) of the variance

    ens_mean = df["prediction"].groupby(id_index).mean().to_frame("mean").sort_index()
    mean_n_obs = ens_mean.join(obs, on=obs.index.names, how="inner")

    error_ = (mean_n_obs["mean"] - mean_n_obs["obs"]).to_frame("error")
    skill_ = np.sqrt((error_ ** 2).groupby(["context", "hp", "basin"]).mean())  # root mean squared error

    spread_skill_ratio = (spread_["var"] / skill_["error"]).to_frame("ssr")
    return spread_, skill_, spread_skill_ratio
