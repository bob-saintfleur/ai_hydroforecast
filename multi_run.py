# Copyright 2025 Gustave Eiffel University
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for full license information.

# !/bin/env python3

# --model_dir run_torch/predict_lstm_error --period 20071001,20071030 --n_sub 4 --discr_model hp1_20260131
# py multi_run.py--model_dir test_old/no_in_mlp --period 19901001,19910930 --discr_model _bv01022500 --run_mode climatology --n_sub 8 --max_ensemble 10

import os
import json
import sys
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import pandas as pd
from utils.launch_climatology import run_clim, run_hindcast
from utils import job_multi
from utils.args_getter import get_clim_args
from utils.logger import logger


def add_i_to_ij_ensemble_df(data_i, data_i_j, col_pattern="yr"):
    """
    Add member i of dataframe A to multi member ij of dataframe B
    """
    check_ptrn = [c for c in data_i.columns if col_pattern.lower() in c.lower()]
    c_ind = data_i.index.intersection(data_i_j.index)
    m_i = [c for c in data_i.columns if col_pattern in c]
    final_ = data_i_j.loc[c_ind].copy()
    for c_i in m_i:
        c_ij = [c for c in data_i_j.columns if c.startswith(c_i)]
        final_[c_ij] = data_i_j.loc[c_ind][c_ij].add(data_i.loc[c_ind][c_i].values, axis=0)
    return final_


def clean_proc_seed(md_path: str, clim_seed: pd.DataFrame = None, mode: str = None):
    """
    Clean output of seed and climatology before saving.

    :param md_path: path of the running model
    :param clim_seed: dataframe of combined seed and climatology runs
    :param mode: indicate if climatology or hindcast
    """
    if mode is None:
        mode = "climatology"
    d_use = json.load(open(md_path + "/data_use_info.json", "r"))
    g_cfg = json.load(open(md_path + "/config.json", "r"))["global_setting"]
    o_m_u = g_cfg["other_model_use"]
    basin = g_cfg["basin"]
    d_path = g_cfg["dataPath"]
    hp = g_cfg["hp"]
    data_raw = pd.read_csv(d_path + f"/{basin}.txt", sep=";", index_col=0, parse_dates=[0])
    clim_seed = clim_seed.loc[:, ~clim_seed.columns.str.contains("yref")]
    clim_cor = clim_seed.copy()
    if "m" in clim_cor.columns[2].lower():
        clim_cor.columns = [c.replace("m", "yr") for c in clim_cor.columns]


    def read_omu_clim(other_model_u):
        """Load climatology or hindcast of other model use """
        omu_ = "sacsma" if "sma" in other_model_u else ("lstm" if "lstm" in other_model_u else None)
        path_c_bm = "climato_bm" if "climato" in mode else f"{mode}_bm" # "hind_bm" if "hind" in mode else "climato_bm"
        clim_b = pd.read_csv(str(Path(d_path).parent) + f"/{path_c_bm}/{omu_}/hp{hp}/{basin}.csv", sep=";",
                             index_col="Date", parse_dates=["Date"])
        return clim_b

    if "date_now" in clim_seed.index.name.lower():
        n_ind = pd.to_datetime(clim_seed.index)
        # n_ind = n_ind + data_raw.index.to_series().diff().mode()[0] * hp
        n_ind = n_ind + timedelta(days=hp)

        if "error" not in o_m_u:
            if d_use["dta_use"]:
                clim_cor = clim_seed.add(data_raw.Q_Obs.loc[clim_seed.index].values, axis=0)
                clim_cor.index = n_ind
                clim_cor.index.name = "Date"
        else:
            clim_bm = read_omu_clim(o_m_u)
            if "m" in clim_bm.columns[2].lower():
                clim_bm.columns = [c.replace("m", "yr") for c in clim_bm.columns]

            clim_cor.index = n_ind
            clim_cor.index.name = "Date"

            # Correction with ensemble forecast
            clim_cor = add_i_to_ij_ensemble_df(clim_bm, clim_cor)

    base_ = data_raw.Q_Obs.to_frame("y_obs")
    base_["y_naive"] = base_[["y_obs"]].shift(hp, axis=0).fillna(method="bfill")
    clim_cor = pd.concat([base_.loc[clim_cor.index], clim_cor], axis=1)
    clim_cor[clim_cor < 0.] = 0.
    return clim_cor.astype("float32")


def run_parallel_sub_dates(func_, arg0=None):
    """ Run func_ separately on dates. The dates are divided and ran separately on replicated config"""
    if arg0 is None:
        arg0 = get_clim_args()
    arg1 = arg0.copy()
    mode=arg0["run_mode"]
    list_arg = []
    arg1.pop("sub_dates")
    arg1.update({"n_sub": 1})
    for period in arg0["sub_dates"]:
        ge_ = arg1.copy()
        ge_.update({"period": period})
        list_arg.append(ge_)

    # put function and its arguments in a list
    list_task = [{"func": func_, "tasks": [dict(arg_user=cfgx) for cfgx in list_arg]}]
    results = job_multi.par_proc(list_task)
    results = [a for a in results if list(a.values())[0] is not None]
    # get, bind and save results to specified path
    md_path = list(set([results[i][func_.__name__][-1] for i in range(len(results))]))[0]
    id_file = md_path.split("_bv")[-1].split("_")[0] + "_hp" + md_path.split("_hp")[-1].split("_")[0]
    # path_to = f"{md_path}/climatology" if arg0["path_to"] is None else arg0["path_to"]
    path_to = f"{md_path}/{mode}" if arg0["path_to"] is None else arg0["path_to"]
    os.makedirs(path_to, exist_ok=True)
    proc_seed = pd.concat([results[i][func_.__name__][0] for i in range(len(results))]).sort_values(by="Date_now")
    # proc_seed = clean_proc_seed(md_path, proc_seed, mode="climatology")
    proc_seed = clean_proc_seed(md_path, proc_seed, mode=mode)
    proc_seed.to_csv(rf"{path_to}/proc_seeds_{id_file}.csv", sep=";")
    logger.info(f"Results path : {path_to}/")


def launch_ensemble_eval():
    """
    Launch on several basins or all models, or filter with discr model
    """
    inputs = get_clim_args()
    mode_ = inputs['run_mode']
    logger.info(f"Mode run: {mode_.upper()}")
    list_mdx = glob(inputs["model_dir"] + "/run_*")
    logger.info(f"Models DIRECTORY: {inputs['model_dir']}")
    logger.info(f"Period: {inputs['period']}")
    logger.info(f"N Sub-Period: {inputs['n_sub']}")

    if inputs["basin_list"]:
        list_mdx = [a for a in list_mdx for b in inputs["basin_list"] if b in a]
    else:
        basins_file = inputs.get("basins_file", "basins_56")
        bv_list = [a.split()[0] for a in open(str(inputs["data_root"])+"/"+basins_file).readlines()]
        list_mdx = [a for a in list_mdx for b in bv_list if b in a]

    if inputs["discr_model"]:
        list_mdx = [a for a in list_mdx if inputs['discr_model'] in a]
    list_str = [a.split(inputs['model_base'] + "_")[-1] for a in list_mdx]
    logger.info(f"Number of models : {len(list_mdx)}")
    if len(list_str) > 0:
        for i, mds in enumerate(list_str):
            inputs.update({"discr_model": mds})
            try:
                run_parallel_sub_dates(run_hindcast if mode_ in {"hindcast", "realtime"} else run_clim, inputs)
            except Exception as e:
                print(f"Line crash with {e}")
                pass
        return
    else:
        msg_ = (f'No MODEL PATH like {inputs["model_dir"]}/run*{inputs["discr_model"] or ""}* or with '
                f'any BASINS in {inputs["basin_list"] or []}. Make sure expected PRE-TRAINED models EXIST')
        logger.warning(msg=msg_)


if __name__ == '__main__':
    logger.info('-' * 35)
    start = datetime.now()
    logger.info("Started")
    launch_ensemble_eval()
    now = datetime.now()
    logger.info(f'Finished. Duration = {now - start}')
    print("Done !")
