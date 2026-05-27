#!/bin/env python3
from glob import glob
import json
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datatools.climatic_from_data import ClimaticScenarios
from datatools.read_data import import_eval_data
from utils.base_model import SklearnModels
from utils.logger import logger
from utils.hindcast_utils import get_hindcast_dict


# def run_clim_sub(models_fold: str,
#                  model_str: str,
#                  data_fold: str = None,
#                  model_base="mlp_skl",
#                  ref_date=(2004, 9, 1),
#                  month: list = None,
#                  day: list = None,
#                  key_: str = "rainfall",
#                  strategy: str = None, n_members: int = 10
#                  ):
#     """
#     This module runs the climatic scenarios with the following parameters.
#
#     :param models_fold: parent folder for model(s) ran under a particular condition
#     :param model_base: the module base for the model, by default it is set to 'mlp_skl'
#     :param ref_date: the reference date to be considered as target date
#     :param model_str: pre-trained model to be tested, in some cases, it refers to the basin name or the run_time
#     :param data_fold: the historic data folder
#     :param strategy: the weather strategy. It can be a string indicating a [drought, normal, wet, random]
#     :param month: the month to watch [1-12]
#     :param day: the day to watch out (no 29 for february)
#     :param key_: the key feature on which assumption is applied
#     :param n_members: expected size of the forecasting member
#     :return: all_res, all_seed, strategy, target, key_
#     """
#     # use some default date if not provided
#     if day is None:
#         day = [1, 10, 20]
#     if month is None:
#         month = [8]
#
#     # In case of several models, take the last
#     md_path = glob(f"{models_fold}/*{model_base}*{model_str}*")[-1]
#     if data_fold is None:
#         data_fold = json.load(open(md_path + "/config.json", "r"))["global_setting"]["dataPath"]
#     # get model info for data usage
#     data_use = json.load(open(rf"{md_path}/data_use_info.json", "r"))
#     target = data_use["target"]
#     hp = data_use["hp"]
#     if hp <= 0:
#         print("No forecasting set-up since hp <= 0")
#         sys.exit(1)
#     max_wind = max(list(data_use["raw_feat_and_wind_used"].values()))
#     fut_col = data_use["futured_var"]
#     hist_data_path = glob(rf"{data_fold}/*{data_use['data_name']}*.*")[0]
#
#     # get raw data and prepare sub for climatology
#     data = import_eval_data(hist_data_path, data_use)
#     clim_box = ClimaticScenarios(data=data, target=target,
#                                  ref_date=ref_date, month_=month, day_=day, period_=hp,
#                                  key_=key_, look_back=max(max_wind, 120))
#
#     data_obs = clim_box.get_climatic_hist()
#     ref_test, ref_sub_test = clim_box.get_base_ref()
#
#     rfd_ = ref_date
#     m_r, d_r = f"{rfd_[1]}".zfill(2), f"{rfd_[2]}".zfill(2)
#     data_obs.update({f"year_ref{rfd_[0]}_{m_r}{d_r}": ref_test})
#     year_key = list(data_obs.keys())
#     year_nm = [f"yr{str(year_key.index(yr) + 1).zfill(2)}_" for yr in year_key if "ref" not in yr]
#     year_nm += [f"yref_" for yr in year_key if "ref" in yr]
#
#     # run for all member
#     year_key.sort()
#     year_nm.sort()
#     dict_clim = {}
#     hist_data = ref_sub_test[:-hp]
#     for id_n, member_ in zip(year_nm, year_key):
#         prev_sc = data_obs[member_]  # get the forecasting data and assign them to the referenced/now dataset
#         prev_sc.index = ref_test.index
#         c_fut = [c for c in prev_sc.columns if c.startswith(tuple(fut_col))]  # identify the forecasted features
#         climate_data = ref_test.copy()
#         climate_data[c_fut] = prev_sc[c_fut]
#         new_set = pd.concat([hist_data, climate_data], axis=0)
#         new_set.index.name = "Date"
#         dict_clim[id_n] = new_set
#     try:
#         out = SklearnModels(model_name=model_base, mode_run="climatology", discr_file=model_str,
#                             op_md_path=md_path, dict_clim=dict_clim).run()
#         return out, md_path
#
#     except (RuntimeError, TypeError, ValueError, KeyError, IndexError) as e:
#         exc_type, exc_obj, exc_tb = sys.exc_info()
#         file_name_ = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#         print(f"There were some problems in the script. \n"
#               f"Error info --> Type: {exc_type.__name__}, File: {file_name_}, Line: {exc_tb.tb_lineno}")
#         logger.warning(f"Exception {e} on {file_name_}")
#         pass
#

def run_clim_sub2(md_path: str, period: tuple[str, str] = None, data_fold: str = None,
                  model_base: str = "mlp_skl", model_str: str = None):
    """
    Run climatology evaluation for a trained model

    :param
        - models_fold:
    """
    dict_clim = get_full_clim(period=period, md_path=md_path, data_fold=data_fold)
    try:
        out = SklearnModels(model_name=model_base, mode_run="climatology", discr_file=model_str,
                            op_md_path=md_path, dict_clim=dict_clim).run()
        return out, md_path

    except (RuntimeError, TypeError, ValueError, KeyError, IndexError) as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name_ = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"There were some problems in the script. \n"
              f"Error info --> Type: {exc_type.__name__}, File: {file_name_}, Line: {exc_tb.tb_lineno}")
        logger.warning(f"Exception {e} on {file_name_}")
        pass


def get_full_clim(period: tuple[str, str], md_path: str, data_fold: str = None, list_spec_date: list[str] = None):
    """
    Get full climatology dict for a model

    :param
        - period: tuple(start, end) for evaluation, format "yyyymmdd"
        - md_path: str of model path
        - data_fold: path of the data directory
        - list_spec_date: pass a list of dates to filter period on
    """
    data_use = json.load(open(rf"{md_path}/data_use_info.json", "r"))
    full_period = pd.date_range(start=str(period[0]), end=str(period[1]), freq="D")
    if list_spec_date is not None:
        list_spec_date = pd.to_datetime(list_spec_date)
        full_period = full_period.intersection(list_spec_date)
    if data_fold is None:
        data_fold = json.load(open(md_path + "/config.json", "r"))["global_setting"]["dataPath"]
    hist_data_path = glob(rf"{data_fold}/*{data_use['data_name']}*.*")[0]
    data = import_eval_data(hist_data_path, data_use).set_index("Date")
    # get model info for data usage
    target = data_use["target"]
    hp = data_use["hp"]
    if hp <= 0:
        print("No forecasting set-up since hp <= 0")
        sys.exit(1)
    max_wind = max(list(data_use["raw_feat_and_wind_used"].values()))
    fut_col = data_use["futured_var"]
    # sub_test = make_date_range(start_=start_date, end_=end_date)
    sub_test = full_period
    zip_dte = [(a.year, a.month, a.day) for a in sub_test]
    pbar = tqdm(zip_dte, desc=f"Clim-Data-Eval-Period: ", position=0, leave=False, file=sys.stdout, total=len(sub_test))
    full_clim = {}
    try:
        for y, m, d in pbar:
            if f"{m}-{d}" == "2-29":  # skip Feb29th
                continue
            # run the climatology function
            dte_ = f"{y}-" + f"{m}".zfill(2) + "-" + f"{d}".zfill(2)
            ref_date = (y, m, d)
            clim_box = ClimaticScenarios(data=data, target=target,
                                         ref_date=ref_date, month_=[m], day_=[d], period_=hp,
                                         key_=target, look_back=max(max_wind, 120))

            data_obs = clim_box.get_climatic_hist()
            ref_test, ref_sub_test = clim_box.get_base_ref()

            # rfd_ = ref_date
            m_r, d_r = f"{m}".zfill(2), f"{d}".zfill(2)
            data_obs.update({f"year_ref{y}_{m_r}{d_r}": ref_test})
            year_key = list(data_obs.keys())
            year_nm = [f"yr{str(year_key.index(yr) + 1).zfill(2)}_" for yr in year_key if "ref" not in yr]
            year_nm += [f"yref_" for yr in year_key if "ref" in yr]

            # run for all member
            year_key.sort()
            year_nm.sort()
            dict_clim = {}
            hist_data = ref_sub_test[:-hp]
            for id_n, member_ in zip(year_nm, year_key):
                prev_sc = data_obs[member_]  # get the forecasting data and assign them to the referenced/now dataset
                prev_sc.index = ref_test.index
                c_fut = [c for c in prev_sc.columns if c.startswith(tuple(fut_col))]  # identify the forecasted features
                climate_data = ref_test.copy()
                climate_data[c_fut] = prev_sc[c_fut]
                new_set = pd.concat([hist_data, climate_data], axis=0)
                new_set.index.name = "Date"
                dict_clim[id_n] = new_set
            full_clim[dte_] = dict_clim
        return full_clim
    except Exception as e:
        print(f"Clim dict Error: {e}")


def run_clim(arg_user: dict):
    """Run climatology mode using user parsed arguments """
    arg_user = arg_user
    model_base = arg_user["model_base"]
    model_discr = arg_user["discr_model"]
    models_fold = arg_user["model_dir"]
    data_fold = arg_user["data_fold"]
    md_path = glob(f"{models_fold}/*{model_base}*{model_discr}*")[-1]
    out = run_clim_sub2(md_path=md_path,
                        period=arg_user["period"],
                        data_fold=data_fold,
                        model_base=model_base,
                        model_str=model_discr)
    return out


def deploy_hindcast(model_str: str,
                    models_fold: str ,
                    hist_fold: str = None,
                    hind_fold: str = None,
                    eval_period: tuple[str, str] = None,
                    model_base="mlp_skl",
                    run_mode="hindcast",
                    max_ensemble: int = None):
    """
    Perform hindcast for a pretrained model, which will be identified using a model_str a box of models (models_box).

    :param model_str: a string identifier for filtering on models_box
    :param models_fold: the parent folder that holds the models
    :param hist_fold: the parent path that holds the observed data
    :param hind_fold: the parent path that holds the hindcast data
    :param eval_period: the evaluation period to consider
    :param model_base: the model string that follows the "run_ " in the model folder
    :param run_mode: one of "hindcast" or "realtime" run mode
    :param max_ensemble: limit the size of the ensemble members
    :return: (processed_results, processed_seeds, full_result, full_seed)

    """
    eval_period = (str(eval_period[0]), str(eval_period[1]))
    md_path = glob(f"{models_fold}/*{model_base}*{model_str}*")[-1]
    if hist_fold is None:
        hist_fold = Path(json.load(open(md_path + "/config.json", "r"))["global_setting"]["dataPath"]).parent
    if hind_fold is None:
        hist_name = Path(hist_fold).stem
        hind_fold = hist_fold.replace(hist_name, run_mode)

    # get model info for data usage
    data_use = json.load(open(rf"{md_path}/data_use_info.json", "r"))
    z_cfg = json.load(open(rf"{md_path}/config.json", "r"))
    hp = data_use["hp"]
    if hp <= 0:
        print("No forecasting set-up since hp <= 0")
        sys.exit(1)

    hind_dte_mbr_full_dico = get_hindcast_dict(d_use=data_use,
                                               period=eval_period,
                                               hist_dir=hist_fold,
                                               hind_dir=hind_fold,
                                               max_ensemble=max_ensemble)
    try:

        out = SklearnModels(model_name=model_base, mode_run=run_mode, discr_file=model_str, z_cfg=z_cfg,
                            op_md_path=md_path, dict_clim=hind_dte_mbr_full_dico).run()
        return out, md_path
    except (RuntimeError, TypeError, ValueError, KeyError, IndexError) as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name_ = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"There were some problems in the script. \n"
              f"Error info --> Type: {exc_type.__name__}, File: {file_name_}, Line: {exc_tb.tb_lineno}")
        logger.warning(f"Exception {e} on {file_name_}")
        pass


def run_hindcast(arg_user: dict):
    """
    Run climatology mode using user parsed arguments
    """
    arg_user = arg_user
    mode_run = arg_user["run_mode"]
    data_fold = arg_user["data_fold"]
    hind_fold = arg_user["hind_fold"]
    if hind_fold is None:
        hind_fold = str(Path(data_fold).parent) + f"/{mode_run}"
    out, md_path = deploy_hindcast(model_str=arg_user["discr_model"],
                                   hist_fold=data_fold,
                                   hind_fold=hind_fold,
                                   models_fold=arg_user["model_dir"],
                                   model_base=arg_user["model_base"],
                                   eval_period=arg_user["period"],
                                   run_mode=mode_run,
                                   max_ensemble=arg_user.get("max_ensemble", None))
    return out, md_path
