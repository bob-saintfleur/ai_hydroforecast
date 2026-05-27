import json
import os, sys
import pickle
from pathlib import Path
from glob import glob
from typing import Any
import yaml
import numpy as np
import pandas as pd
import psutil
from scipy.ndimage import shift
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from datetime import datetime, timedelta
from utils.logger import logger

num_cpus = psutil.cpu_count(logical=False)

list_model = ["rfr_skl", "mlp_skl", "svr_skl"]
color = ["k", "darkorange", "c"]


def get_time_run(precision: str = "s") -> str:
    """Return timestamp string with specified precision."""
    now = datetime.now()
    formats = {
        "tf": now.strftime("%Y%m%d%H"),
        "d": now.strftime("%Y%m%d"),
        "h": now.strftime("%Y%m%d_%H"),
        "m": now.strftime("%Y%m%d_%H%M"),
        "s": now.strftime("%Y%m%d_%H%M%S"),
        "ms": now.strftime("%Y%m%d_%H%M%S") + f"_ms{now.microsecond:06d}"
    }
    return formats.get(precision, formats["s"])


def get_default_params_from_module(module, exclude="params"):
    """Get the default parameters of a module """
    import inspect
    parx = inspect.signature(module)
    defaults = {}
    for name, value in parx.parameters.items():
        if not name.startswith(exclude):
            defaults[name] = value

    clean = {
        name: (param.default if param.default is not inspect._empty else None)
        for name, param in defaults.items()
    }
    return clean

def main_cfg():
    """Get the master config"""
    return json.load(open("utils/main_config.json", "r+", encoding="utf-8"))


def get_user_options(option_file: str = "_user_option.yml"):
    """Get the user options from the yaml file """
    return yaml.load(open(option_file, "r"), Loader=yaml.SafeLoader)


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


def read_basin_list_from_file(basins_file: str):
    """Read a list of basins from a file """
    return [a.split()[0] for a in open(basins_file, "r").readlines()]


def read_list_of_basins(path_or_list: str or list = None):
    """Read the list of the available basins"""
    # if path_or_list is None:
    #     path_or_list = "data/basin_list"
    if isinstance(path_or_list, str):
        with open(path_or_list, "r") as list_bv:
            bs_l = list_bv.readlines()
        basins = [c.split()[0] for c in bs_l]
    elif isinstance(path_or_list, list):
        basins = path_or_list
    else:
        raise AttributeError
    return basins


def get_m_list(parser_cfg):
    """ Get the list of models from user arguments"""
    model_dir = parser_cfg.run_dir
    model_base = parser_cfg.model_base
    discr_file = parser_cfg.discr_model if parser_cfg.discr_model else "*"

    models = glob(f"{model_dir}/run_{model_base}*{discr_file}*")
    if len(models) == 0:
        models = glob(f"{model_dir}/**/run_{model_base}*{discr_file}*", recursive=True)
    if len(models) == 0:
        logger.warn(f"No MODEL PATH found like: {model_dir}/run*{model_base or ''}*{discr_file or ''}")
        return
    return models


def get_data_path_for_basin_id(basin_id, basin_data_folder: str):
    """Find the data path based on basin id"""
    # if basin_data_folder is None:
    #     basin_data_folder = "data/all_sim_obs_lstm"
    try:
        data_bv_list = glob(rf"{basin_data_folder}/{basin_id}.*")
        if len(data_bv_list) == 0:
            data_bv_list = glob(rf"{Path(basin_data_folder).parent}/{basin_id}.*")
        data_path = data_bv_list[0]
        if data_path is not None:
            return data_path
        else:
            logger.error(f"No path found for {basin_id}")
            sys.exit()
    except Exception as e:
        logger.error(f"Could not find basin {basin_id} in {basin_data_folder}")
        sys.exit('Absence of basin')


def subdiv_date(period, n_sub=1):
    """Divide a period into n sub-periods"""
    dd_ = pd.date_range(start=pd.to_datetime(period[0], format="%Y%m%d"),
                        end=pd.to_datetime(period[1], format="%Y%m%d"),
                        freq="D")
    n_sub = min(n_sub, dd_.shape[0] // 2)
    by_sub = dd_[::len(dd_) // n_sub]
    dd_f = [(by_sub[i - 1], by_sub[i] + timedelta(days=-1)) for i in range(1, len(by_sub))]
    if by_sub[-1] != dd_f[-1]:
        dd_f = dd_f + [(by_sub[-1], dd_[-1])]
    dd_f = [(c[0].strftime("%Y%m%d"), c[1].strftime("%Y%m%d")) for c in dd_f]
    return dd_f


def map_ymd(date_):
    """
    Get YYYY, MM, DD from a condensed date

    :param date_: str or int in the form of yyyymmdd
    :return: yyyy, mm, dd
    """
    y4_, m2_, d2_ = str(date_)[:4], str(date_)[4:6], str(date_)[6:8]
    return tuple([int(x) for x in [y4_, m2_, d2_]])


def make_date_range(start_: tuple, end_: tuple = None):
    """
    Convert start and end_ into a date-range.

    :param
        - start: tuple(yyyy, mm, dd), the starting date for the evaluation
        - end_: tuple(yyyy, mm, dd), the ending date [optional]. if none, the end will be start+365
    :return
        - a daterange object
    """
    ref_date = datetime(start_[0], start_[1], start_[2], 0, 0)
    if end_ is None:
        per_ = pd.date_range(start=ref_date, periods=365)
    else:
        end_ = datetime(end_[0], end_[1], end_[2], 0, 0)
        per_ = pd.date_range(start=ref_date, end=end_)
    return per_


def get_str_boolean(str_true_or_false: str):
    """
    Get the boolean equivalent from str
    """
    possible_ = {"false": False, "true": True, "none": None, "null": None}
    str_lower = str_true_or_false.lower()
    if str_lower in possible_.keys():
        return possible_[str_lower]
    else:
        return str_true_or_false


def ensure_date_col(data: pd.DataFrame):
    """Make sure date is present in dataframe"""
    c_date = [c for c in data.columns if c.lower().startswith("date")]
    if len(c_date) == 0:
        data["Date"] = pd.to_datetime(data.index)
    else:
        data.rename(columns={c_date[0]: "Date"}, inplace=True)
    return data


def get_meta_data(path_) -> dict:
    """
    Retrieve the metadata from a model folder.

    :param path_: the path of the concerned model
    :return: a dictionary of the metadata found
    """
    md_ = rf"{path_}/data_use_info.json"
    with open(md_, "r+") as fp_:
        meta_info = json.load(fp=fp_)
    return meta_info


def change_etiage_over_global(config_, new_value: bool = False):
    """
    Specific boolean function to reset the focus on the dataset (Drought or global or any pre-set rate flow).

    :param config_: the config to be changed
    :param new_value: the new value, False for Global, True for Drought
    """
    status_ = 0
    with open(rf"{config_}", "r+") as fp:
        base_c = json.load(fp)
        if base_c["global_setting"]["seuil_target"]["use"]:
            status_ = 1
        base_c["global_setting"]["seuil_target"]["use"] = new_value
    return base_c, status_


def mse(y_true, y_pred):
    num_ = np.mean(np.square(y_true - y_pred))
    deno_ = np.var(y_true)
    nse = (1 - num_ / deno_) if deno_ != 0 else np.inf
    return round(nse, 4)

def nse(y_true, y_pred):
    num_ = np.mean(np.square(y_true - y_pred))
    deno_ = np.var(y_true)
    nse = (1 - num_ / deno_) if deno_ != 0 else np.inf
    return round(nse, 4)


def persistence(y_true, y_pred, hp):
    """Compute persistence using hp"""
    if hp == 0:
        pers_ = np.nan
    else:
        y_naiv = shift(y_true, hp, mode="nearest")
        num_ = np.mean(np.square(y_true - y_pred))
        deno_ = np.mean(np.square(y_true - y_naiv))
        pers_ = round(1 - num_ / deno_, 4)
    return pers_


def div_list(list_, n):
    """ Divide a list into n sublist"""
    k, m = divmod(len(list_), n)
    return [list_[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def get_path(opti_path: str, model_name: str, discriminator: str = None):
    """
    get the corresponding path of the passed arguments
    :return: the first element found
    """
    if discriminator is not None:
        all_found = glob(rf"{opti_path}/run_*{model_name}*{discriminator}*")
    else:
        all_found = glob(rf"{opti_path}/run_*{model_name}*")
    str_found = all_found[0] if len(all_found) != 0 else model_name

    if os.path.isdir(Path(str_found)):
        path_found = str_found
    elif os.path.isfile(Path(str_found)):
        path_found = Path(f'{str_found}').parent
    else:
        raise FileNotFoundError(f"No folder found with {opti_path} + {model_name} + {str_found}")
    return path_found


# Hyperparameter loading
def load_param_space(model_name: str):
    """Load the parameter to use according to model_name"""
    with open("utils/param_space.json", 'r') as fps:
        my_params = json.load(fps)

    if model_name not in list_model:
        print(f"model name ['{model_name}'] is not in {list_model}")
        raise TypeError
    else:
        modx_par = my_params[f"{model_name}_param_"]

    # Do not use any parameter with null value
    par_to_keep = {}
    for par, v in modx_par.items():
        if v is not None:
            par_to_keep[par] = v
    param_to_tune = par_to_keep
    return param_to_tune


def load_model_instance(model):
    """Load the model instance"""
    switcher = {
        "mlp_skl": MLPRegressor,
        "rfr_skl": RandomForestRegressor,
        "svr_skl": SVR,
    }
    return switcher.get(model, "Invalid module")
