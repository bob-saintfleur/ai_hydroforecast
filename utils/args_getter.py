#!/bin/env python3
import argparse
import os
from pathlib import Path
from glob import glob

import pandas as pd
import yaml
from utils.utils import subdiv_date
from utils.utils import select_bv_by_class, read_list_of_basins
from utils.logger import logger

GLOBAL_SETTING = {"train_start": ("19911001", "20090901"),
                  "train_end": ("19990930", "20180831"),
                  "test_start": ("19891001", "20180901"),
                  "test_end": ("19910930", "20210831"),
                  "ens_start": ("19891001", "20180901"),
                  "ens_end": ("19910930", "20210831"),
                  "camels_dir_name": ("/camelsus", "/camelsfr"),
                  }


def tupling_arg(arg):
    """Ensure parsed arguments as tuple"""
    try:
        # Parse the input string as a tuple
        return tuple(map(int, arg.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {arg}")


class StoreDictKeyPair(argparse.Action):
    """ Adapt command inputs to dict[key: value] format"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for kv in values:
            k, v = kv.split("=")
            try:
                getattr(namespace, self.dest)[k] = int(v)
            except (TypeError, ValueError):
                getattr(namespace,
                        self.dest)[k] = False if v.lower() == "false" else (True if v.lower() == "true" else v)


class StoreGridDictKeyPair(argparse.Action):
    """ Adapt command inputs to dict[key: list] format"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for kv in values:
            k, v = kv.split("=")
            try:
                getattr(namespace, self.dest)[k] = [int(i) for i in v.split(",")]
            except (TypeError, ValueError):
                getattr(namespace, self.dest)[k] = \
                    [False if x.lower() == "false" else (True if x.lower() == "true" else x) for x in v.split(",")]


def get_clim_args():
    """
    Parse input arguments

    Returns
    -------
        dict: Dictionary containing the run config.
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_base', default="mlp_skl")
    parser.add_argument('--run_mode', required=True,
                        choices=["climatology", "hindcast", "realtime"], help="Run mode")
    parser.add_argument('--model_dir', type=str, help="Models box", required=True)
    parser.add_argument('--camels_context',
                        type=str, default="test", choices=["fr", "us", "test"], help="Which camels concerned")
    parser.add_argument('--data_paper_path', type=str, help="Path to the main data dir")
    parser.add_argument('--data_root', type=str, help="Root for data")
    parser.add_argument('--data_fold', type=str, help="Folder of data")
    parser.add_argument('--hind_fold', type=str, help="Hindcast folder data")
    parser.add_argument('--path_to', type=str, help="Path to save outputs from model_dir")
    parser.add_argument('--period', nargs="*", help="start stop", metavar="yyyymmdd yyyymmdd [space separator]")
    parser.add_argument('--n_sub', type=int, default=1, help="Split the period into n_sub parts")
    parser.add_argument('--max_ensemble', type=int, help="Limit the ensemble size")
    parser.add_argument('--discr_model', type=str, help="Identify the models to evaluate")
    parser.add_argument('--basins_file', type=str, default="basins_56", help="File of basins")
    parser.add_argument('--basin_list', nargs="*", help="list of basins to run on separate by space")
    parser.add_argument('--nb_basins', type=int, default=1, help="range of basin counter")
    parser.add_argument('--start_basin', type=int, help="Beginning of a slice", required=False)
    args = parser.parse_args()

    idx = {"us": 0, "fr": 1, "test": 0}.get(args.camels_context, "test")  # Tied to the present paper
    camels_dir_name = "" if args.camels_context not in ["us", "fr"] else GLOBAL_SETTING.get("camels_dir_name")[idx]
    if args.period is None:
        args.period = (GLOBAL_SETTING.get("test_start")[idx], GLOBAL_SETTING.get("test_end")[idx])
    sub_dates = [args.period]

    if args.camels_context == "test":
        args.data_paper_path = "."
    else:
        assert args.data_paper_path, \
            "Please, [--data_paper_path PATH] argument is expected along side [--camels_context us OR fr] "
    if args.data_root is None:
        args.data_root = str(args.data_paper_path) + f"/data{camels_dir_name}"
    data_root = str(args.data_root)
    if args.data_fold is None:
        args.data_fold = data_root + "/all_sim_obs_lstm"
        if not Path(args.data_fold).is_dir():
            args.data_fold = data_root + "/training"
    args_ = vars(args)
    if args_["n_sub"] > 1:
        if args_["run_mode"] in {"hindcast", "realtime"}:
            # Avoid taking slice with no valid date, since hindcast and realtime are mainly discontinuous
            min_sub = min(len(pd.date_range(*args_["period"][:2])) // 5, args_["n_sub"])  # FIXME, may fail for feq!=D
            args_.update({"n_sub": max(1, min_sub)})
        sub_dates = subdiv_date(args_["period"], args_["n_sub"])
    args_["sub_dates"] = sub_dates
    return args_


def get_run_args():
    """Parse input arguments
        Returns
        -------
        Container of user config.
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_base', default="mlp_skl", choices=["mlp_skl", "rfr_skl", "svr_skl"])
    parser.add_argument('--run_mode', default="search", choices=["search", "apply"], help="Run mode")
    parser.add_argument('--camels_context',
                        type=str, default="test", choices=["fr", "us", "test"], help="Which camels concerned")
    parser.add_argument('--data_paper_path', type=str, help="Path to the main data dir")
    parser.add_argument('--run_dir', type=str, default="./runs", help="Path to hold the runs")
    parser.add_argument('--data_root', type=str, default="./data", help="Data root directory")
    parser.add_argument('--data_path', type=str, help="Data path to basins")
    parser.add_argument('--discr_model', type=str, help="discriminate model to use")

    parser.add_argument('--param_grid', nargs="*", action=StoreGridDictKeyPair,
                        metavar=" k1='a,b,c' kn='v,k,..,z' ",
                        help="Param grid for the grid_search mode")
    parser.add_argument('--param_set', nargs="*", action=StoreDictKeyPair, metavar='k1=v1 k2=v2 kn=vn',
                        help="Parameter set for one search mode. <int>: hp, all_window. <float>: seuil_target. "
                             " <bool> : target_as_input, future, future_exception, use_cumulate, "
                             "moving_average, inertia, auto_windowing")

    parser.add_argument('--basins_file', type=str, default="basins_list", help="File of basins")
    parser.add_argument('--basin_id', type=str, help="basin id or name")
    parser.add_argument('--basin_list', nargs="*", help="Space separated list of basins to use")
    parser.add_argument('--start_basin', type=int, help="Beginning of a slice")
    parser.add_argument('--nb_basins', type=int, default=531, help="Number of basins")
    parser.add_argument('--test_period', nargs="*", help="period for test", metavar="yyyymmdd yyyymmdd")
    parser.add_argument('--train_period', nargs="*", help="period for train", metavar="yyyymmdd yyyymmdd")
    parser.add_argument('--nb_seeds', type=int, default=3, help="number of run, aka n_seeds")
    parser.add_argument('--seed', type=int, help="A specific seed to run on")
    parser.add_argument('--cv_size', type=int, help="Cross-Validation size")
    parser.add_argument('--n_jobs', type=int, help="Number of parallel worker. Be careful with its uses")

    parser.add_argument('--hp', type=int, default=1, help="Forecasting horizon pr lead time")
    parser.add_argument('--drop_inter_hp', default="True", type=str, help="Drop intermediate hp data")
    parser.add_argument('--other_model_use', default="no_in_mlp", type=str, help="other models usage")
    parser.add_argument('--target_as_input', default="True", type=str, help="status of the target variable")
    parser.add_argument('--predictDta', default="True", type=str, help="predict dt")
    parser.add_argument('--use_future', default="True", type=str, help="status of predictable features")
    parser.add_argument('--focus_drought', default="False", type=str, help="flow to set focus on, global or drought")
    parser.add_argument('--flow_threshold', type=float, help="specify the flow threshold")
    args = parser.parse_args()

    idx = {"us": 0, "fr": 1, "test": 0}.get(args.camels_context, "test")  # Tied to the present paper
    camels_dir_name = "" if args.camels_context not in ["us", "fr"] else GLOBAL_SETTING.get("camels_dir_name")[idx]
    if args.test_period is None:
        args.test_period = (GLOBAL_SETTING.get("test_start")[idx], GLOBAL_SETTING.get("test_end")[idx])
        args.train_period = (GLOBAL_SETTING.get("train_start")[idx], GLOBAL_SETTING.get("train_end")[idx])

    if args.camels_context != "test":
        assert args.data_paper_path, \
            "Please, [--data_paper_path PATH] argument is expected along side [--camels_context us OR fr] "
    else:
        args.data_paper_path = "."
    data_root = str(args.data_paper_path) + f"/data{camels_dir_name}"

    if "err" in args.other_model_use:
        args.predictDta = "False"
    if args.data_path is None:
        args.data_path = data_root + "/all_sim_obs_lstm"
    if args.basins_file is not None:
        args.basins_file = data_root + "/" + args.basins_file
        list_basins = read_list_of_basins(args.basins_file)
    else:
        list_basins = read_list_of_basins(data_root + "/basins_list")

    av_basins = [b for b in list_basins for f in glob(args.data_path + "/*.txt") if b in f]
    if len(list_basins) > len(av_basins):
        logger.warn(f"{len(list_basins) - len(av_basins)} basins are missed in {args.data_path}")
    list_basins = av_basins
    list_basins.sort()

    if args.basin_id is not None:
        args.basin_list = [args.basin_id]
        args.nb_basins = None

    if args.basin_list is not None and args.nb_basins is not None:
        args.nb_basins = None

    if args.nb_basins:
        args.basin_list = list_basins[:args.nb_basins]

    if args.start_basin and args.nb_basins:
        args.basin_list = list_basins[args.start_basin:args.start_basin + args.nb_basins]

    if args.basin_list is None:
        args.basin_list = list_basins

    if args.param_set is not None:
        args.param_grid = None

    return args


def get_args_from_yaml_file(yaml_file: str = None):
    """Load user options from a yaml file"""
    if yaml_file is None:
        yaml_file = "../_user_option.yml"
    u_dict = yaml.load(open(yaml_file, "r"), Loader=yaml.SafeLoader)
    return argparse.Namespace(**u_dict)
