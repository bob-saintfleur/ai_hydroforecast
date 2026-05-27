#!/bin/env python3
import argparse
import copy
import sys
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from utils import job_multi
from utils.utils import get_time_run, div_list, num_cpus, main_cfg, get_m_list
from utils.prepare_config import prepare_config, update_main_cfg
from utils.base_model import SklearnModels
from utils.logger import logger

import warnings

warnings.filterwarnings("ignore")


def train_model(model_base, z_cfg, mode_run, nb_seeds, disc_f=None, dict_x=None, time_run=None):
    """Train a model using SklearnModels wrapper."""
    SklearnModels(model_name=model_base,
                  z_cfg=z_cfg,
                  mode_run=mode_run,
                  n_seed=nb_seeds if "svr" not in str(model_base) else 1,
                  discr_file=disc_f,
                  show_plot=False,
                  check_on_global=False,
                  x_run_dict=dict_x,
                  time_file=time_run
                  ).run()


def eval_model(model_base="mlp_skl", disc_f=None, time_run=None, opti_path=None):
    """Evaluate a model using SklearnModels wrapper."""
    SklearnModels(model_name=model_base,
                  mode_run="apply",
                  opti_path=opti_path,
                  discr_file=disc_f,
                  show_plot=False,
                  check_on_global=False,
                  time_file=time_run
                  ).run()


def train_on_list(model_base, z_cfg, mode_run, list_bv, param_list, nb_seeds):
    """Run training across combinations of basins and parameters."""
    logger.info(f"Running concerns {len(list_bv)} BASINS and {len(param_list)} USER-OPTIONS SETS.")
    for i, param in enumerate(param_list):
        logger.info(f"~~~~~~~~~~~ OPTION : {i+1} ~~~~~~~~~~~~ ")
        for basin in list_bv:
            logger.info('-'*35)
            logger.info(f"Basin : {basin}")
            time_step = get_time_run("d")
            run_id = f"bv{basin}_hp{param['hp']}_{time_step}"
            cfg1 = prepare_config(basin_id=basin, config0=z_cfg)
            train_model(model_base, cfg1, mode_run, nb_seeds, dict_x=param, time_run=run_id)


def eval_on_list(list_models: list):
    """Evaluate a list of models, using their path string."""
    if not list_models:
        logger.error("EVAL mode requires pre-trained models")
        sys.exit()
    run_ids = [Path(m).name.split(f"mlp_skl_")[-1] for m in list_models]
    opti_path = Path(list_models[0]).parent
    for i, run_id in enumerate(run_ids):
        try:
            eval_model(disc_f=run_id, opti_path=opti_path)
        except Exception as e:
            logger.warning(f"Exception {e}")
            continue


def eval_from_args(parser_cfg):
    """Evaluate all models following user parsed args """
    models = get_m_list(parser_cfg)
    if models:
        logger.info(f"EVAL mode for {len(models)} PRE-TRAINED models. ***")
        eval_on_list(models)
    else:
        print("Eval list was empty")


def launch_console(parser_cfg):
    """Dispatch function based on the selected run mode."""
    mode_run = parser_cfg.run_mode
    model_base = parser_cfg.model_base

    if mode_run in {"apply", "evaluate"}:
        eval_from_args(parser_cfg)
        return
    if mode_run =="tune":
        pass

    if mode_run =="search":
        z_cfg = update_main_cfg(main_cfg(), parser_cfg)
        param_set = parser_cfg.param_set or parser_cfg.param_grid
        param_list = list(ParameterGrid([param_set]))
        basin_list = parser_cfg.basin_list
        nb_seeds = [parser_cfg.seed] if parser_cfg.seed is not None else parser_cfg.nb_seeds

        g_cfg = z_cfg["global_setting"]
        logger.info(f"Run dir: {g_cfg['run_dir']}")
        logger.info(f"Base MODULE: {model_base}")
        logger.info(f"Assimilation mode : {'MLP Simple' if g_cfg['other_model_use'].startswith('no_in_') else g_cfg['other_model_use']}")
        logger.info(f"Data Path : {g_cfg['dataPath']}")
        logger.info(f"List of basins (N = {len(basin_list)}): {basin_list[:3]} ..")
        logger.info(f"USER-OPTIONS: {param_set}")
        logger.info(f"SEEDS Number: {nb_seeds}")
        logger.info(f"Cross-validation size: {g_cfg['cv']}")
        logger.info(f"Cross-validation parallel jobs: {g_cfg['n_jobs']}")
        train_on_list(model_base, z_cfg, mode_run, basin_list, param_list, nb_seeds)


def launch_parallel_bv_train(u_parser):
    """ Dispatch runs on multiple cpus based un number of basins """
    func = launch_console
    basin_list = u_parser.basin_list
    av_cpu = num_cpus // 5  # to handle cv in gridsearch, if n_jobs < 5, you can lower this
    needed_cpu = 1
    arg_t = ()
    logger.info(f"Available CPU: {av_cpu}")
    if len(basin_list) > 1:
        needed_cpu = min(len(basin_list), av_cpu)
        n_sub = len(basin_list) // av_cpu if needed_cpu < len(basin_list) else needed_cpu
        logger(f"Concurrent Tasks: {n_sub}")
        for bv_l in div_list(basin_list, n_sub):
            tmp_args = argparse.Namespace(**copy.deepcopy(vars(u_parser)))
            tmp_args.basin_list = bv_l
            arg_t += (tmp_args,)
    else:
        arg_t = (u_parser,)
    list_task = [{"func": func, "tasks": [dict(parser_cfg=cfgx) for cfgx in arg_t]}]
    job_multi.par_proc(list_task, num_cpus=needed_cpu)


def eval_parallely(u_parser):
    """ Dispatch evaluation on multiple cpus based un number of model found """
    func = eval_on_list
    list_md = get_m_list(parser_cfg=u_parser)
    av_cpu = num_cpus - 2  # to handle cv in gridsearch, if n_jobs < 5, you can lower this
    needed_cpu = 1
    if len(list_md) > 1:
        needed_cpu = min(len(list_md), av_cpu)
        n_sub = len(list_md) // av_cpu if needed_cpu < len(list_md) else needed_cpu
    else:
        n_sub = 1
    logger.info(f"Available CPU: {av_cpu}")
    logger.info(f"Needed CPU: {needed_cpu} for {n_sub} Concurrent TASKS")
    list_task = [{"func": func, "tasks": [{"list_models": a} for a in div_list(list_md, n_sub)]}]
    job_multi.par_proc(list_task, num_cpus=needed_cpu)


if __name__ == "__main__":
    print(f"It is : {get_time_run('m')}")
