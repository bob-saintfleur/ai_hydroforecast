#!/bin/env python3

# Copyright 2025 Gustave Eiffel University
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for full license information.

import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from tqdm import tqdm
# from permetrics.regression import RegressionMetric as REG
import HydroErr as he


import warnings
warnings.filterwarnings("ignore")


debug_size = 2

def clean_xr(x: xr.Dataset) -> xr.Dataset:
    """Clean xarray file """
    # stack year, seed as member
    x_ = x.stack(member=["year", "seed"])
    # clean nan values in obs
    x_ = x_.dropna(dim='Date', how="all", subset=["obs"])
    # clean nan values in prediction
    x_ = x_.dropna(dim='Date', how="all", subset=["prediction"])
    # clean nan values in member
    x_ = x_.dropna(dim='member', how="any", subset=["prediction"])

    if len(x_.chunks) > 0:
        x_ = x_.chunk({"member": -1})
    return x_


def rank_basin_xr_ds(x: xr.Dataset, bins: int = 10) -> xr.DataArray:
    # prepare final vector of bins
    final_bins = range(1, bins + 1)
    # Dataset is empty
    if x.sizes["Date"] == 0:
        coords = x.coords.copy()
        for key in ["Date", "year", "seed"]:
            del coords[key]
        coords = coords.assign(rank_bins=final_bins)
        dims = tuple(coords.sizes.values())
        ranks = np.empty(dims)
        ranks[:] = np.nan
        ranks = xr.DataArray(data=ranks, coords=coords, name="histogram_rank")
        return ranks

    # here x_ should be full without any np.nan value in it, thus we can compute ranks
    ranks = xs.rank_histogram(observations=x["obs"][..., 0],
                              forecasts=x["prediction"],
                              dim="Date",
                              member_dim="member")
    ranks = ranks.groupby_bins("rank", bins=bins).sum()
    ranks = ranks.assign_coords(rank_bins=final_bins)
    ranks = ranks / ranks.sum()
    return ranks


def persistence(y_pred, y_true, y_naive):
    """Compute persistence """
    num_ = np.mean(np.square(y_true - y_pred))
    den_ = np.mean(np.square(y_true - y_naive))
    return round(1 - (num_ / den_), 4)


def run_pp(path_to: str = None, debug: bool = None, pred_all: pd.DataFrame = None):
    """
    Compute metrics on a list of prediction stored in a folder

    :param pred_all: a ready to use dataframe, if None, gather files from run_dir and prepare them for processing
    :param path_to: a path to drop result
    :param debug: set on to run debug
    """
    if debug is None:
        debug = False
    t0 = time.time()
    if path_to is None:
        path_to = "./post_processed"
    if debug is True and "debug" not in path_to.lower():
        path_to = f"{path_to}_DEBUG"

    os.makedirs(path_to, exist_ok=True)
    post_process_out = path_to

    list_ctx = pred_all.index.get_level_values("context").unique()

    list_hp = pred_all.index.get_level_values("hp").unique()
    lst_yr = pred_all.index.get_level_values("year").unique()

    pred_all = pred_all.sort_index()
    all_basins = pred_all.index.get_level_values("basin").unique()

    n_ranks_intervals = 10
    qt_list = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]
    rocs, rocs_i, briers, briers_i, ranks, spread, pred_obs_res, crps = [], [], [], [], [], [], [], []

    if debug is True:
        list_ctx = list_ctx[:2]
        list_hp = list_hp[:2]
        n_ranks_intervals = 2

    grp_dims = ["context", "basin", "hp"]

    groups = pred_all.groupby(grp_dims, observed=True)
    if debug is True:
        groups = list(groups)[:2]

    with tqdm(total=len(list(groups)), desc="By group: ") as pbar:
        for (keys, group) in groups:
            try:
                if (len(lst_yr) > 1) & (-1 in lst_yr):
                    group = group.loc[group.index.get_level_values("year") != -1]
                if "seed" in group.columns:
                    group.set_index("seed", append=True, inplace=True)

                # ENSEMBLE METRICS  #
                # warnings.simplefilter("ignore", UserWarning)

                ## convert to xarray
                pred_obs = group.to_xarray()
                pred_obs = clean_xr(pred_obs)

                # ROC/AUC

                # for debug purpose:
                if debug is True:
                    pred_obs = pred_obs.isel(member=slice(0, debug_size), Date=slice(0, 50))
                    qt_list = [0.1, 0.5]

                obs = pred_obs["obs"][..., :1]  # As obs has been replicated on seed and year, keep the first
                pred = pred_obs["prediction"]
                obs_quantiles = obs.quantile(qt_list)

                # Process ">" condition
                roc = xs.roc(observations=(obs > obs_quantiles).mean("member").astype(np.int8),
                             forecasts=(pred > obs_quantiles).mean("member"),
                             bin_edges="continuous",
                             dim="Date",
                             return_results="all_as_metric_dim").to_dataframe(name="roc")

                roc = roc.filter(like="roc").droplevel([c for c in roc.index.names if c in ["year", "seed"]])
                roc = roc[~roc.index.duplicated(keep="last")]
                roc = roc.reset_index().pivot(columns="metric",
                                              index=grp_dims + ["quantile", "probability_bin"],
                                              values="roc")

                roc = roc.rename({"area under curve": "AUC",
                                  "false positive rate": "FP",
                                  "true positive rate": "TP", },
                                 axis=1)
                rocs.append(roc.copy())
                del roc
                # Process "<=" condition
                roc = xs.roc(observations=(obs <= obs_quantiles).mean("member").astype(np.int8),
                             forecasts=(pred <= obs_quantiles).mean("member"),
                             bin_edges="continuous",
                             dim="Date",
                             return_results="all_as_metric_dim").to_dataframe(name="roc")

                roc = roc.filter(like="roc").droplevel([c for c in roc.index.names if c in ["year", "seed"]])
                roc = roc[~roc.index.duplicated(keep="last")]
                roc = roc.reset_index().pivot(columns="metric",
                                              index=grp_dims + ["quantile", "probability_bin"],
                                              values="roc")
                roc = roc.rename({"area under curve": "AUC",
                                  "false positive rate": "FP",
                                  "true positive rate": "TP", },
                                 axis=1)

                rocs_i.append(roc.copy())
                del roc

                # compute RANK_HISTOGRAM
                ranks_ = rank_basin_xr_ds(pred_obs, n_ranks_intervals)

                # convert back to pandas
                ranks_ = ranks_.to_dataframe()["histogram_rank"]
                ranks_ = ranks_.reorder_levels(grp_dims + ["rank_bins"])
                ranks.append(ranks_.copy())
                del ranks_

                # compute CRPS
                crps_ = xs.crps_ensemble(observations=obs[..., 0],
                                         forecasts=pred,
                                         dim="Date",
                                         member_dim="member").to_dataframe(name="crps")
                crps.append(crps_.copy())
                del crps_

                # compute BRIER
                obs_thresholds = obs.quantile(qt_list)

                # Compute BRIER with ">" conditions
                brier = xs.brier_score(observations=(obs > obs_thresholds).mean("member"),
                                       forecasts=(pred > obs_thresholds).mean("member"),
                                       dim="Date").to_dataframe(name="brier")

                brier = brier.filter(like="bri").droplevel([c for c in brier.index.names if c in ["year", "seed"]])
                brier = brier[~brier.index.duplicated(keep="last")]
                brier = brier.reorder_levels(grp_dims + ["quantile"])

                briers.append(brier.copy())
                del brier

                # Compute BRIER with "<=" conditions
                brier = xs.brier_score(observations=(obs <= obs_thresholds).mean("member"),
                                       forecasts=(pred <= obs_thresholds).mean("member"),
                                       dim="Date").to_dataframe(name="brier")
                brier = brier.filter(like="brie").droplevel([c for c in brier.index.names if c in ["year", "seed"]])
                brier = brier[~brier.index.duplicated(keep="last")]
                brier = brier.reorder_levels(grp_dims + ["quantile"])

                briers_i.append(brier.copy())
                del brier
                del pred_obs

                # DETERMINISTIC METRICS
                index_ = pd.MultiIndex.from_product([[key] for key in keys], names=grp_dims)

                # deterministic with all ensemble values as individual members
                obs_i, pred_i, naive_i = group['obs'], group["prediction"], group["naive"]

                # deterministic with median taken on year/seed values
                median_pred = group.groupby(["Date"]).agg("median")
                obs_m, pred_m, naive_m = median_pred['obs'], median_pred["prediction"], median_pred["naive"]

                pred_obs_res_ = pd.DataFrame({
                    # "nse": he.nse(pred_i, obs_i),
                    # "kge": he.kge_2012(pred_i, obs_i),
                    # "rmse": he.rmse(pred_i, obs_i),
                    # "pers": persistence(pred_i, obs_i, naive_i),
                    "nse_on_median": he.nse(pred_m, obs_m),
                    "kge_on_median": he.kge_2012(pred_m, obs_m),
                    "rmse_on_median": he.rmse(pred_m, obs_m),
                    "pers_on_median": persistence(pred_m, obs_m, naive_m)


                },
                    index=index_)

                pred_obs_res.append(pred_obs_res_.copy())
                del obs_i, pred_i, naive_i, pred_obs_res_, obs_m, pred_m, naive_m
                del pred
                # del group
            except Exception as e:
                print(f"Exception found {e}")
                continue
            pbar.update(1)
        del groups
        t1 = time.time()
        print(f"Pandas computation = {t1 - t0}")

    for out, name in [(briers, "briers"), (rocs, "rocs"), (briers_i, "briers_ie"), (rocs_i, "roc_ie"),
                      (pred_obs_res, "metrics"), (crps, "crps")]:
        if len(out) != 0:
            out_ = pd.concat(out)
            out_ = out_[~out_.index.duplicated(keep="last")]
            out_.to_parquet(os.path.join(post_process_out, f"pp_{name}_pandas.parquet.gzip"),
                            compression="GZIP",
                            engine="pyarrow")

    ranks = pd.concat(ranks).to_frame()
    ranks = ranks[~ranks.index.duplicated(keep="last")]
    ranks.to_parquet(os.path.join(post_process_out, "pp_ranks_pandas.parquet.gzip"),
                     compression="GZIP",
                     engine="pyarrow")

    print(f"Expected size of ranks = {len(list_ctx) * len(list_hp) * len(all_basins) * n_ranks_intervals}")
    print(f"Actual size of ranks = {len(ranks)}")
    print(f"Details: list_ctx [{len(list_ctx)}] * nhp [{len(list_hp)}] * nbv[{len(all_basins)}] * "
          f"n_rank_interval[{n_ranks_intervals}]")
    print(f"Results saved in : {post_process_out}")
    return


if __name__ == '__main__':
    run_pp()
    print("Done !")
