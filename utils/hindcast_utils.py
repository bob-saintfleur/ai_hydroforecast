#!/bin/env python3
# Copyright 2025 Gustave Eiffel University
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for full license information.
import os
import pandas as pd
from glob import glob
from datetime import timedelta
from tqdm import tqdm


def read_obs_data_csv(f, index_col=None):
    """
    Read csv file with while specifying index
    """
    if index_col is None:
        index_col = ["Date"]
    df = pd.read_csv(f, sep=";", parse_dates=["Date"], index_col=index_col)
    return df


def read_forc_data_pq(f):
    """Read parquet file"""
    df = pd.read_parquet(f, engine="pyarrow")
    return df


def get_avail_dte_forc(df):
    """Get available date from hindcast dataframe """
    dt_0 = pd.to_datetime(df.index.get_level_values("time").unique()).strftime("%Y-%m-%d")
    return dt_0


def prepare_period_sequence(period: tuple[str, str] = ("19891001", "19910930"),
                            seq_size: int = 30,
                            max_lead: int = 7):
    """Prepare full period for obs data extended by sequence (backward) and lead time (upward) """
    start_df = pd.to_datetime(period[0]) - timedelta(days=seq_size)
    end_df = pd.to_datetime(period[1]) + timedelta(days=max_lead)
    full_period = sorted(pd.date_range(start=start_df, end=end_df, freq="D").strftime("%Y-%m-%d"))
    return full_period


def make_past_and_fut_from_t0(t0: str, back_seq: int = 30, fut_len: int = 1):
    """
    Prepare past index and future index given a timstamp t0, back to 'back_seq' and projected to 'fut_len'
    """
    past_ = pd.date_range(start=t0, periods=back_seq, freq="-1D").sort_values()
    fut_ = pd.date_range(start=t0, periods=fut_len + 1, freq="1D").sort_values()[1:]
    return past_, fut_


def load_hist_fut(basin: str, obs_dir: str, fut_dir: str, index_forc: list = None):
    """
    Load historical observations and hindcasting data

    :param:
        - basin: the basin code
        - obs_dir: the data observation directory
        - fut_dir: the forecast data directory
        - index_forc: multi-index names of the forecast data
    """
    if index_forc is None:
        index_forc = ["time", "step", "member", "Date"]
    df_ob = read_obs_data_csv(glob(obs_dir + f"/{basin}.*")[0])
    df_ft = read_obs_data_csv(glob(fut_dir + f"/{basin}.*")[0], index_col=index_forc)
    return df_ob, df_ft


def make_hindcast_basin_from_time_batch_forecast(time_batch_fold: str,
                                                 dir_to: str = None,
                                                 src_name: str = "ecmwf",
                                                 rename_col: dict = None):
    """
    Group time-batched hindcast into basin-batched hindcast data. It assumes that "basin" is part of the index keys

    :param:
        - time_batch_fold: time batched folder
        - dir_to: directory to save new formatted files
        - src_name: specify a source string as hindcast may come from seral sources
        - rename_col: if needed, past a dict to renaming columns
    """
    if dir_to is None:
        dir_to = "../data/hindcast"
    dir_to = f"{dir_to}/{src_name}"
    os.makedirs(dir_to, exist_ok=True)
    all_tb = glob(time_batch_fold + "/*parquet.gzip")
    all_fc = []
    TB_BAR = tqdm(all_tb, desc="By-TimeBatchFile")
    for f in TB_BAR:
        df_fc = read_forc_data_pq(f)
        all_fc.append(df_fc)
    all_fc = pd.concat(all_fc, axis=0).sort_index()
    if rename_col is not None:
        all_fc = all_fc.rename(columns=rename_col)
    FC_BAR = tqdm(all_fc.groupby("basin"), desc="By-BasinFullBatch")
    for b, d_bv in FC_BAR:
        bv_ = f"{b}".zfill(8)
        FC_BAR.set_postfix_str(f"On basin {bv_}")
        d_bv.droplevel("basin").to_csv(f"{dir_to}/{bv_}.csv", sep=";")


class LoadHindCastOperationalData:
    """Load data for hindcasting """

    def __init__(self,
                 hist_dir: str,  # dir of observed data
                 hindcast_dir: str,
                 eval_period: tuple[str, str],
                 data_usage:dict,
                 max_ensemble: int = None,
                 ):
        """
        Prepare hindCasting data to evaluate model.

        :param:
            - hist_dir: directory of past observation data, like ./hist_dir/bbbbbb.txt
            - hindcast_dir: directory of hincast data, like ./hindcast_dir/bbbbbb.txt
            - eval_period: evaluation period, must be available in historical data
            - data_usage: dict with specific data usage configuration from pre-trained model
            - max_ensemble: litmit the ensemble size
        """
        self.hist_dir = hist_dir
        self.hindcast_dir = hindcast_dir
        self.data_usage=data_usage
        self.period = eval_period
        basin = data_usage["data_name"]
        seq_max = max(data_usage["raw_feat_and_wind_used"].values())
        lead_max = data_usage["hp"]
        replace_P_Q = ()
        pq_x = [c for c in data_usage["futured_var"] if c.lower().startswith("p_q")]
        if len(pq_x)!=0:
            repl_ = [(c, c.replace("P_Q", "Q_")) for c in pq_x]
            replace_P_Q = tuple(repl_)
        self.full_period = prepare_period_sequence(period=eval_period,
                                                   seq_size=seq_max,
                                                   max_lead=lead_max)
        self.basin = basin
        self.seq_max = seq_max
        self.lead_max = lead_max
        df_obs, df_fut = load_hist_fut(basin, self.hist_dir, self.hindcast_dir) #FIXME: Load P_Qltm here conditionned by other_model_use
        if len(replace_P_Q)!=0:
            df_obs["Q_Targ"] = df_obs["Q_Obs"]
            for a,b in replace_P_Q:
                if b not in df_obs.columns:
                    # FIXME: TOOOOO specific  with "_" or not .e.g (P_QSACSMA, to Q_SAC_SMA instead of Q_SACSMA)
                    b = [c for c in df_obs.columns if c.lower().replace("_", "")==b.lower().replace("_", "")][0]
                df_obs[a]=df_obs[b]
        if max_ensemble is not None and max_ensemble > 0:
            df_fut = df_fut.loc[df_fut.index.get_level_values("member") <= max_ensemble]
        self.df_fut = df_fut
        self.df_obs = df_obs.loc[self.full_period]
        forc_date_t0 = get_avail_dte_forc(self.df_fut)
        lim_forc = pd.date_range(*eval_period, freq="D")
        self.avail_date = [a for a in forc_date_t0 if a in self.full_period and a in lim_forc]

    def _get_obs_fut(self):
        """ Get hist and hindcast dataframe """
        return self.df_obs, self.df_fut

    def _get_forcast_dict(self, date_0):
        """ Make pairs of hist-obs and forecast for basin from date_0 """
        obs, fut = self.df_obs, self.df_fut
        past_, future_ = make_past_and_fut_from_t0(date_0, self.seq_max, self.lead_max)
        past_ = past_.intersection(obs.index)
        if len(past_) < self.seq_max:
            return
        hist_df = obs.loc[past_]
        fut_obs = obs.loc[obs.index.intersection(future_)]
        hist_df.index.name = "Date"
        fut_obs.index.name = "Date"
        forc_dict = {"sequence": hist_df, "fut_obs": fut_obs}

        # Correct
        fut_df = fut.loc[fut.index.get_level_values("time") == date_0]
        members = {}
        for m, df_m in fut_df.groupby("member", observed=True):
            d_fc = df_m.droplevel("member")
            d_fc.index.name = "Date"
            members[f"m{m}"] = d_fc
        forc_dict["members"] = members
        return forc_dict

    def get_forcast_period_bv(self):
        """ Get the forecast period data for a given basin """
        BAR_DATE = tqdm(self.avail_date, desc=f"Data-Integrity-ck -b:{self.basin} -T:{self.period}", leave=False)
        FULL_T_FORC = ()
        for date_0 in BAR_DATE:
            BAR_DATE.set_postfix_str(f"Date : {date_0}")
            d_i = self._get_forcast_dict(date_0=date_0)
            if d_i is None:
                continue
            FULL_T_FORC += ((date_0, d_i),)
        return FULL_T_FORC


def get_hindcast_dict(d_use,
                      period: tuple[str, str],
                      hist_dir,
                      hind_dir,
                      max_ensemble: int = None):
    """
    Get hindcasting data based on preset args.

    :param:
        - d_use: data usage dict from trained model
        - period: period to be considered for evaluation
        - hist_dir: directory of observed data, expected like hist_dir/basin.txt
        - hind_dir: directory of forecast, expected hind_dir/basinXX.csv
        - max_ensemble: limit the ensemble size, in case of large number
    """
    hindcast_loader = LoadHindCastOperationalData(hist_dir=hist_dir,
                                                  hindcast_dir=hind_dir,
                                                  eval_period=period,
                                                  data_usage=d_use,
                                                  max_ensemble=max_ensemble,
                                                  )
    bv_for_dict = hindcast_loader.get_forcast_period_bv()
    hindcast_bar = tqdm(bv_for_dict, leave=False, desc="Data-Avail-Process")
    all_df = {}
    for dte_, dict_FC in hindcast_bar:
        seq_df, fut_obs, dict_mbr = dict_FC["sequence"], dict_FC["fut_obs"], dict_FC["members"]
        all_mbr = ()
        for m, mbr_i in dict_mbr.items():
            act_fut = fut_obs.copy()
            act_fut[mbr_i.columns] = mbr_i.droplevel(["time", "step"])
            full_df = pd.concat([seq_df, act_fut], axis=0).sort_index()
            full_df = full_df.loc[~full_df.index.duplicated(keep="last")].fillna(0, axis=0)
            all_mbr += (full_df,)
        all_df[dte_] = all_mbr
    return all_df

