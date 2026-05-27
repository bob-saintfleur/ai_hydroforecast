#!/bin/lib python
import json
import os
import sys
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import shift
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datatools import manage_input_data as data_engine
from datatools.features_setting import GetFromCrossCorrelation, add_cummul_on_instant_indicator
from datatools.features_setting import get_sub_index, get_feature_attributes, reduce_features, make_features_combo
from datatools.read_data import import_data, import_eval_data
from utils.utils import get_data_path_for_basin_id, ensure_date_col
from utils.logger import logger
warnings.filterwarnings("ignore")


class BaseData:
    def __init__(self, modx_name, run_mode, cfg_, pre_md_path, op_df: pd.DataFrame = None, data_use=None):
        self.modx_name = modx_name
        self._run_mode = run_mode
        self.cfg = cfg_
        self.op_df = op_df
        g_cfg = self.cfg['global_setting']
        self.cv_ = g_cfg["cv"]
        self.n_jobs = g_cfg["n_jobs"]
        fcg = self.cfg
        self.sub_bv_info = fcg["batch_variables"]["meta_data"]["Subbasin"]
        self.drop_inter_hp = g_cfg["drop_inter_hp"]

        xCor_cfg = fcg["correlation_process"]
        feat_tool = get_feature_attributes(self.cfg)
        hp = self.hp = feat_tool["hp"]
        hp_max = self.hp_max = feat_tool["hp_max"]
        hp_step = self.hp_step = feat_tool["hp_step"]
        basin = self.basin = g_cfg["basin"]
        target_as_input = self.target_as_input = feat_tool["target_as_input"]
        self.data_path = g_cfg["dataPath"]
        self.use_dta = feat_tool["use_dta"]
        self.recursive_ok = g_cfg["recursive_mode"]
        self.back_4_no_future = g_cfg["back_4_no_future"]
        self.future_back_on = feat_tool['future_back_on']
        add_up_hp = self.add_up_hp = feat_tool["add_up_hp"]
        use_future = self.use_future = g_cfg["future"]
        self.forecast_avail = g_cfg["prediction_recursive"]["forecasting_available"]
        self.normalize_tg = g_cfg["normalize_target"]
        self.out_label = feat_tool["label"]
        self.out_unit = feat_tool["unit"]
        self.label_n_unit = feat_tool["out_label"]

        future_except_case = fcg["batch_variables"]["meta_data"]["future_exception"]

        value_in_future = feat_tool["future_v"]
        variables_window = feat_tool["window_v"]
        variables_unit = feat_tool["unit_v"]
        target = self.target = target_id = feat_tool["target_v"]
        variables_used = feat_tool["used_v"]
        to_reduce = feat_tool["to_reduce"]
        to_cumulate = feat_tool["to_cumulate"]
        all_reduced = feat_tool["all_reduced"]
        fut_reduced = feat_tool["fut_reduced"]
        reduction_mode = feat_tool["reduction_mode"]
        win_reduced = feat_tool["win_reduced"]
        self.feat_tool = feat_tool
        self.test_year = g_cfg["testEvent"]
        self.data_basin_id_path = get_data_path_for_basin_id(basin_id=basin, basin_data_folder=g_cfg["dataPath"])

        # Load data
        other_model_use = g_cfg["other_model_use"]
        if run_mode not in ["climatology", "apply", "evaluate", "hindcast", "realtime"]:
            logger.info("Data processing STARTED")
        data_raw = None
        if self._run_mode in ["search", "train"]:
            data_raw, _ = import_data(f"{self.data_basin_id_path}", other_model_use)

        if self._run_mode in ["apply", "evaluate"]:
            data_path = f"{g_cfg['dataPath']}/{basin}.txt"
            data_raw = import_eval_data(path_=data_path, data_use=data_use)

        if self._run_mode in ["operational", "climatology", "hindcast", "realtime"]:
            if self.op_df is not None:  # added on may6th 2024
                data_raw = ensure_date_col(self.op_df)
            else:
                data_path = f"{pre_md_path}/_temp_data.csv"  # for parallelization
                if Path(data_path).is_file():
                    data_raw = import_eval_data(path_=data_path, data_use=data_use)
                    self.data_op_name = Path(data_path).stem
                else:
                    data_path = f"{g_cfg['dataPath']}/{data_use['data_name']}.txt"
                    if Path(data_path).is_file():
                        data_raw = import_eval_data(path_=data_path, data_use=data_use)
                        self.data_op_name = Path(data_path).stem
                    else:
                        sys.exit()
        data = data_raw
        self.use_rate_tg = g_cfg["seuil_target"]["use"]
        self.rate_tg = g_cfg["seuil_target"]["max"] if self.use_rate_tg else int(data[target].max() * 2)

        # selection concerned variables
        new_col = ["Event", "Date"] + variables_used
        data = data[[col for col in new_col if col in data.columns]]
        data = data.loc[:, ~data.columns.duplicated()].copy()

        if use_future:
            approach_status = "Perfect"
            if future_except_case["use"]:
                approach_status = "Operational"
        else:
            approach_status = "Limited"

        data_mode = "Global" if not self.use_rate_tg else "Etiage"
        data_use_info = {"target": target, "target_as_input": target_as_input,
                         "hp_step": hp_step, "recursive_mode": self.recursive_ok, "hp": hp, "hp_max": hp_max,
                         "back_4_future": g_cfg["back_4_no_future"], "window_upgraded_hp": add_up_hp,
                         "target_Normalization": self.normalize_tg, "Approach": approach_status,
                         "data_focus": data_mode, "futured_var": [k for k, v in value_in_future.items() if v > 0]}

        # case of assimilation of other models
        if other_model_use is not None:
            self.use_dta = False if "_error" in other_model_use else self.use_dta
            data_use_info.update({"other_model_use": other_model_use})
        data_use_info["dta_use"] = self.use_dta

        if g_cfg["use_variable_reduction"]:
            logger.info("Reduction mode: activated")
            data_reduced = reduce_features(data=data,
                                           to_reduce=to_reduce,
                                           reduction_mode=reduction_mode)
            logger.info(f"Reduction mode args : To-reduce = {to_reduce}; Mode: {reduction_mode}")
            # remove the features used to make the reduction in order to avoid double usage
            if to_reduce:
                for var_k in to_reduce.keys():
                    data_reduced.drop(to_reduce[var_k], axis=1, inplace=True)
                reduced_added = [el for el in data_reduced.columns[1:] if el not in variables_used]
                variables_reduced = [el for el in variables_used if el not in all_reduced] + reduced_added

                # Update of features sequencing sizes
                for v_x in all_reduced:
                    value_in_future.pop(v_x)
                    variables_window.pop(v_x)

                for var_k, red_var_k in zip(to_reduce.keys(), reduced_added):
                    value_in_future.update({f"{red_var_k}": fut_reduced[var_k]})
                    variables_window.update({f"{red_var_k}": win_reduced[var_k]})
                    for el in to_cumulate:
                        if el == var_k:
                            to_cumulate.remove(var_k)
                            to_cumulate.append(red_var_k)
            else:
                logger.error(msg="Reduction mode improperly activated")
                sys.exit()

            data = data_reduced
            variables_used = variables_reduced
            to_cumulate = [el for el in to_cumulate if el in variables_used]
            to_cumulate = to_cumulate

        data_use_info.update({"features_reduction": g_cfg["use_variable_reduction"], "data_name": g_cfg["data_name"]})

        if use_future:
            no_fut_cols = []
            # non-predictable features are not projected into the future
            for col, vfut in value_in_future.items():
                if vfut <= 0:
                    data[col] = shift(data[col], -vfut, cval=0)
                    no_fut_cols.append(col)
                    # set to zero to avoid indexation mistakes in the windowing processing
                    value_in_future[col] = 0

        combine_case = fcg["batch_variables"]["meta_data"]["To_combine"]
        data_use_info.update({"use_future": self.use_future,
                              "moving_average": g_cfg["add_moving_average"]["use"],
                              "features_combination": combine_case["use"]})

        # if combination of two variables, the new created feature will inherit that with the sequence
        if combine_case["use"]:
            hist_c = combine_case["historical"]
            list_to_combine = [c for c in combine_case.keys() if c.startswith("Combo_")]
            list_combo, list_operation = [], []
            for comb_ in list_to_combine:
                comb_x = combine_case[comb_]
                combo_x = make_features_combo(data, comb_x)
                ft_1, ft_2 = comb_x["list_features"][0], comb_x["list_features"][1]

                # identify the critical feature from which the new Combo_x will inherit values
                combx_key_var = [ft_1, ft_2][np.array([value_in_future[ft_1], value_in_future[ft_2]]).argmin()]
                variables_window.update({combo_x: variables_window[combx_key_var] if hist_c else 1})
                if use_future:
                    value_in_future.update({combo_x: value_in_future[combx_key_var]})
                    if combx_key_var in no_fut_cols:
                        no_fut_cols.append(combo_x)
                list_combo.append((combo_x, comb_x["operation"]))
            data_use_info.update({"Combo_list": list_combo})
            logger.info("Combine mode: checked")

        self.value_in_future = value_in_future
        self.variables_unit = variables_unit
        self.variables_used = variables_used
        self.to_reduce = to_reduce
        self.reduction_mode = reduction_mode
        self.all_reduced = all_reduced
        self.win_reduced = win_reduced
        self.fut_reduced = fut_reduced
        self.to_cumulate = to_cumulate
        self.data = data.copy()

        fut_list = [f for f, v in value_in_future.items() if v > 0]

        # Formatting data
        reformat = data_engine.DataReFormat(data, basin=basin, target=target, use_target_as_input=target_as_input,
                                            hp=hp, future=use_future, use_dta=self.use_dta, fut_var_list=fut_list)

        self.dta_info = reformat.make_dta_wrt_hp()
        self.dt_classic = reformat.data_classic()

        xCor_data = reformat.data_for_xcorrelation()
        if xCor_cfg["auto_windowing"]:
            inertia_, low_lim, up_lim = xCor_cfg["inertia"], xCor_cfg["min_depth"], xCor_cfg["max_depth"]
            X_cor_box = GetFromCrossCorrelation(data=xCor_data, target_ft="QObs", low_lim=low_lim, up_lim=up_lim)
            cor_wind = X_cor_box.get_specific_wind(inertia=inertia_)
            if self.use_future:
                if future_except_case["use"]:
                    for c, v in value_in_future.items():
                        if c in cor_wind.keys():
                            if v > 0:
                                cor_wind.update({c: 0})
            variables_window = cor_wind

        # Ready for windowing and applying lag
        self.variables_window = variables_window
        self.var_lag = variables_window

        windowing = data_engine.InputWindowing(xCor_data, opt_lag=self.var_lag)
        data_lagged = windowing.apply_lag(hp_target=hp, future=use_future, dict_future=value_in_future)
        inter_hp_col = [c for c in data_lagged.columns[data_lagged.columns.str.startswith("P_Q")] if "-" not in c if
                        not c.endswith(("_0j", f"_{self.hp}j"))]

        if (self.drop_inter_hp is True) and (len(inter_hp_col) != 0):
            data_lagged.drop(columns=inter_hp_col, inplace=True)

        # Renaming features in case of back future shift
        tail_cutter = None  # No meteo at t0
        if use_future:
            tail_cutter = -self.hp if self.hp > 0 else None
            for nf_c in no_fut_cols:
                col_n_fut = [c for c in data_lagged.columns if c.startswith(nf_c)]
                for cln in col_n_fut:
                    indj = int(cln.split("j")[0].split("_")[-1])
                    data_lagged.rename(columns={f"{cln}": f"{nf_c}_{indj + (-hp_max + 1) * self.future_back_on}j"},
                                       inplace=True)
            self.no_fut_cols = no_fut_cols

        # We compute rolling mean values for all the features in , except for the target if not used
        if not target_as_input and target_id:
            variables_used = [el for el in variables_used if el != target_id]

        def add_moving_mean_to_classic_data(size: int = 10):
            """Add moving average"""
            for var_w in variables_used:
                to_mean = [c for c in data_lagged.columns if c.startswith(var_w)]
                if len(to_mean) >= size:
                    v_to_mean = to_mean[-size:]
                else:
                    v_to_mean = to_mean
                m_av_ = data_lagged[v_to_mean].mean(axis=1)
                data_lagged[f'{var_w}_MvA{len(v_to_mean)}'] = round(m_av_, 2)

        if g_cfg["use_cumulate"]:
            cum_par = g_cfg["cumulator_params"]
            event_gap = cum_par["between_event_gap"]
            forg_fact = cum_par["forget_factor"]
            event_gap = min(data_lagged.shape[0], event_gap)
            data_lagged = add_cummul_on_instant_indicator(data_lagged,
                                                          self.to_cumulate,
                                                          event_gap, "last",
                                                          forg_fact, False)

        # consider as well the moving average on the historic features
        if g_cfg["add_moving_average"]["use"]:
            add_moving_mean_to_classic_data(size=g_cfg["add_moving_average"]["size"])

        data_lagged = data_lagged[[c for c in data_lagged.columns if not c.endswith(("_MvA1", "_MvA2"))]]

        # relative shifting respect to variables historical relative gap
        warm_up = max(self.var_lag.values()) - 1

        # Use dt if needed
        if reformat.use_dta and hp > 0:
            d_obs = self.dta_info[f"dta{hp}_QObs"][warm_up:tail_cutter].values
            data_lagged["QObs"] = d_obs

        # prepare base results
        data_lagged["date"] = reformat.date_c[warm_up:tail_cutter]  # handle end of lagged

        mask_Test, mask_Stop = get_sub_index(reformat_data=data_lagged, g_cfg=g_cfg, run_mode=self._run_mode)
        data_lagged = data_lagged.drop('date', axis=1)
        # data_lagged.to_csv(HOME_PATH + "data/my_data_windowing_n.csv", sep=";", index_label="Date_now")

        data_use_info.update({"hydrologic_year": g_cfg["hydrologicalYear"],
                              "long_date_used": g_cfg["use_following_dates"]})

        # Preparing a basis for the results with FOUR main columns
        self.y_naif = np.array(reformat.y_naive[warm_up:tail_cutter], "float32").ravel()
        self.y_obs_t0 = np.array(reformat.y_t0[warm_up:tail_cutter], "float32").ravel()
        self.y_obs_thp = np.array(reformat.y_obs[warm_up:tail_cutter], "float32").ravel()
        self.data_lagged = data_lagged

        df_results = pd.DataFrame()
        df_results["date"] = data_lagged.index
        df_results["y_t0"] = self.y_obs_t0
        df_results["y_obs"] = self.y_obs_thp
        df_results["y_naive"] = self.y_obs_t0  # y_naif
        self.warm_up = warm_up

        # df_results.index = df_results["date"]
        df_results.set_index("date", inplace=True, drop=True)
        df_results.index.name = "date"

        # Prepare base results for train and test
        base_results = df_results[mask_Test]
        data_lagged_c = data_lagged.copy()
        data_test = data_lagged_c[mask_Test]
        data_stop = data_lagged_c[mask_Stop]
        data_train = data_lagged_c[~mask_Test]
        base_results_train = df_results[~mask_Test]

        self.data_train, self.data_test, self.data_stop = data_train, data_test, data_stop
        self.base_results, self.base_results_train = base_results, base_results_train

        # get the threshold mask in order to set focus on solely a part of the data
        self.mask_train = self.base_results_train["y_obs"] <= self.rate_tg
        self.mask_test = self.base_results["y_obs"] <= self.rate_tg
        # self.base_results.to_csv(HOME_PATH + "data/base_results.csv", sep=";")

        # defining a scaler for train data
        self.scaler_X = MinMaxScaler(feature_range=(0, g_cfg["scaleFactor"])) if g_cfg["normMethod_minimax"] is True \
            else StandardScaler()
        self.scaler_Y = MinMaxScaler(feature_range=(0, g_cfg["scaleFactor"])) if g_cfg["normMethod_minimax"] is True \
            else StandardScaler()

        # defining a scaler for test data
        self.scaler_xT = MinMaxScaler(feature_range=(0, g_cfg["scaleFactor"])) if g_cfg["normMethod_minimax"] is True \
            else StandardScaler()
        self.scaler_yT = MinMaxScaler(feature_range=(0, g_cfg["scaleFactor"])) if g_cfg["normMethod_minimax"] is True \
            else StandardScaler()

        Train_x, _, Train_y, _ = self.get_xy_train_test()

        # Data are scaled at twice of the max
        if self._run_mode in ["train", "search"]:
            self.scaler_X.fit(Train_x * 2)
            self.scaler_Y.fit(np.array(Train_y * 2).reshape(-1, 1))

        # Preparation of data for a keras models
        dTrain, dTest = self.get_train_test()
        all_col = dTrain.columns[1:]
        y_col = dTrain.columns[0]

        x_col = all_col  # All input features
        X_tr = dTrain[x_col]
        y_tr = np.asarray(dTrain[y_col]).reshape(-1, 1)
        X_ts = dTest[x_col]
        y_ts = np.asarray(dTest[y_col]).reshape(-1, 1)
        self.X_ts_op = X_ts
        ft_names = x_col
        self.ft_names = ft_names
        self.x_col = x_col
        if self._run_mode in ["train", "search"]:
            # Scaling the data, leave as it is
            self.scaler_X.fit_transform(X_tr)
            self.scaler_X.fit_transform(X_ts)
            self.scaler_yT.fit_transform(y_ts)
            self.scaler_Y.fit_transform(y_tr)

        self.base_shape = X_tr.shape[1]
        data_use_info.update({"vector_train_nb": len(self.ft_names),
                              "raw_feat_and_wind_used": self.variables_window,
                              "use_rate_in_Loss": self.use_rate_tg,
                              "rate_target": self.rate_tg})
        self.data_use_info = data_use_info

        if run_mode in ["train", "search"]:
            logger.info("Data processing COMPLETE")

    @property
    def run_mode(self):
        return self._run_mode

    @run_mode.setter
    def run_mode(self, new_value):
        self._run_mode = new_value

    def save_data_info(self, path_: str = None):
        if path_ is None:
            path_ = "."
        with open(os.path.join(path_, "data_use_info.json"), "w+") as mdf:
            json.dump(self.data_use_info, mdf, indent=4, ensure_ascii=True)
        return self.data_use_info

    def base_result(self):
        """Get base for ground data analysis"""
        return self.base_results

    def get_train_test(self):
        return self.data_train, self.data_test

    def get_unscaled_train_test(self):
        return self.get_xy_train_test()

    def save_scaler_param(self, path_cfg):
        """Save scaler parameters"""
        if self.run_mode in ["search", "train"]:
            scaler_params = {"X_scaler": self.scaler_X, "y_scaler": self.scaler_Y}
            joblib.dump(scaler_params, rf"{path_cfg}/scaler_param.joblib")

    def get_xy_train_test(self):
        """get train test data"""
        colx = self.data_train.columns[1:]
        coly = self.data_train.columns[0]
        xtr = self.data_train[colx]
        ytr = self.data_train[coly]
        xts = self.data_test[colx]
        yts = self.data_test[coly]
        return round(xtr, 4), round(xts, 4), round(ytr, 4), round(yts, 4)

    def get_data_op(self):
        """Get operational data"""
        if self._run_mode in ["operational", "climatology", "hindcast", "realtime"]:
            data_op_ = self.X_ts_op[-self.hp - 1:]
        else:
            data_op_ = None
        return data_op_, self.base_results[-self.hp - 1:]


def check_data(run_mode="search"):
    """Check the formatted data, for quick debugging mode"""
    with open("../utils/main_config.json", "r+") as fp:
        cfg = json.load(fp)
    model_name = None
    run_mode = run_mode
    return BaseData(modx_name=model_name, run_mode=run_mode, cfg_=cfg)
