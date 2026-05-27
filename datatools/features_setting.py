import json
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa import stattools


line_style = ['--', '-.', '-', ':', 'solid', 'dashed', 'dashdot', 'dotted', (0, (1, 10)), (0, (1, 1)), (0, (1, 1)),
              (5, (10, 3)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]


def check_str(elt_, pattern_list):
    """
    Get an element that matches a pattern from a list. e.g, pattern = [*ak, *pok*], if element elt matches *ak or *pok*,
     it should be returned. It works like a filter.
    """
    el_ = [elt_ for b in pattern_list if b.replace("*", "") in elt_]
    elt = el_[0] if len(el_) != 0 else None
    return elt


def get_feature_attributes(cfg: dict = None):
    """ Manage the attributes of the features, like their input sequence and lead times """
    if not cfg:
        cfg = json.load(open("utils/main_config.json", "r+"))

    value_in_future, variables_window, variables_unit, to_reduce, variables_used = {}, {}, {}, {}, []
    reduction_mode, all_reduced, win_reduced, fut_reduced, to_cumulate, to_reduce = {}, [], {}, {}, [], []
    basin_id, target_info = '', ''
    fcg = cfg
    g_cfg = cfg['global_setting']
    target = g_cfg["target"]
    p_rec = "prediction_recursive"
    rc_mode = "recursive_mode"
    hp = g_cfg["hp"] if g_cfg[rc_mode] is False else g_cfg[p_rec]["hp_start"]
    hp_max = g_cfg[("%s" % p_rec)]["hp_end"] if g_cfg[rc_mode] is True else hp
    hp_step = g_cfg[p_rec]["hp_step"]

    target_as_input = g_cfg["targetAsInput"]
    use_future = g_cfg["future"]
    use_dta = g_cfg["predictDta"]
    recursive_ok = g_cfg[rc_mode]
    add_up_hp = (hp_max if g_cfg[rc_mode] else hp) if g_cfg["add_hp_to_window"] else 0
    fut_fact = 1 if use_future else 0
    back_4_no_future = g_cfg["back_4_no_future"]
    future_back_on = 1 if back_4_no_future else 0
    bx_var = "batch_variables"

    future_except_case = fcg[bx_var]["meta_data"]["future_exception"]
    future_exception_use = future_except_case["use"] * use_future
    future_exception_check = future_except_case["feature_list_or_marker"] if future_exception_use else "?????"
    future_except_var, future_only_var = [], []

    # Other needed variables
    out_label = "Discharge" if target.title() == "Debit" else "Water table"
    out_unit = r'$m^{3}s^{-1}$' if str(out_label) in ["Discharge", "Runoff", "Streamflow", "Flow"] else "mm"
    label_unit = f"{out_label} ({out_unit})"

    # use batch _variables
    target_id = fcg[("%s" % bx_var)]["meta_data"]["TARGET_ID"]
    target_id = target_id
    basin_id = fcg[bx_var]["meta_data"]["BASIN_ID"]
    to_cumulate = fcg[bx_var]["meta_data"]["To_cumulate"]

    for var in fcg[bx_var].keys():
        if not var.startswith("meta_data"):
            if fcg[bx_var][var]["use"]:
                # use all batch variables picked as use
                for elt in fcg[bx_var][var]["list_names"]:
                    fut_except_window, future_except_future = 1, 1
                    if elt == check_str(elt, future_exception_check):
                        future_except_var.append(elt)
                        fut_except_window, future_except_future = 1, 0

                    v_fut = fcg[bx_var][var]["future"] * (future_except_future if future_exception_use else 1)
                    win_s = fcg[bx_var][var]["window_size"]

                    if hp > 0:
                        if v_fut > 0:
                            value_in_future[elt] = (hp_max if g_cfg[rc_mode] is True else hp) * fut_fact
                            if future_exception_use:
                                future_only_var.append(elt)
                                win_s = win_s * 0
                        else:
                            value_in_future[elt] = ((- hp_max) * future_back_on if g_cfg[rc_mode] else 0) * fut_fact
                    else:
                        value_in_future[elt] = 0
                    if target_id in value_in_future.keys():
                        value_in_future[target_id] = 0
                    variables_window[elt] = win_s + add_up_hp

                    variables_unit[elt] = fcg[bx_var][var]["unit"]  # unit of the declared variables
                variables_used += fcg[bx_var][var]["list_names"]  # list of the variables to be used

            if fcg[bx_var][var]["reduce"]:
                to_reduce[var] = fcg[bx_var][var]["list_names"]
                reduction_mode[var] = fcg[bx_var][var]["reduction_mode"]
                all_reduced += fcg[bx_var][var]["list_names"]
                win_reduced[var] = fcg[bx_var][var]["window_size"] + add_up_hp
                v_fut_ = fcg[bx_var][var]["future"]
                if v_fut_ > 0:
                    fut_reduced[var] = (hp_max if g_cfg[("%s" % rc_mode)] else hp) * fut_fact
                else:
                    fut_reduced[var] = ((- hp_max) * future_back_on if g_cfg[rc_mode] else - hp) * fut_fact

    if not target_as_input:
        del value_in_future[target_id]
        del variables_window[target_id]
        del variables_unit[target_id]

    # The target variable is an element of a batch, then it only needs to be identified, not removed here
    target = target_id

    out = {
        "used_v": variables_used,
        "target_v": target,
        "basin": basin_id,
        "out_label": label_unit,
        "label": out_label,
        "unit": out_unit,
        "window_v": variables_window, "win_reduced": win_reduced,
        "future_v": value_in_future, "fut_reduced": fut_reduced,
        "unit_v": variables_unit,
        "hp": hp,
        "hp_max": hp_max,
        "hp_step": hp_step,
        "add_up_hp": add_up_hp,
        "future_back_on": future_back_on,
        "use_future": use_future,
        "use_dta": use_dta,
        "target_as_input": target_as_input,
        "target_info": target_info,
        "to_cumulate": to_cumulate,
        "to_reduce": to_reduce, "all_reduced": all_reduced,
        "reduction_mode": recursive_ok
    }
    return out


def get_sub_index(reformat_data, g_cfg=None, run_mode=None):
    """Ensure matching index (dates) and options for transformed data"""
    c_date = [c for c in reformat_data.columns if c.lower().startswith("date")][0]
    reformat_data.index = reformat_data[c_date]
    if not g_cfg:
        g_cfg = json.load(open("utils/main_config.json", "r+"))["global_setting"]

    # Get the year in cfg
    test_year = g_cfg["testEvent"]
    stop_year = g_cfg["stopEvent"]
    hydr_year = g_cfg["hydrologicalYear"]

    # Adapt year in civil or hydrological context  year
    test_year = test_year if not hydr_year else (test_year - 1)
    stop_year = stop_year if not hydr_year else (stop_year - 1)
    start_month = 1 if not hydr_year else 9
    end_month = 12 if not hydr_year else 8

    # case of one year classical test or stop
    start_Td = datetime(test_year, start_month, 1)
    end_Td = datetime(test_year, end_month, 31)
    start_Sd = datetime(stop_year, start_month, 1)
    end_Sd = datetime(stop_year, end_month, 31)

    # cases where users define long dates
    u_start_Td = pd.Timestamp(g_cfg["testStartDate"])
    u_end_Td = pd.Timestamp(g_cfg["testEndDate"])
    u_start_Sd = pd.Timestamp(g_cfg["stopStartDate"])
    u_end_Sd = pd.Timestamp(g_cfg["stopEndDate"])

    # if user prefers using full defined dates. This choice goes beyond hydrological or civilian year
    if g_cfg["use_following_dates"] is True:
        start_Td = u_start_Td
        end_Td = u_end_Td
        start_Sd = u_start_Sd
        end_Sd = u_end_Sd

    mask_Test = (reformat_data[c_date] >= start_Td) & (reformat_data[c_date] <= end_Td)
    mask_Stop = (reformat_data[c_date] >= start_Sd) & (reformat_data[c_date] <= end_Sd)

    if run_mode in ["operational", "climatology", "hindcast", "realtime"]:
        _hp = g_cfg["hp"]
        date1 = list(reformat_data[c_date])[0]
        end_Op = reformat_data[c_date][-1]
        mask_Op = (reformat_data[c_date] >= date1) & (reformat_data[c_date] <= end_Op)
        mask_Test = mask_Op
    return mask_Test, mask_Stop


def make_cross_correlate(reacting_s, impulsor_s, show_plot: bool = False):
    """
    Cross-correlate 2 signals and return the resulting cross-correlogram, response-lag, memory-lag and inertia-lag

    :param reacting_s: the dependant signal
    :param impulsor_s: the source or impulsor signal
    :param show_plot: Only for quick visualization, if True plot the correlogram
    :return: tuple(correlogram, response-lag, memory-lags and inertia-lags)
    """
    corr_ = stattools.ccf(reacting_s, impulsor_s)
    vect_zero = np.argwhere(corr_ <= 0.0)
    first_0 = 0 if vect_zero.shape[0] < 1 else (vect_zero[0][0] if vect_zero[0][0] != 0 else vect_zero[1][0])
    corr_ = corr_[:min(first_0 + 1, 365)]
    score_max = max(corr_)
    lag_rep_ = np.argmax(corr_)
    lag_mem_ = np.argwhere(corr_ >= 0.2)[-1][0] if score_max > 0.2 else lag_rep_
    if show_plot:
        plt.plot(corr_)
    return corr_, lag_rep_ + 1, lag_mem_ + 1, first_0 + 1


class GetFromCrossCorrelation:
    def __init__(self, data: pd.DataFrame(), target_ft: str, impulsor_ft_list: list = None, low_lim=10, up_lim=20):
        """
        Process cross-correlation using stattools.ccf() on a dataframe between all other feature and the target one

        :param data: dataframe holding the features data
        :param target_ft: the feature on which the cross-correlation is aimed
        :param impulsor_ft_list: the features that influence the target feature
        :param low_lim : the lowest allowed limit in for the window
        :param up_lim : the highest allowed limit in for the window
        """
        self.up_lim = up_lim
        self.low_lim = low_lim
        self.data = data
        self.target_ft = target_ft
        if impulsor_ft_list is None:
            impulsor_ft_list = [c for c in data.columns if not c.lower().startswith(("date", "event",
                                                                                     target_ft.lower()))]
        self.impulsor_ft_list = impulsor_ft_list
        self.r_ = np.random.randint(1, 220, len(self.impulsor_ft_list))
        self.g_ = np.random.randint(1, 240, len(self.impulsor_ft_list))
        self.b_ = np.random.randint(1, 255, len(self.impulsor_ft_list))
        self.ls_ = line_style

    def get_windows_from_correlation(self):
        """Get the lag as windows (or sequence) size  for features"""
        all_correl = {}
        out_ = pd.DataFrame(index=["t_max", 't_mem', 't_rep', "score"], columns=self.impulsor_ft_list)
        tgt_ = self.data[self.target_ft].values
        for col in self.impulsor_ft_list:
            imp_ = self.data[col].values
            cg_, tr_, tm_, ti_ = make_cross_correlate(tgt_, imp_, show_plot=False)
            all_correl[col] = cg_
            out_[col] = [ti_, tm_, tr_, cg_.max()]
        return out_, all_correl

    def get_specific_wind(self, inertia: str = "memory") -> dict:
        """
        Compute for each available feature the corresponding windowing to apply according to its cross-correlogram with
        the target feature and bounded by limits specified by the user.

        :param inertia: expected strings are 'response' for using the system's reaction time; "full" for the full length
            of the first part positive correlogram; and 'memory' for the memory time corresponding to a score = 0.2
        :return: {"feature": "window"}
        """
        recap_correl, _ = self.get_windows_from_correlation()
        var_ = list(recap_correl.columns)
        if inertia == "response":
            k_ = "t_rep"
        elif inertia == "full":
            k_ = "t_max"
        else:
            k_ = "t_mem"
        win_var = {}
        for col in var_:
            w_ = int(recap_correl.loc[k_, col])
            if k_ == "t_mem":
                w_ = max(self.low_lim, w_)
                w_ = min(self.up_lim, w_)
            if k_ == "t_rep":
                w_ = max(self.low_lim, w_)
            if k_ == "t_max":
                w_ = max(self.low_lim, w_)
                w_ = min(self.up_lim, w_)
            win_var[col] = w_
        return win_var


def combine_features(data, feat_1: str, feat_2: str, operation_: str = "product"):
    """
    Usefully to create new features from others using an arithmetic operation.

    :param data: dataset which holds the corresponding single features to be combined.
    :param feat_1: feature 1 as the leftmost one.
    :param feat_2: feature 2 as the operand one.
    :param operation_: operation type specified : acceptable ones are 'product', 'sum', 'divide' and 'subtract'.
    :return: the new created combined feature from the parents. The dataframe if modified on place.
    """
    def apply_mode(op_=operation_):
        switch = {
            "sum": data[feat_1] + data[feat_2],
            "product": data[feat_1] * data[feat_2],
            "divide": data[feat_1] / data[feat_2],
            "subtract": data[feat_1] * data[feat_2]
        }
        return switch.get(op_, "invalid_mode")

    comb_var = f"{operation_}_{feat_1}_and_{feat_2}"
    data[comb_var] = apply_mode()
    return comb_var


def make_features_combo(data, combo_dict):
    """
    Usefully to create new features from others using an arithmetic operation.

    :param data: dataset which holds the corresponding single features to be combined.
    :param combo_dict: A dict holding the combination parameters : {"name": str, "list_feat": [], "operation": str}.
        The given name will be set to "Combo_name", the list will must contain only 2 arguments, if more is needed,
        consider the feature reductor option. The expecting operations are "product", "sum", "divide", "subtract" in
        the form of Combo_name = "feat_1" operation "feat_2"
    :return: the new created combined feature from the parents. The dataframe if modified on place.
    """
    feat_list = combo_dict["list_features"][:2]
    operation_ = combo_dict["operation"]
    name_ = combo_dict['name']
    feat_1, feat_2 = feat_list[0], feat_list[1]

    def apply_mode(op_=operation_):
        switch = {
            "sum": data[feat_1] + data[feat_2],
            "product": data[feat_1] * data[feat_2],
            "divide": data[feat_1] / data[feat_2],
            "subtract": data[feat_1] * data[feat_2]
        }
        return switch.get(op_, "invalid_mode or features")

    comb_var = f"Combo_{name_}"
    for ft in [feat_1, feat_2]:
        if ft not in list(data.columns):
            print("\n", "* " * 20, " PROCESS ABORTED ON FEATURE COMBINATIONS STEP ", " *" * 20,
                  f"\n Cause: *[{ft}]* is not present in the specified data features. "
                  f"The availables ones are : {list(data.columns)} \n"
                  f"Hints, set the corresponding batch..->..use-> TRUE in the main_config.json "
                  f"or do not use combination")
            sys.exit()
    data[comb_var] = apply_mode()
    return comb_var


def reduce_features(data, to_reduce: dict, reduction_mode: dict):
    """
    This function is set to performed a reduction operation on a batch of variable of the same type. It will use the
    the name of the batch_box to store the new created value.

    :param data: dataset which holds the corresponding single features to be combined.
    :param to_reduce: dictionary holding the batch variables to reduce in the form of:
        {"RAINFALL": ["RF1", "RFk", "AP1"], "FLOW": ["Q1", "Q2", "Qam3, ..]}.
    :param reduction_mode: dict of the reduction mode specified. {"RAINFALL": "mean", "FLOW": "sum", ...}.
    :return: The data modified.
    """
    data_reduce = data.copy()

    # applying the choice specified by the user
    def apply_reduction_mode(mode_reduction, var_key):
        switch = {
            "sum": np.sum(data_reduce[to_reduce[var_key]], axis=1),
            "mean": np.mean(data_reduce[to_reduce[var_key]], axis=1),
            "std": np.std(data_reduce[to_reduce[var_key]], axis=1),
            "min": np.min(data_reduce[to_reduce[var_key]], axis=1),
            "max": np.max(data_reduce[to_reduce[var_key]], axis=1),
            "median": np.median(data_reduce[to_reduce[var_key]], axis=1)}
        return switch.get(mode_reduction, "Invalid mode")

    for var_k in to_reduce.keys():
        mode = reduction_mode[var_k]
        data_reduce[f"{var_k}_{mode}"] = apply_reduction_mode(mode_reduction=mode, var_key=var_k)
    return data_reduce


def add_cumul_by_name(data, feature_to_cumulate: list, event_gap: int, forget_factor: float = 0.0):
    """
    This function is set to process the cumulate of a feature from a dataframe and add latter to the dataframe with its
    name ended with "_cum".

    :param data: dataframe to add cumulate on.
    :param feature_to_cumulate: list of feature to cumulate. They must be present in the dataframe
    :param event_gap: in integer to stop the cumulator when the corresponding range gives a null increasing
    :param forget_factor: a factor [0, 1) to set to which proportion past values would be remembered . If 0, all the
        past values are remembered in the present value. Should it be set to 1, no cumulate would actually be performed.
    :return: The completed dataframe, but the modification is performed in place
    """
    event_gap = event_gap
    lmbda_fgt = (1 - forget_factor) if 0 <= forget_factor < 1 else 1

    for v_cum in feature_to_cumulate:
        data[f"{v_cum}_cum"] = data[v_cum]
        limit_max = data.shape[0]
        n_i = 1
        for h in range(n_i, event_gap):
            print(data.loc[h - 1, f"{v_cum}_cum"])
            val__ = lmbda_fgt * data.loc[h - 1, f"{v_cum}_cum"] + data.loc[h, v_cum]
            data.loc[h, f"{v_cum}_cum"] = val__

        for i in range(event_gap, limit_max, 1):
            if data.loc[i - event_gap:i, v_cum].sum() <= 0:
                data.loc[i, f"{v_cum}_cum"] = data.loc[i, v_cum]
            else:
                val__ = lmbda_fgt * data.loc[i - 1, f"{v_cum}_cum"] + data.loc[i, v_cum]
                data.loc[i, f"{v_cum}_cum"] = val__
    return data


def add_cummul_on_instant_indicator(data, feature_to_cumulate: list, event_gap: int, instant_="last",
                                    forget_factor: float = 0.0, by_hist: bool = True):
    """
    This function is set to process the cumulate of a feature from a dataframe and add latter to the dataframe with its
    name ended with "_cum". This function should be performed if the concerned dataframe holds its features in a
    windowed form such as [x_-nj, ..., x_-2j, x_-1j, x_0j, x_1j, ..., x_nj]. It will then use the instant passed in the
    in the place of the [-n, ..., n]

    :param data: dataframe to add cumulate on.
    :param feature_to_cumulate: list of feature to cumulate. They must be present in the dataframe
    :param event_gap: in integer to stop the cumulator when the corresponding range gives a null increasing
    :param instant_: this instant is used to specify which instant of the flattened feature to cumulate on. By default,
        it uses "last" but can be any available integer
    :param forget_factor: a factor [0, 1) to set to which proportion past values would be remembered . If 0, all the
        past values are remembered in the present value. Should it be set to 1, no cumulate would actually be performed.
    :param by_hist: Indicate if it os preferred to make cumulative according to historical depth or the old fashion
        consisting in considering the whole event limited or separated by an event gap
    :return: The completed dataframe, but the modification is performed in place
    """
    lmbda_fgt = (1 - forget_factor) if 0 <= forget_factor < 1 else 1
    data = data.round(6)

    def get_last_instant_feat(data_, feat_):
        c_cum = [c for c in data_.columns if c.startswith(feat_)]
        inst_cum = [c for c in c_cum if c.endswith("j")][-1]
        i_c = inst_cum.split(f'{v_cum}_')[-1].split('j')[0]
        return feat_, i_c

    for v_cum in feature_to_cumulate:
        if by_hist is True:
            cum_depth_col = [c for c in data.columns if c.startswith(v_cum)]
            data[f"{v_cum}_cum"] = data[cum_depth_col].sum(axis=1)
        else:
            if instant_ == "last":
                _, instant = get_last_instant_feat(data_=data, feat_=v_cum)
            else:
                instant = int(instant_)
            limit_max = data.shape[0]
            newest_v = f"{v_cum}_{instant}j"
            data[f"{v_cum}_cum"] = data[newest_v].values
            n_i = 1
            loca = list(data.index)
            for h in range(n_i, min(event_gap, data.shape[0])):
                val__ = lmbda_fgt * data.loc[loca[h - 1], f"{v_cum}_cum"] + data.loc[loca[h], newest_v]
                data.loc[loca[h], f"{v_cum}_cum"] = round(val__, 4)

            if event_gap < limit_max:
                for i in range(event_gap, limit_max, 1):
                    if data.loc[loca[i - event_gap:i], newest_v].sum() <= 0:
                        data.loc[loca[i], f"{v_cum}_cum"] = data.loc[loca[i], newest_v]
                    else:
                        val__ = lmbda_fgt * data.loc[loca[i - 1], f"{v_cum}_cum"] + data.loc[loca[i], newest_v]
                        data.loc[loca[i], f"{v_cum}_cum"] = round(val__, 4)
    return data
