import sys
from pathlib import Path
import pandas as pd
from utils.utils import get_data_path_for_basin_id


def adapt_inline_changes_data(data_real: pd.DataFrame, config_feat: list):
    """
    Sometimes, users have made some changes on the features name from raw data (to be AVOIDED as soon as possible), but
    models trained with these changes may struggle to re-use the same raw_data without the same exercise, whih is often
    forgotten. Here, this function tries to figure out if these changes are part of some preset registers and fix them.
    As an example, Q_lstm [dataframe] may have be renamed P_Qlstm [effective-use] to be taken as forecasted feature,
    then the keyword P_Q is used to identify cases where we have Q_lstm and rename it to P_Qlstm

    :param data_real: the raw dataframe
    :param config_feat: list of features known by the model
    :return: corrected dataframe
    """
    list_feat = config_feat
    data_col = [c for c in data_real.columns if not c.lower() == "date"]
    base_col = [c for c in list_feat if not c.lower() == "date"]

    # Check for features inn and not inn
    not_in = [c for c in base_col if c not in data_col]
    inn_ = [c for c in base_col if c in data_col]

    if "Q_Targ" in not_in:
        md_err = "Q_err_LSTM" if "P_Qlstm" in base_col else ("Q_err_SACSMA" if "P_QSACSMA" in base_col else None)
        data_real.rename(columns={"Q_Obs": "Q_Targ"}, inplace=True)
        data_real.rename(columns={md_err: "Q_Obs"}, inplace=True)
        not_in.remove("Q_Targ")
        inn_.append("Q_Targ")

    # TODO: those cases can be extended, may be try to generalize that, or store the mode from read_data to data_use
    case_PQ = [c for c in not_in if c.startswith("P_Q")]
    case_P = [c for c in not_in if c.startswith("P_") and c not in case_PQ]
    case_Q = [c for c in not_in if c.startswith("Q_") and c not in case_PQ]
    case_SWI = [c for c in not_in if c.startswith("SWI_")]

    add_in, re_nam = [], {}
    for case_str, case_ in zip(["P_", "Q_", "SWI_", "P_Q"], [case_P, case_Q, case_SWI, case_PQ]):
        if len(case_) != 0:
            for ccol_ in case_:  # if any feature hase been renamed, fix that in the loaded dataframe
                to_in = [c for c in data_col if c.endswith((ccol_.split(case_str)[-1]))][0]
                to_in = "Q_SAC_SMA" if ccol_ == "P_QSACSMA" else to_in  # FIXME: the P_QSACSMA is not well managed
                re_nam[to_in] = ccol_
                add_in.append(to_in)
    base_col = ["Date"] + inn_ + add_in
    data_ok = data_real[base_col]
    data_ok.rename(columns=re_nam, inplace=True)
    return data_ok


def apply_hybrid_mode_to_raw_data(raw_data: pd.DataFrame, mode_: str):
    """Check and assimilate other models outputs"""
    real_data = raw_data.copy()
    to_drop = []
    try:
        real_data.rename(columns={"SWI_CM_swe": "P_SWE"}, inplace=True)
    except KeyError:
        pass
    if mode_ == "no_in_mlp":
        to_drop = ["Q_err", "Q_lstm", "Q_SAC_SMA", "Q_err_LSTM", "Q_err_SACSMA"]
    if mode_ == "full_feat":
        to_drop = []
    if mode_ == "lstm_in_mlp":
        if not {"Q_lstm"}.issubset(real_data.columns):
            print("*"*10, mode_, "not complies with the passed data")
            sys.exit()
        to_drop = ["Q_err", "Q_SAC_SMA", "Q_err_LSTM", "Q_err_SACSMA"]
        real_data.rename(columns={"Q_lstm": "P_Qlstm"}, inplace=True)

    if mode_ == "sma_in_mlp":
        if not {"Q_SAC_SMA"}.issubset(real_data.columns):
            print("*"*10, mode_, "not complies with the passed data")
            sys.exit()
        to_drop = ["Q_err", "Q_lstm", "Q_err_LSTM", "Q_err_SACSMA"]
        real_data.rename(columns={"Q_SAC_SMA": "P_QSACSMA"}, inplace=True)

    if mode_ == "predict_lstm_error":
        if not {"Q_err_LSTM"}.issubset(real_data.columns):
            print("*"*10, mode_, "not complies with the passed data")
            sys.exit()
        real_data.rename(columns={"Q_lstm": "P_Qlstm", "Q_Obs": "Q_Targ"}, inplace=True)
        real_data.rename(columns={"Q_err_LSTM": "Q_Obs"}, inplace=True)
        to_drop = ["Q_err", "Q_SAC_SMA", "Q_err_SACSMA"]

    if mode_ == "predict_sma_error":
        if not {"Q_err_SACSMA"}.issubset(real_data.columns):
            print("*"*10, mode_, "not complies with the passed data")
            sys.exit()
        real_data.rename(columns={"Q_SAC_SMA": "P_QSACSMA", "Q_Obs": "Q_Targ"}, inplace=True)
        real_data.rename(columns={"Q_err_SACSMA": "Q_Obs"}, inplace=True)
        to_drop = ["Q_err", "Q_lstm", "Q_err_LSTM"]

    if len(to_drop) > 0:
        valid_to_drop = [c for c in to_drop if c in real_data.columns]
        try:
            real_data = real_data.drop(columns=valid_to_drop)
            if "Q_Obs" not in real_data.columns and mode_ in ["lstm_by_mlp", "sma_by_mlp"]:
                real_data.rename(columns={"Q_err": "Q_Obs"}, inplace=True)
        except KeyError:
            print("No drop")
    return real_data, mode_


def import_data(path, mode_: str = None) -> tuple[pd.DataFrame, str]:
    """
    This module helps dealing with a part of the issue that may occur when date is not in an uncommon format. Please
    avoid any date that starts on december 31st. It should even be better if the date was in YYYY-mm-dd format.
    This module checks only if day is first or not. No other case is guaranteed here.

    :param mode_: mode to run on, if no other model's output is needed as input, set mode_="no_in_mlp", set this to "full"
        for any evaluation mode
    :param path: the data path.
    :return: corresponding dataframe.
    """
    if mode_ is None:
        mode_ = "no_in_mlp"
    data_0 = pd.read_csv(path, sep=";")
    data_0 = data_0.loc[:, ~data_0.columns.str.contains('^Unnamed')]
    c_date = [c for c in data_0.columns if c.lower().startswith("date")]
    if len(c_date) == 0:
        data_0["date"] = data_0.index
    c_date = [c for c in data_0.columns if c.lower().startswith("date")][0]
    sep = "/" if "/" in str(data_0[c_date][0]) else "-"
    fst, lst = [data_0[c_date][i].split(sep)[0] for i in range(3)], \
               [data_0[c_date][i].split(sep)[2] for i in range(3)]
    fst_, lst_ = list(set(fst)), list(set(lst))
    day_first = True if len(fst_) > len(lst_) else False
    real_data = pd.read_csv(path, sep=";", dtype="float", parse_dates=[c_date], dayfirst=day_first)
    real_data = real_data.loc[:, ~real_data.columns.str.contains('^Unnamed')]
    real_data = real_data.fillna(method="ffill", axis=0)
    real_data = real_data.fillna(method="bfill", axis=0)
    return apply_hybrid_mode_to_raw_data(real_data, mode_)


def import_eval_data(path_: str, data_use: dict):
    """
    Import data according to a model set-up. Import the whole dataframe, then choose the features used by the model

    :param path_: path of the data
    :param data_use: usage info of the data by the preset-model

    :return: the data with only pre_used features
    """
    list_feat_ = list(data_use["raw_feat_and_wind_used"].keys())
    data_, _ = import_data(path=path_, mode_="full_feat")
    data_ = adapt_inline_changes_data(data_real=data_, config_feat=list_feat_)
    return data_


def get_adapted_data_from(data_path: str = None, list_feat=None) -> pd.DataFrame():
    """
    This module aims to adapt a random data to preformatted model's one in terms of columns name. It requires only the
    path of the data to be used.

    :param list_feat: pass the expected list feature
    :param data_path: the data path
    :return: the adapted data
    """
    data_real, _ = import_data(data_path, "full_feat")
    if list_feat is None:
        base_d = pd.read_csv("data/_base_operational_data.csv", sep=";")
        base_col = [c for c in base_d.columns]
        data_ok = data_real[base_col]
    else:
        data_ok = adapt_inline_changes_data(data_real=data_real, config_feat=list_feat)
    return data_ok


def map_raw_basin_attribute_to_cfg(basin_id, main_cfg: dict):
    """ Adapt config file according to the data features available """
    full_cfg_ = main_cfg.copy()
    path_ = get_data_path_for_basin_id(basin_id=basin_id, basin_data_folder=full_cfg_["global_setting"]["dataPath"])
    data, _ = import_data(path=path_, mode_="full_feat")

    if full_cfg_['global_setting']["hp"] == 0:
        for nk_ in ["future", "targetAsInput", "predictDta"]:
            full_cfg_['global_setting'][nk_] = False
        full_cfg_["batch_variables"]["meta_data"]["future_exception"]["use"] = False

    test_len = full_cfg_["global_setting"]["test_size"]
    if test_len is not None:
        test_len_ = test_len
        c_dte = [c for c in data.columns if c.lower().startswith("date")]
        c_dte = c_dte[0] if len(c_dte) > 0 else None
        date_ = list(data.index) if "date" not in data.columns.str.lower() else list(data[c_dte])
        start_T, end_T = date_[-test_len_], date_[-1]
        start_S, end_S = date_[-(test_len_ + 365)], date_[-(test_len_ + 1)]
        start_Tr, end_Tr = date_[30], date_[-(test_len_ + 1)]
        st_T, nd_T = f"{start_T.year}/{start_T.month}/{start_T.day}", f"{end_T.year}/{end_T.month}/{end_T.day}"
        st_S, nd_S = f"{start_S.year}/{start_S.month}/{start_S.day}", f"{end_S.year}/{end_S.month}/{end_S.day}"
        st_Tr, nd_Tr = f"{start_Tr.year}/{start_Tr.month}/{start_Tr.day}", f"{end_Tr.year}/{end_Tr.month}/{end_Tr.day}"

        full_cfg_["global_setting"]["trainStartDate"] = st_Tr
        full_cfg_["global_setting"]["trainEndDate"] = nd_Tr
        full_cfg_["global_setting"]["stopStartDate"] = st_S
        full_cfg_["global_setting"]["stopEndDate"] = nd_S
        full_cfg_["global_setting"]["testStartDate"] = st_T
        full_cfg_["global_setting"]["testEndDate"] = nd_T

    to_cum = [c for c in data.columns if c.startswith("P_")]
    old_cum = full_cfg_["batch_variables"]["meta_data"]["To_cumulate"]
    to_cum_here = [c for c in old_cum if c in data.columns]
    to_cum_ = to_cum_here if len(to_cum_here) != 0 else to_cum[:2]

    full_cfg_["global_setting"]["basin"] = basin_id
    full_cfg_["global_setting"]["data_name"] = basin_id

    full_cfg_["batch_variables"]["meta_data"]["data_name"] = basin_id
    full_cfg_["batch_variables"]["meta_data"]["BASIN_ID"] = basin_id
    full_cfg_["batch_variables"]["meta_data"]["TARGET_ID"] = "Q_Obs"
    full_cfg_["batch_variables"]["meta_data"]["To_cumulate"] = to_cum_
    return full_cfg_


def set_date_as_index(data):
    """Ensure data is date indexed"""
    data_ = data.copy()
    c_date = [c for c in data_.columns if c.lower().startswith("date")]
    if len(c_date) != 0:
        _date = c_date[0]
        data_.index = pd.to_datetime(data_[_date])
        data_.drop(columns=[_date], inplace=True, axis=1)
    return data_

