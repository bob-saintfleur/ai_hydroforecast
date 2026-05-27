import json
from utils.utils import get_data_path_for_basin_id, get_str_boolean
from datatools.read_data import map_raw_basin_attribute_to_cfg, import_data
from utils.logger import logger


def update_main_cfg(base_c=None, parser_arg=None):
    """ Update the main config file with the user's parsed basic arguments """
    if base_c is None:
        base_c = json.load(open("utils/main_config.json", "r+"))
    if parser_arg is None:
        from utils.args_getter import get_run_args
        parser_arg = get_run_args()
    if parser_arg.run_mode not in ["apply", "evaluate", "climatology", "hindcast", "realtime"]:
        if parser_arg.other_model_use is not None:
            base_c["global_setting"]["other_model_use"] = parser_arg.other_model_use
        if parser_arg.data_path is not None:
            base_c["global_setting"]["dataPath"] = parser_arg.data_path
        if parser_arg.basins_file is not None:
            base_c["global_setting"]["basins_file"] = parser_arg.basins_file
        if parser_arg.run_dir is not None:  # The model assimilated is identified in the model path
            base_c["global_setting"]["run_dir"] = parser_arg.run_dir + "/" + base_c["global_setting"]["other_model_use"]
        if parser_arg.hp is not None:
            base_c["global_setting"]["hp"] = parser_arg.hp
        if parser_arg.drop_inter_hp is not None:
            base_c["global_setting"]["drop_inter_hp"] = get_str_boolean(parser_arg.drop_inter_hp)
        if parser_arg.predictDta is not None:
            base_c["global_setting"]["predictDta"] = get_str_boolean(parser_arg.predictDta)
        if parser_arg.target_as_input is not None:
            base_c["global_setting"]["targetAsInput"] = get_str_boolean(parser_arg.target_as_input)
        if parser_arg.use_future is not None:
            base_c["global_setting"]["future"] = get_str_boolean(parser_arg.use_future)
        if parser_arg.focus_drought is not None:
            base_c["global_setting"]["seuil_target"]["use"] = get_str_boolean(parser_arg.focus_drought)
        if parser_arg.cv_size is not None:
            base_c["global_setting"]["cv"] = parser_arg.cv_size
        if parser_arg.seed is not None:
            base_c["global_setting"]["randomSeed"] = parser_arg.seed
        if parser_arg.n_jobs is not None:
            base_c["global_setting"]["n_jobs"] = parser_arg.n_jobs
        if parser_arg.test_period is not None:
            base_c["global_setting"]["testStartDate"] = parser_arg.test_period[0]
            base_c["global_setting"]["testEndDate"] = parser_arg.test_period[1]
        if parser_arg.train_period is not None:
            base_c["global_setting"]["trainStartDate"] = parser_arg.train_period[0]
            base_c["global_setting"]["trainEndDate"] = parser_arg.train_period[1]
            base_c["global_setting"]["test_size"] = None
            base_c["global_setting"]["use_following_dates"] = True
        logger.info("Main config : UPDATED")
        return base_c


def map_option_to_cfg_dict(base_c: dict, new_params_):
    """
    Apply user-specified options to a config before running.

    :param base_c: the loaded config on its dict format
    :param new_params_: dict of the new parameter value pair to map
    :return: the dict
    """
    if "hp" in new_params_.keys():
        base_c["global_setting"]["hp"] = new_params_["hp"]
    if "drop_inter_hp" in new_params_.keys():
        base_c["global_setting"]["drop_inter_hp"] = new_params_["drop_inter_hp"]
    if "run_dir" in new_params_.keys():
        base_c["global_setting"]["run_dir"] = new_params_["run_dir"]
    if "data_root" in new_params_.keys():
        base_c["global_setting"]["data_root"] = new_params_["data_root"]
    if "data_path" in new_params_.keys():
        base_c["global_setting"]["dataPath"] = new_params_["data_path"]
    if "future" in new_params_.keys():
        base_c["global_setting"]["future"] = new_params_["future"]
    if "seuil_target" in new_params_.keys():
        base_c["global_setting"]["seuil_target"]["use"] = new_params_["seuil_target"]
    if "target_as_input" in new_params_.keys():
        base_c["global_setting"]["targetAsInput"] = new_params_["target_as_input"]
    if "test_period" in new_params_.keys():
        base_c["global_setting"]["testStartDate"] = new_params_["test_period"][0]
        base_c["global_setting"]["testEndDate"] = new_params_["test_period"][1]
    if "train_period" in new_params_.keys():
        base_c["global_setting"]["trainStartDate"] = new_params_["train_period"][0]
        base_c["global_setting"]["trainEndDate"] = new_params_["train_period"][1]
        base_c["global_setting"]["test_size"] = None
        base_c["global_setting"]["use_following_dates"] = True
    if "predictDta" in new_params_.keys():
        base_c["global_setting"]["predictDta"] = new_params_["predictDta"]
    if "future_exception" in new_params_.keys():
        base_c["batch_variables"]["meta_data"]["future_exception"]["use"] = new_params_["future_exception"]
    if "all_window" in new_params_.keys():
        l_bv_k = [c for c in list(base_c["batch_variables"].keys()) if not c.startswith("meta_data")]
        l_dv_k = [c for c in list(base_c["detailed_variables"].keys()) if not c.startswith("meta_data")]
        if base_c["detailed_variables"]["use_me"]:
            for ky_ in l_dv_k:
                if base_c["detailed_variables"][ky_]["use"]:
                    base_c["detailed_variables"][ky_]["window"] = new_params_["all_window"]
        else:
            for ky_ in l_bv_k:
                if base_c["batch_variables"][ky_]["use"]:
                    base_c["batch_variables"][ky_]["window_size"] = new_params_["all_window"]
    if "use_cumulate" in new_params_.keys():
        base_c["global_setting"]["use_cumulate"] = new_params_["use_cumulate"]
    if "moving_average" in new_params_.keys():
        base_c["global_setting"]["add_moving_average"]["use"] = new_params_["moving_average"]
    if "batch_variables_reduction" in new_params_.keys():
        base_c["global_setting"]["use_variable_reduction"] = new_params_["batch_variables_reduction"]
    if "auto_windowing" in new_params_.keys():
        base_c["correlation_process"]["auto_windowing"] = new_params_["auto_windowing"]
    if "inertia" in new_params_.keys():
        base_c["correlation_process"]["auto_windowing"] = True
        base_c["correlation_process"]["inertia"] = new_params_["inertia"]
    return base_c


def setup_feature_config(data_path, cfg_dict: dict):
    """
    This function aims to map the features from a csv file to the config file to be used.

    :param cfg_dict: The config that will hold the features names
    :param data_path: INPUTS that contains the features with  codified names. ex for P_SF_UPSTREAM is a rainfall
    :return: The modified config at its location
    """
    mode_ = cfg_dict["global_setting"]["other_model_use"]
    data, _ = import_data(data_path, mode_)
    data_features = [c for c in data.columns if c.lower() not in ["date", "event"]]

    # get the features types according to the beginning of their names
    keys_features_data = list(set([c.split("_")[0] for c in data_features]))

    feat_cfg = cfg_dict.copy()
    var_keys_cfg = [c for c in feat_cfg["batch_variables"].keys() if not c.lower().startswith("meta_")]

    # get all aliases for these keys
    full_name_key_in_cfg = {feat_cfg["batch_variables"][a]["alias"]: a for a in var_keys_cfg}

    # Link all features from the dataframe to their corresponding keys in the config
    group_features = {k: [c for c in data_features if c.startswith(f"{k}_")] for k in keys_features_data}
    group_features = dict(sorted(group_features.items()))

    # map all sub_group_features to their corresponding list-names in the config
    check_list = ["Date"]
    key_in_data = []
    for alias, full_name in full_name_key_in_cfg.items():
        feat_cfg["batch_variables"][full_name]["use"] = False
        feat_cfg["batch_variables"][full_name]["list_names"] = [None]

    for alias, full_name in full_name_key_in_cfg.items():
        for key_sub_ft, sub_group_ft in group_features.items():
            if alias == key_sub_ft:
                key_in_data.append(full_name)
                feat_cfg["batch_variables"][full_name]["list_names"] = sub_group_ft
                feat_cfg["batch_variables"][full_name]["use"] = True
                check_list += sub_group_ft

    # Set as OTHER (@Alt) all features that come with no predefined keys, by default they are not used
    if set(data.columns) != set(check_list):
        other = [c for c in data.columns if c not in check_list]
        print(other, " left out")
        feat_cfg["batch_variables"]["OTHERS"]["list_names"] = other
        feat_cfg["batch_variables"]["OTHERS"]["use"] = False
        key_in_data.append("OTHERS")
    return feat_cfg


def prepare_config(basin_id, config0):
    """Prepare and adapt raw config"""
    config1 = map_raw_basin_attribute_to_cfg(basin_id, config0)
    bv_path = get_data_path_for_basin_id(basin_id=basin_id, basin_data_folder=config1["global_setting"]["dataPath"])
    config1 = setup_feature_config(data_path=bv_path, cfg_dict=config1)
    return config1
