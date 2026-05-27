import collections
from glob import glob
import sys
import json
from datetime import timedelta
import os.path
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import shift
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from datatools.base_data import BaseData
from utils import utils
from utils.prepare_config import map_option_to_cfg_dict
from utils.logger import logger

warnings.filterwarnings("ignore")

save_stdout = sys.stdout
sys.stdout = open('../trash', 'w')

sys.stdout = save_stdout
""" End of silencing"""

time_file_s = utils.get_time_run()


def get_pre_cfg(cfg, mode_run, model_path, user_opt, check_on_global):
    """Get adapted config

    :params:
        - raw_cfg: config to be used
        - mode_run: mode run used
        - model_path: model path
        - user_opt: dict of user options
        - check_on_global: bool for global flow
    """
    status_orig = 0
    data_usage, cfg_md = None, None

    # start with raw config
    if mode_run in ["search", "train"]:
        with open("../_running_cfg.json", "w+") as fg:
            if user_opt is not None:
                # cfg = utils.map_option_to_cfg_dict(cfg, new_params_=user_opt)
                cfg = map_option_to_cfg_dict(cfg, new_params_=user_opt)
            json.dump(cfg, fg, indent=4)
        with open(f"{model_path}/config.json", "w") as tsv:
            json.dump(cfg, tsv, indent=4)

    # Gather pre-runs configs
    else:
        try:
            cfg_md = glob(rf"{model_path}/config.json")[0]
            dt_use = glob(rf"{model_path}/data_use_info.json")[0]
            data_usage = json.load(open(dt_use, "r"))
        except Exception as e:
            print(f"Config files expected in {model_path}: {e}")
            pass

        if check_on_global:
            cfg, status_orig = utils.change_etiage_over_global(rf"{cfg_md}")
        else:
            with open(cfg_md, "r") as fp:
                cfg = json.load(fp)
        json.dump(cfg, open("../_running_cfg.json", "w+"), indent=2)
    return cfg, status_orig, data_usage


class SklearnModels:
    """Prepare sklearn base models """

    def __init__(self, model_name, z_cfg=None, mode_run=None, n_seed: int or list = 10, discr_file: str = None,
                 verbose=0, show_plot: bool = False, check_on_global: bool = False, x_run_dict=None, time_file=None,
                 op_md_path: str = None, op_df: pd.DataFrame = None, dict_clim=None, opti_path: str = None):
        """
        This module stands as the base for the sklearn regressor models. Its accepts shortnames for these
        modules such as 'mlp_skl' for MLPRegressor. It does not return anything in particular, but all the output can
        be found in a folder according to the given model name. e.g..: ./runs/run_mlp_skl_****. This class code will be
        optimized in the upcoming version, since some arguments are not necessary

        :param model_name: the module shortname pre-configured. e.g mlp_skl
        :param z_cfg: the config to be used
        :param mode_run: any of 'search, apply or evaluate, train, operational'. Each mode has its goal. To create a
            model, one should use the 'search' mode. 'apply or evaluate' a fitted model. Or 'train' aka re-train a
            pre-fitted model within a particular setting, that is useful to avoid making a new gridsearchCV.
        :param n_seed: Number of time to run a model under a different initialisation. Bring up uncertainties.
        :param discr_file: No effect for 'search mode'. It is used to filter the models to evaluate, retrain, etc..
        :param x_run_dict : dict of options from user
        :param dict_clim: climatological ensemble data is passed as a dict {hist:val, members:{m_i:val_i}}
        :param op_df: dataframe for operational mode
        :param opti_path: path for optimized models
        :param show_plot: Show the plots while running. Not recommended, as the number of plots can be high onto screen.
        :param check_on_global: useful to evaluate a threshold on model on a global dataset.
        :return:
        """
        self.op_md_path = op_md_path
        self.z_cfg = z_cfg
        self.verbose = verbose
        self.model_name = model_name
        self._mode_run = mode_run
        self._discr_file = discr_file
        self.n_seed = n_seed
        self.show_plot = show_plot
        self.check_on_global = check_on_global
        self.x_run_dict = x_run_dict
        self.time_file = time_file
        self.op_df = op_df
        self.dict_clim = dict_clim
        self.dict_clim_ = None
        if opti_path is None:
            opti_path = "runs"
        self.opti_path = opti_path
        self.nb_run = n_seed if isinstance(n_seed, int) else (len(n_seed) if isinstance(n_seed, list) else 10)

        if self._mode_run == "search":
            self.path_f = self.set_base_dir()
        elif self._mode_run in ["operational", "climatology", "hindcast",
                                "realtime"] and self.op_md_path is not None:  # added on may 3
            self.path_f = self.op_md_path
        else:
            self.path_f = utils.get_path(self.opti_path, model_name=self.model_name, discriminator=discr_file)

        if self._mode_run.lower() in ["apply", "evaluate"]:
            child = "/evaluation_on_best"
        elif "climatology" in str(self._mode_run):
            child = "/climatology"
        elif "hindcast" in str(self._mode_run):
            child = "/hindcast"
        elif "realtime" in str(self._mode_run):
            child = "/realtime"
        else:
            child = ''

        self.child = child
        self.path_run_ = rf"{self.path_f}{self.child}"
        os.makedirs(self.path_run_, exist_ok=True)
        orig_status = None
        if self._mode_run in ["climatology"]:
            if self.dict_clim is not None:
                dict_clim_, temp_op_df = {}, None
                for yr_k, mbr_dfs in tqdm(self.dict_clim.items(), desc="Format Climato :Date->Members", leave=False):
                    by_mbr = ()
                    for mbr_, mbr_df in mbr_dfs.items():
                        self.data_L, orig_status = self._set_data(mbr_df)
                        temp_op_df, _ = self.data_L.get_data_op()
                        by_mbr += (temp_op_df[-1:],)
                    by_mbr = pd.concat(by_mbr, axis=0)
                    dict_clim_[yr_k] = by_mbr
                self.dict_clim_ = dict_clim_
        elif self._mode_run in ["hindcast", "realtime"]:
            if self.dict_clim is not None:
                dict_clim_, temp_op_df = {}, None
                for yr_k, mbr_dfs in tqdm(self.dict_clim.items(), desc="Format Hindcast batchs", leave=False):
                    by_mbr = ()
                    for mbr_, mbr_df in enumerate(mbr_dfs):
                        mbr_df = mbr_df.rename(columns={"SWI_CM_swe": "P_SWE"})
                        self.data_L, orig_status = self._set_data(mbr_df)
                        temp_op_df, _ = self.data_L.get_data_op()
                        by_mbr += (temp_op_df,)
                    by_mbr = pd.concat(by_mbr, axis=0)
                    dict_clim_[yr_k] = by_mbr
                self.dict_clim_ = dict_clim_
        else:
            self.data_L, orig_status = self._set_data()
        seed_space = [111]
        if self._mode_run in ["search", "train"]:
            if isinstance(n_seed, int):
                rng = np.random.default_rng(self.data_L.cfg['global_setting']["randomSeed"])
                seed_space = rng.integers(low=100, high=1e5, size=n_seed)
                # seed_space = np.random.randint(low=100, high=1e5, size=nb_run)
            elif isinstance(n_seed, list):
                seed_space = n_seed
            self.seed_space = seed_space

        self.data_L.data_use_info["Etiage_on_Global"] = self.check_on_global
        if orig_status == 1:
            self.data_L.data_use_info.update({"data_focus": "Etiage"})
        if self.mode_run not in ["search", "train"]:
            scaler_p = joblib.load(rf"{self.path_f}/scaler_param.joblib")
            self.X_scaler = scaler_p["X_scaler"]
            self.Y_scaler = scaler_p["y_scaler"]
        else:
            # self.X_train, self.X_test, self.y_train, self.y_test = self.data_L.get_xy_train_test()
            self.X_scaler = self.data_L.scaler_X
            self.Y_scaler = self.data_L.scaler_Y

        self.X_train, self.X_test, self.y_train, self.y_test = self.data_L.get_xy_train_test()
        self.pers_loss_ = make_scorer(self._loss_persist, greater_is_better=True)
        self._normalize_tg = self.data_L.normalize_tg
        self.cv_ = self.data_L.cv_
        self.n_jobs = self.data_L.n_jobs
        self.dta_used = self.data_L.use_dta
        self.other_model_used = self.data_L.data_use_info["other_model_use"]

    def _loss_persist(self, y_true, y_pred):
        """Set a custom persistence as a loss metric option"""
        hp = self.data_L.hp
        mask_test, mask_train = self.data_L.mask_test, self.data_L.mask_train
        A_O_I = mask_train if mask_train.values.reshape(-1, 1).shape == y_true.shape else mask_test
        use_rate = self.data_L.use_rate_tg
        y_true2 = y_true.reshape(-1, 1)
        y_pred2 = y_pred.reshape(-1, 1)
        y_naive = shift(y_true, hp, mode="nearest") if not self.data_L.use_dta else np.zeros_like(y_true)
        y_naive2 = y_naive.reshape(-1, 1)
        num_ = np.mean(np.square(y_true2 - y_pred2))
        deno_ = np.mean(np.square(y_true2 - (y_naive2 if hp > 0 else y_true2.mean())))
        if use_rate:
            num_ = np.mean(np.square(y_true2[A_O_I] - y_pred2[A_O_I]))
            deno_ = np.mean(np.square(y_true2[A_O_I] - (y_naive2[A_O_I] if hp > 0 else y_true2[A_O_I].mean())))
        pers_ = 1 - num_ / deno_
        return round(pers_, 4)

    def _set_data(self, ope_df: pd.DataFrame = None):
        """Get the formatted data"""
        cfg_, orig_status, data_usage = get_pre_cfg(self.z_cfg, self._mode_run, self.path_f, self.x_run_dict,
                                                    self.check_on_global)
        if ope_df is None:
            ope_df = self.op_df
        data_ = BaseData(modx_name=self.model_name, run_mode=self._mode_run, cfg_=cfg_,
                         pre_md_path=self.path_f, op_df=ope_df, data_use=data_usage)
        return data_, orig_status

    def set_base_dir(self):
        """Set model directory """
        g_cfg = self.z_cfg["global_setting"]
        tm_f = time_file_s if not self.time_file else self.time_file
        opti_path = g_cfg["run_dir"]
        base_dir = rf"{opti_path}/run_{self.model_name}_{self.nb_run}seeds_{tm_f}/"
        logger.info(f"RUN DIR : {base_dir}")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    @property
    def mode_run(self):
        """Mode run"""
        return self._mode_run

    @property
    def discr_file(self):
        """Model discriminant string"""
        return self._discr_file

    @mode_run.setter
    def mode_run(self, new_value):
        self._mode_run = new_value

    @discr_file.setter
    def discr_file(self, new_value):
        self._discr_file = new_value

    def search_best(self):
        """Train and optimize models"""
        self._mode_run = "search"
        self._search_best_models_sk()

    def apply_bests(self):
        """Apply or evaluate models"""
        self._mode_run = "apply"
        self._run_best_models()

    def _search_best_models_sk(self):
        """Train and optimized models"""
        self._mode_run = "search"
        data_l = self.data_L
        base_res = data_l.base_results
        base_res_tr = data_l.base_results_train

        param_u = utils.load_param_space(model_name=self.model_name)
        dir_best = self.path_f

        # data_l.save_data_info(dir_best)
        loss_c_hist = pd.DataFrame(index=np.arange(0, 51, 1))
        loss_v_hist = pd.DataFrame(index=np.arange(0, 51, 1))
        grid_hist = {}
        test_sc_hist, train_sc_hist, best_index_hist, best_par = [], [], [], []
        _res, _res_tr, ft_imp, raw_output = (
            pd.DataFrame(index=self.y_test.index), pd.DataFrame(index=self.y_train.index),
            pd.DataFrame(), data_l.base_results.copy())

        x_test = self.X_scaler.transform(self.X_test)
        x_train = self.X_scaler.transform(self.X_train)
        y_train = self.normalize_target(self.y_train) if self._normalize_tg else self.y_train

        cpt = 0
        all_best_md = []
        pbar = tqdm(self.seed_space, desc=f"TRAIN : ", position=0, leave=True, file=sys.stdout)
        logger.info("SEARCH STARTED")
        for seed in pbar:
            cpt += 1
            model = utils.load_model_instance(self.model_name)(random_state=seed)
            model_grid = GridSearchCV(model, param_u, n_jobs=self.n_jobs,
                                      cv=self.cv_,
                                      scoring=self.pers_loss_,
                                      refit=True, pre_dispatch='n_jobs',
                                      return_train_score=True, verbose=self.verbose)
            model_grid.fit(x_train, y_train.ravel())
            joblib.dump(model_grid.best_estimator_, dir_best + rf"best_model_seed{seed}.pkl")
            all_best_md.append(model_grid)
        pbar.close()

        pbar_ev = tqdm(all_best_md, desc=f"EVAL: ", position=0, leave=True, file=sys.stderr)
        for md_g, seed in zip(pbar_ev, self.seed_space):
            score_ = md_g.score(x_test, self.y_test.ravel())
            pred_ = md_g.predict(x_test)
            raw_output[f"seed{seed}"] = np.round(pred_, 3)
            pred_tr = md_g.predict(x_train)
            _res[f"seed{seed}"] = self.add_up_dta(self.un_normalize_pred(pred_)
                                                  if self._normalize_tg else pred_, train_set=False)
            _res_tr[f"seed{seed}"] = self.add_up_dta(self.un_normalize_pred(pred_tr)
                                                     if self._normalize_tg else pred_tr, train_set=True)

            # plot loss curve
            if "mlp" in self.model_name:
                if md_g.best_estimator_.loss_curve_:
                    loss_c_hist[f"score_train_sd{seed}"] = pd.Series(md_g.best_estimator_.loss_curve_)
                    loss_v_hist[f"score_val_sd{seed}"] = 1 - pd.Series(md_g.best_estimator_.validation_scores_)

            grid_hist[f"seed_{seed}"] = md_g.cv_results_
            test_sc_hist.append(np.round(score_, 3))
            train_sc_hist.append(np.round(md_g.best_score_, 3))
            best_index_hist.append(md_g.best_index_)
            best_par.append(md_g.best_params_)
        pbar_ev.close()

        try:
            _res = pd.concat([base_res[["y_obs", "y_naive"]], _res], join="inner", axis=1)
            _res_tr = pd.concat([base_res_tr[["y_obs", "y_naive"]], _res_tr], join="inner", axis=1)
            if data_l.use_dta is True or "error" in self.other_model_used:
                ts = pd.to_datetime(_res.index).to_series().diff().mode()[0]
                _res.index = pd.to_datetime(_res.index) + ts * data_l.hp
                _res_tr.index = pd.to_datetime(_res_tr.index) + ts * data_l.hp

            if "error" in self.other_model_used:
                _res = clean_epp(self.path_f, _res)
                _res_tr = clean_epp(self.path_f, _res_tr)

            _res, _res_tr = _res.astype("float32"), _res_tr.astype("float32")
            _res.to_csv(dir_best + f"prediction_on_{self.nb_run}seeds_test.csv", sep=";", index_label="Date")
            _res_tr.to_csv(dir_best + f"prediction_on_{self.nb_run}seeds_train.csv", sep=";", index_label="Date")

            model_grid, seed = all_best_md[-1], self.seed_space[-1]
            bst_grid = {k: v for k, v in collections.Counter(best_index_hist).items()}
            logger.info(f"Best gridsearch stat: {bst_grid}")

            results = {"prediction": _res,
                       "pred_test": np.median(_res, axis=1),
                       "pred_train": np.median(_res_tr, axis=1),
                       "model_name": self.model_name,
                       "train_score": train_sc_hist,
                       "test_score": test_sc_hist,
                       "best_params": model_grid.best_params_,
                       "best_model": model_grid.best_estimator_}

            joblib.dump(results, dir_best + f"z_training_backup.pkl")
            data_l.save_data_info(path_=dir_best)
            data_l.save_scaler_param(path_cfg=self.path_f)
            logger.info("SEARCH COMPLETE")
            return results

        except (RuntimeError, TypeError, ValueError, KeyError, IndexError) as e:
            _, _, exc_tb = sys.exc_info()
            file_name_ = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.warning(msg=f"Exception: {e} on file {file_name_}")
            pass

    def _run_best_models(self):
        """Re-apply or evaluate the trained models"""
        data_L = self.data_L
        base_res = self.data_L.base_results
        base_res_tr = self.data_L.base_results_train
        x_test = self.X_scaler.transform(self.X_test)
        x_train = self.X_scaler.transform(self.X_train)
        list_best = glob(rf"{self.path_f}/*best_model*")
        _res, _res_tr = pd.DataFrame(index=self.y_test.index), pd.DataFrame(index=self.y_train.index)

        cpt = 0
        pbar = tqdm(list_best, desc=f"EVAL : ", position=0, leave=True, file=sys.stdout)
        for md in pbar:
            idm = md.split("best_model_")[-1].split(".")[0]
            modx = joblib.load(md)
            pred_T = modx.predict(x_test)
            pred_Tr = modx.predict(x_train)

            _res[f"{idm}"] = (
                self.add_up_dta(self.un_normalize_pred(y=pred_T) if self._normalize_tg else pred_T, train_set=False))
            _res_tr[f"{idm}"] = (
                self.add_up_dta(self.un_normalize_pred(y=pred_Tr) if self._normalize_tg else pred_Tr, train_set=True))
            cpt += 1
        pbar.close()
        try:
            ts = pd.to_datetime(base_res.index).to_series().diff().mode()[0]
            _res = pd.concat([base_res[["y_obs", "y_naive"]], _res], join="inner", axis=1)
            _res_tr = pd.concat([base_res_tr[["y_obs", "y_naive"]], _res_tr], join="inner", axis=1)
            if data_L.use_dta is True or "error" in self.other_model_used:
                _res.index = pd.to_datetime(_res.index) + ts * data_L.hp
                _res_tr.index = pd.to_datetime(_res_tr.index) + ts * data_L.hp

            if "error" in self.other_model_used:
                _res = clean_epp(str(Path(self.path_run_).parent), _res)
                _res_tr = clean_epp(str(Path(self.path_run_).parent), _res_tr)

            _res, _res_tr = _res.astype("float32"), _res_tr.astype("float32")
            _res.to_csv(rf"{self.path_run_}/prediction_test.csv", sep=";", index_label="Date")
            _res_tr.to_csv(rf"{self.path_run_}/prediction_train.csv", sep=";", index_label="Date")
            logger.info(f"Results path : {self.path_run_}/")

        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            file_name_ = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.warning(msg=f"Exception: {e} on file {file_name_}")
            pass

    def climatology(self):
        """ Run climatology mode """
        try:
            data_use_md = json.load(open(f"{self.path_f}/data_use_info.json", "r"))
            d_path = json.load(open(self.path_f + "/config.json", "r"))["global_setting"]["dataPath"]
            other_md = data_use_md["other_model_use"]  # kind of model assimilated
            bv_x = data_use_md["data_name"]
            hp_x = data_use_md["hp"]
            P_Qvar = [c for c in data_use_md["futured_var"] if c.startswith("P_Q")]  # the feature from the extra_model
            # inject if necessary the precomputed climatology from the extra model
            OUT_CLIM = ()
            for ref_date, x_members in self.dict_clim_.items():
                if (other_md != "no_in_mlp") & (len(P_Qvar) != 0):
                    bm_clim_str = str(Path(d_path).parent) + "/climato_bm"
                    bm_clim_path = bm_clim_str if Path(bm_clim_str).is_dir() else "../CLIMATOLOGY"  # TODO, to be optimized
                    P_Qx = P_Qvar[0]
                    other_mx = "lstm" if "lstm" in other_md else ("sacsma" if "sma" in other_md else None)
                    alt_md_clim = pd.read_csv(f"{bm_clim_path}/{other_mx}/hp{hp_x}/{bv_x}.csv",
                                              sep=";", parse_dates=True, index_col="Date").filter(like="yr")
                    target_date = pd.Timestamp(ref_date) + timedelta(days=hp_x)
                    clim_ref = alt_md_clim.loc[target_date]
                    clim_ref = clim_ref.fillna(clim_ref.mean(numeric_only=True))
                    x_members = x_members[:min(clim_ref.shape[0], x_members.shape[0])]
                    x_members[f"{P_Qx}_{hp_x}j"] = clim_ref.values[:min(clim_ref.shape[0], x_members.shape[0])]
                # populate the evaluation with the adapted matrix
                x_test = self.X_scaler.transform(x_members)
                list_best = glob(rf"{self.path_f}/*best_model*")
                raw_output = pd.DataFrame()
                for i, md in enumerate(list_best):
                    modx = joblib.load(md)
                    pred_ = modx.predict(x_test)
                    raw_output["s" + f"{i + 1}".zfill(2)] = np.round(pred_, 3)
                raw_output.index = ["m" + f"{i + 1}".zfill(2) for i in range(raw_output.shape[0])]
                raw_output = raw_output.stack().to_frame(ref_date).T
                raw_output.index.name = "Date_now"
                OUT_CLIM += (raw_output,)
            OUT_CLIM = pd.concat(OUT_CLIM, axis=0)
            OUT_CLIM.columns = [f'{a}_{b}' for a, b in OUT_CLIM.columns]
            return OUT_CLIM

        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            file_name_ = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.warning(f"Exception {e}. Check in {file_name_} at Line: {exc_tb.tb_lineno}")
            pass

    def hindcast(self):
        """ Run hindcast mode """
        try:
            data_use_md = json.load(open(f"{self.path_f}/data_use_info.json", "r"))
            d_path = json.load(open(self.path_f + "/config.json", "r"))["global_setting"]["dataPath"]
            other_md = data_use_md["other_model_use"]  # kind of model assimilated
            bv_x = data_use_md["data_name"]
            hp_x = data_use_md["hp"]
            P_Qvar = [c for c in data_use_md["futured_var"] if c.startswith("P_Q")]  # the feature from the extra_model
            list_best = glob(rf"{self.path_f}/*best_model*")

            # inject if necessary the precomputed climatology from the extra model
            OUT_PUTS = ()
            for ref_date, x_members in self.dict_clim_.items():
                if (other_md != "no_in_mlp") & (len(P_Qvar) != 0):
                    bm_clim_path = str(Path(d_path).parent) + f"/{self.mode_run}_bm"
                    P_Qx = P_Qvar[0]
                    other_mx = "lstm" if "lstm" in other_md else ("sacsma" if "sma" in other_md else None)
                    alt_md_clim = pd.read_csv(f"{bm_clim_path}/{other_mx}/hp{hp_x}/{bv_x}.csv",
                                              sep=";", parse_dates=True, index_col="Date").filter(like="yr")
                    target_date = pd.Timestamp(ref_date) + timedelta(days=hp_x)
                    clim_ref = alt_md_clim.loc[target_date].values
                    clim_ref = clim_ref.fillna(clim_ref.mean(numeric_only=True))
                    x_members[f"{P_Qx}_{hp_x}j"] = clim_ref[:min(x_members.shape[0], len(clim_ref))]

                # populate the evaluation with the adapted matrix
                x_test = self.X_scaler.transform(x_members)
                raw_output = pd.DataFrame()
                for i, md in enumerate(list_best):
                    modx = joblib.load(md)
                    pred_ = modx.predict(x_test)
                    raw_output["s" + f"{i + 1}".zfill(2)] = np.round(pred_, 3)
                raw_output.index = ["m" + f"{i + 1}".zfill(2) for i in range(raw_output.shape[0])]
                raw_output = raw_output.stack().to_frame(ref_date).T
                raw_output.index.name = "Date_now"
                OUT_PUTS += (raw_output,)
            OUT_PUTS = pd.concat(OUT_PUTS, axis=0)
            OUT_PUTS.columns = [f'{a}_{b}' for a, b in OUT_PUTS.columns]
            return OUT_PUTS

        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            file_name_ = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.warning(f"Exception {e}. Check in {file_name_} at Line: {exc_tb.tb_lineno}")
            pass

    def run(self, mode: str = None):
        """Run a mode"""
        if mode:
            if mode in ["apply", "evaluate", "search", "climatology", "hindcast", "realtime"]:
                self._mode_run = mode
        if self._mode_run in ["apply", "evaluate"]:
            self.apply_bests()
        elif self._mode_run == "search":
            self.search_best()
        elif self._mode_run == "climatology":
            climate_ = self.climatology()
            return climate_
        elif self._mode_run in ["hindcast", "realtime"]:
            hc_ = self.hindcast()
            return hc_
        else:
            logger.error("Available run_mode are : 'apply, search, climatology", "hindcast", "realtime")
            sys.exit()

    def normalize_target(self, y):
        """Normalize target"""
        return self.Y_scaler.transform(np.array(y).reshape(-1, 1))

    def un_normalize_pred(self, y):
        """Denormalize target"""
        return self.Y_scaler.inverse_transform(np.array(y).reshape(-1, 1)).ravel()

    def add_up_dta(self, prediction, train_set: bool = False):
        """Handle prediction and dta"""
        if train_set is True:
            base_0 = self.data_L.base_results_train["y_t0"]
        else:
            base_0 = self.data_L.base_results["y_t0"]
        base_0 = base_0 if self.data_L.use_dta is True else 0.
        out_ = prediction + base_0
        if "error" not in self.other_model_used:
            out_[out_ < 0] = 0.0
        return out_


def clean_epp(md_path: str, out_pp: pd.DataFrame):
    """
    Adapt simulated error based on original simulation

    :param md_path: path of the running model
    :param out_pp: output of simulated error. The index must have been adapted accordingly
    """
    gb_cfg = json.load(open(md_path + "/config.json", "r"))["global_setting"]
    o_m_u, basin, d_path, hp = gb_cfg["other_model_use"], gb_cfg["basin"], gb_cfg["dataPath"], gb_cfg["hp"]
    data_raw = pd.read_csv(d_path + f"/{basin}.txt", sep=";", index_col=0, parse_dates=[0])
    naive = data_raw[["Q_Obs"]].shift(hp, axis=0).fillna(method="bfill")
    naive.columns = ["y_naive"]
    to_pp = out_pp.copy()
    if "sma" in o_m_u:
        to_pp = to_pp.add(data_raw.Q_SAC_SMA.loc[to_pp.index].values, axis=0)
    if "lstm" in o_m_u:
        to_pp = to_pp.add(data_raw.Q_lstm.loc[to_pp.index].values, axis=0)
    to_pp["y_naive"] = naive.loc[to_pp.index, "y_naive"]
    to_pp[to_pp < 0.] = 0.
    return to_pp
