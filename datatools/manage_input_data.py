import numpy as np
import pandas as pd
from scipy.ndimage import shift
import warnings
warnings.filterwarnings("ignore")


class DataReFormat(object):
    def __init__(self, data: pd.DataFrame(), basin: str = None, target: str = None, fut_var_list: list = None,
                 use_target_as_input: bool = False, hp: int = None, future: bool = False, use_dta: bool = False):
        self.fut_var_list = fut_var_list
        self.data = data
        self.basin = basin
        self.target = target
        self.use_target = use_target_as_input
        self.hp = hp
        self.use_dta = use_dta
        self.y_t0 = None
        self.y_naive = None
        self.y_obs = None
        self.future = future
        self.fut = self.hp if self.future else 0

        # check the types of the data
        name_col = [col for col in self.data.columns]
        self.col_value = [col for col in self.data.columns if self.data.dtypes[col] in ["float", "int"]]
        self.col_other = [col for col in self.data.columns if self.data.dtypes[col] not in ["float", "int"]]

        col_date = [col for col in self.data.columns if str(col.lower()).startswith("date")][0]

        self.date_c = pd.to_datetime(self.data[col_date].values)
        not_num = self.col_other
        df_cl = self.data.drop(columns=not_num)
        df_cl.insert(0, "Date", self.date_c)
        df_cl.insert(0, "Event", self.data[col_date].apply(lambda x: x.year))
        _obs_t0 = self.data[[c for c in self.data.columns if (str(c).lower()) == self.target.lower()]]
        _obs_t0 = np.array(_obs_t0, "float32").ravel()
        self.y_obs = shift(_obs_t0, -self.hp, mode="nearest")
        self.y_t0 = _obs_t0
        df_cl["QObs"] = self.y_obs.ravel()
        self.y_naive = self.y_t0
        targ = [col for col in self.data.columns if (str(col).lower()).startswith(str(self.target).lower())][0]
        self.targ = targ
        if self.use_target:
            df_cl[targ] = self.y_t0
            self.df_classic = df_cl
        else:
            self.df_classic = df_cl.drop(columns=targ)

        if self.use_dta:
            if self.hp > 0:
                self.df_classic["QObs"] = np.array(self.y_obs - self.y_t0, 'float32').ravel()

    def data_classic(self):
        """Get the classic dataset """
        return self.df_classic

    def make_dta_wrt_hp(self):
        """Make dta in respect of hp"""
        dta = pd.DataFrame()
        _tar = self.targ
        _obs_t0 = np.array(self.data[_tar])
        if self.hp > 0:
            dta["Obs_t0"] = self.y_t0.ravel()
            dta[f"Obs_t{self.hp}"] = self.y_obs.ravel()
            dta[f"dta{self.hp}_QObs"] = np.round(self.data_classic()["QObs"].values, 3)
            return dta
        else:
            print("\n", "!-!" * 2, ".... no dt since hp=0 ....")
            return None

    def data_for_xcorrelation(self):
        """ Make cross correlation"""
        data = self.data_classic()
        cols = [col for col in data.columns if data.dtypes[col] not in ["float32", "float64"]]
        targ = [col for col in self.data.columns if (str(col).lower()).startswith(str(self.target).lower())]
        data_xc = data.drop(columns=cols)
        data_xc["QObs"] = self.data[targ]
        return data_xc


class InputWindowing(object):
    """
    Sequence data according to their inputs arguments
    """

    def __init__(self, data: pd.DataFrame(), opt_lag: dict = None):
        self.data = data.copy()
        self.opt_lag = opt_lag
        self.limit_lag = None
        self.original_cols = self.data.columns  # keep the original columns somewhere
        self.lag_wind = opt_lag
        var_val = {}

        if not dict(self.opt_lag):
            print(" >> With auto_windowing=False, opt_lag must be a dictionary mapping input to windows values")

        elif not len(self.opt_lag) <= len(self.original_cols):
            print(" >|| The number of the variables must match the input data size")

        else:
            for col in self.opt_lag.keys():
                if col not in self.original_cols:
                    print(" >|| The specified variables must be present in the dataFrame\n")
                    print(f" >> The available input names are {list(self.original_cols)}\n")
                    self.call_exit = True
                    break

                else:
                    self.lag_win = {}
                    for var, lag in self.opt_lag.items():
                        var_val[var] = lag

            self.lag_wind = var_val

    def get_lag(self):
        return self.lag_wind

    def apply_lag(self, hp_target: int = 0, future: bool = False, dict_future: dict = None):
        """
        Applying control on the auto-calculated lag
        """
        applying_lag = self.get_lag()
        # check variable-correl in pd.dataframe
        all_wind = applying_lag.values()
        the_cols = applying_lag.keys()

        if future is False:
            for var, wind in applying_lag.items():
                if var in self.data.columns:
                    self.data[var] = self.data[var].shift((wind - max(all_wind))).fillna(method="ffill")
                    for shft in range(int(wind)):
                        self.data[f"{var}_{shft - wind + 1}j"] = self.data[var].shift(-shft).fillna(method="ffill")

            self.data.drop(the_cols, axis=1, inplace=True)
            unshifted_cols = [col for col in self.data.columns if col in self.original_cols]

            for col in unshifted_cols:
                self.data[col] = self.data[col].shift(-max(all_wind)).fillna(method="ffill")
            data_shifted = self.data.iloc[:-(max(all_wind) - 1), :]

            if hp_target:
                data_shifted["QObs"] = shift(data_shifted["QObs"], -hp_target, mode="nearest")

        else:
            # create an adapted windows length in order to warranty no hp greater than any window length
            adapted_lag_windows = {}
            fut_list = []
            for i_var, i_val in applying_lag.items():
                if hp_target > i_val:
                    fut_list.append(i_var)
                adapted_lag_windows[i_var] = i_val  # hp_target
            applying_lag = adapted_lag_windows

            # No future lag must be greater than hp value
            adapted_fut_windows = {}
            if not dict_future:
                # if no data is provided for the futuring, use the available input_lag
                for vr, vl in applying_lag.items():
                    adapted_fut_windows[vr] = hp_target
            else:
                for fut_var, fut_val in dict_future.items():
                    if hp_target < fut_val:
                        adapted_fut_windows[fut_var] = hp_target
                    else:
                        adapted_fut_windows[fut_var] = fut_val

            # Apply the lag on the inputs
            for var, wind in applying_lag.items():
                if var in self.data.columns:
                    adaptor_shift = wind - max(applying_lag.values())
                    self.data[var] = shift(self.data[var], adaptor_shift, mode="nearest")
                    fut_here = adapted_fut_windows[var]
                    c_wind = wind + fut_here
                    for shft in range(int(c_wind)):
                        marker = shft - c_wind + 1 + (hp_target if fut_here > 0 else 0)
                        self.data[f"{var}_{marker}j"] = shift(self.data[var], -shft, mode="nearest")

            self.data.drop(the_cols, axis=1, inplace=True)
            unshifted_cols = [col for col in self.data.columns if col in self.original_cols]
            for col in unshifted_cols:
                self.data[col] = self.data[col].shift(-max(applying_lag.values())).fillna(method="ffill")
            data_shifted = self.data.iloc[:-(max(all_wind)+hp_target - 1), :]
        return data_shifted
