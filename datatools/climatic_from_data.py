import random
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

dt = datetime.now()
yr, mt_, dy_ = dt.year, dt.month, dt.day


# TODO: This module is limited to daily data, it needs to be adapted for other timestep sample data


class ClimaticScenarios:
    def __init__(self, data,
                 target,
                 ref_date: tuple,
                 month_: list,
                 day_: list,
                 period_,
                 key_: str = "rainfall",
                 look_back: int = 20):
        """
        This module is set to ease the selection of a climatic sub-data by providing both month and day. It
        will use the period_ provided to extend the period in the future. This selection will concern all single year
        in the dataframe provided. An extra function is used to select the precedent state of the basin according to
        the cumulated rainfall and the mean flow over a lookback period of 20 days. The hydro-state function is designed
         to process the climatic sub-data selection given an assumption over the season's state [drought, normal, wet].
        The user can choose to use a data_state function where a value is expected as the median of a look-to-get
        sub-set of data. This value is linked to the key_ specified before. These two functions are in exclusive use.
        If the idea of a random selection is on the table, the random_sub(size) can be used.

        :param data: the data to be used, it is expected with at least the following columns date and data for rainfall,
            etp, discharge,...
        :param month_: an integer [1 , 12] standing for the classical months
        :param day_: the day, a number according to the choosing month. Do not use 29 for february
        :param period_: the length of the data to be extracted from every year starting from the provided date
        :param key_: The key feature on which the selection will consider. can be rainfall or flow
        :param look_back: the historic to consider in the past days in order to prepare the criteria of selection. This
            latter can be either the cumulated rainfall or the mean observed flow
        :param target: the target features, such as a discharge station's name or any indicated target feature

        """
        self.p_data = None
        if ref_date is None:
            ref_date = (yr - 2, 9, 1)
        self.ref_date = ref_date
        self.rf_c = None
        self.assumption = None
        self.keep_years = None
        self.prior_state = None
        self.ref_var = None
        self.far_right = None
        self.far_lest = None
        self.data = data
        self.month_ = month_
        self.day_ = day_
        self.period_ = period_
        self.key_ = key_
        self.look_back = look_back
        self.look_back_T = None
        self.n_members = None
        self.target_v = None
        self.target = target
        self.clim_all_subsets = None
        self.pluie = [c for c in data.columns if c.lower().startswith("p_")][0]
        self.debit = self.target
        rf_c = self.pluie
        if self.key_ in ["flow", "discharge", "debit", self.target]:
            rf_c = self.debit
        self.rf_c = rf_c

    def get_climatic_hist(self):
        """
        This module is set to ease the selection of a climatic sub-data by providing both month and day. It will
        use the period provided to extend the period in the future. This selection will concern all single year in the
        dataframe provided. An extra function is used to select the precedent state of the basin according to the
        cumulated rainfall and the mean flow over a lookback period of 20 days. This function is designed to process
        climatic scenarios on a model simulation or forecasting. The user can choose the year_state that he considers
        as closer to what he actually observed. While this last idea is not mandatory, the modeller can choose to use
        a random selection based on the year_state.

        :return: tuple(dict, prior_state). A dict holding all sub-periods with keys as 'year_xxxx', and values as sub
            dataframe selected. The prior_state as a dataframe holding the cumulated rainfall and the mean_flow over
            the look_back prior days.
        """
        data = self.data
        c_date_ = [c for c in data.columns if c.lower().startswith("date")]
        if len(c_date_) == 0:
            c_date = "Date"
            data[c_date] = data.index
        else:
            c_date = c_date_[0]
        data = data.loc[:, ~data.columns.duplicated()].copy()
        list_year = data[c_date].apply(lambda x: x.year).to_list()
        list_year = list(set(list_year))[1:]
        start_list, sub_key = [], []
        for y in list_year:
            for m in self.month_:
                for d in self.day_:
                    m_, d_ = f"{m}".zfill(2), f"{d}".zfill(2)
                    sub_key.append(f"year_{y}_{m_}{d_}")
                    start_list.append(datetime(y, month=m, day=d))
        lim_ = list(data[c_date])[-1]
        lim_ = lim_ - timedelta(days=self.period_ + 1)
        list_start = [c for c in start_list if c <= lim_]
        sub_period_list = [pd.date_range(start=k + timedelta(days=1), periods=self.period_,
                                         freq="D") for k in list_start]

        prior_state_date = [pd.date_range(start=k, periods=self.look_back,
                                          freq="-1D").sort_values() for k in list_start]

        list_end = [list(dd)[-1] for dd in sub_period_list]

        set_4_test = [pd.date_range(start=k, periods=self.look_back + self.period_ + 1,
                                    freq="-1D").sort_values() for k in list_end]

        fg, fp_, test_set = pd.DataFrame(sub_period_list[0], columns=["Date"]), \
            pd.DataFrame(prior_state_date[0], columns=["Date"]), \
            pd.DataFrame(set_4_test[0], columns=["Date"])
        for i in range(len(sub_period_list)):
            fg[f"{sub_key[i]}"] = pd.DataFrame(sub_period_list[i])
            fp_[f"{sub_key[i]}"] = pd.DataFrame(prior_state_date[i])
            test_set[f"{sub_key[i]}"] = pd.DataFrame(set_4_test[i])

        clim_data, prior_state, sub_data_sets = {}, {}, {}
        data.index = data["Date"]
        data.drop(columns=c_date, axis=1, inplace=True)
        self.p_data = data
        c_pluie, c_debit, c_year, d_prior = [], [], [], pd.DataFrame()
        for per in fg.columns[1:]:
            c_year.append(per)
            clim_data[per] = data.loc[pd.to_datetime(fg[per].values), :]
        start_d = f"{prior_state_date[0][0].day}/{prior_state_date[0][0].month}"
        end_d = f"{prior_state_date[0][-1].day}/{prior_state_date[0][-1].month}"
        self.look_back_T = [start_d, end_d]
        return clim_data

    def get_base_ref(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the reference date period for the specified reference date

        :return: the reference test and the linked subdata periods
        """
        ref_date = datetime(self.ref_date[0], self.ref_date[1], self.ref_date[2], 0, 0)
        ref_hp = pd.date_range(start=ref_date + timedelta(days=1), periods=self.period_)
        ref_sub_test = pd.date_range(start=ref_hp[-1], periods=self.look_back + self.period_,
                                     freq="-1D").sort_values()
        data = self.p_data.copy()
        ref_test = data.loc[ref_hp, :]
        ref_sub_test = data.loc[ref_sub_test, :]
        return ref_test, ref_sub_test

    def get_ref_attributes(self):
        """Get attributes of the actual member (year or reference)  """
        ref_date = datetime(self.ref_date[0], self.ref_date[1], self.ref_date[2], 0, 0)
        ref_back_state = pd.date_range(start=ref_date, periods=self.look_back, freq="-1D").sort_values()
        data = self.p_data.copy()
        ref_d = pd.date_range(start=ref_date, periods=1)
        key_val = data.loc[ref_d, self.rf_c].values[0]
        cum_state = data.loc[ref_back_state, self.rf_c].sum(axis=0)
        mean_state = data.loc[ref_back_state, self.rf_c].mean(axis=0)
        ref_val = {"ref_val": key_val, self.rf_c: cum_state if self.rf_c == self.pluie else mean_state}
        return ref_val

    def get_clim_data_by_hydro_state(self, assumption: str = "drought"):
        """
        This function makes the selection of the climatic sub_data on the basis of one of the given assumption. The
        latter corresponds to the state of the actual season looking back on look_back days.

        :param assumption: [drought, normal, wet] = f(q[0-0.25, 0.25-0.75, 0.75-1])
        :return: a dict wit the found years and the look back data
        """
        all_sub_clim, prior_state, all_subsets = self.get_climatic_hist()
        rf_c = self.rf_c
        sep_scenario = list(np.quantile(prior_state[rf_c], [0.25, 0.75]))
        prior_state = prior_state.sort_values(by=rf_c, ascending=True)
        self.assumption = assumption
        if assumption == "wet":
            sce_ = prior_state[prior_state[rf_c] >= sep_scenario[1]]
        elif assumption == "normal":
            mask = (prior_state[rf_c] >= sep_scenario[0]) & (prior_state[rf_c] <= sep_scenario[1])
            sce_ = prior_state.loc[mask]
        else:
            sce_ = prior_state[prior_state[rf_c] <= sep_scenario[0]]
        list_yr = list(sce_.index)

        print(f"\nReport:\nAssumption: {assumption.title()}\nKey feature: "
              f"Past observed {self.key_} \nFound: {list_yr}")
        sce_out = {k: v for k, v in all_sub_clim.items() if k in list_yr}
        all_state_out = {k: v for k, v in all_subsets.items() if k in list_yr}
        return sce_out, all_state_out

    def get_close_clim_data_state(self, n_members: int = 10):
        """
        This function selects the climatic sub_data based on a given threshold according to the present state of the
        target key feature. It requires as well the size of the ensemble to be considered in order to avoiding
        deterministic selection. It processes as well a sliding mean on the n_members and select the close half
        members on top and the other half on below. If one of the halves does not find the total member, it stops on
         the closest bound.

        :param n_members: Number of member to consider in the ensemble set
        :param target_v: the target value to get close to according to the mean (or cumulated) of the last lookback
            observed flow (or rainfall). If nothing is given, it considers the min * 1.1
        :return: a dict with the resulted lookback data, and a list of the found years or periods
        """
        # _, _, ref_val = self.get_base_ref()
        n_members = n_members if n_members >= 2 else 2
        n_members = n_members if n_members % 2 == 0 else n_members + 1
        all_sub_clim, prior_state, all_subsets = self.get_climatic_hist()
        rf_c = self.rf_c
        ref_att = self.get_ref_attributes()
        ref_state = ref_att[self.rf_c]
        rf_c = [c for c in prior_state.columns if rf_c in str(c)][0]
        prior_state = prior_state.sort_values(by=rf_c, ascending=True)
        prior_state[f"med_{n_members}"] = prior_state[rf_c].rolling(n_members, min_periods=2, center=True).mean()
        check_min = prior_state[f"med_{n_members}"].min()
        target_v = ref_state if ref_state > check_min else check_min

        ord_yr = list(prior_state.index)  # list of all year
        ind_x = prior_state[prior_state[f"med_{n_members}"] <= target_v].shape[0]  # Index of target value in ascending
        far_left = ind_x - n_members // 2 if (ind_x - n_members // 2) >= 0 else 0  # left and right boundary of target
        far_right = ind_x + n_members // 2 if (ind_x + n_members // 2) <= len(ord_yr) else len(ord_yr)
        list_yr_interest = ord_yr[far_left: far_right]
        print(f"\nTarget years for {rf_c} ~= {target_v} centered around and within {n_members} members are: \n")
        print(prior_state.loc[list_yr_interest, f"med_{n_members}"])

        dict_close_state = {a: b for a, b in all_sub_clim.items() if a in list_yr_interest}
        close_state_subs = {a: b for a, b in all_subsets.items() if a in list_yr_interest}
        self.far_lest, self.far_right, self.ref_var = far_left, far_right, rf_c
        self.prior_state = prior_state
        self.target_v, self.n_members, self.keep_years = target_v, n_members, list_yr_interest
        return dict_close_state, list_yr_interest, close_state_subs, target_v

    def get_random_sub(self, size: int = 10):
        """
        Make a random sampling of n subset to assess uncertainties of your built model.

        :param size: the size of the subset, it must be < the size of the whole, even if no restriction. if greater
            than the available list, there will be resample or doubling
        :return:
        """
        sub_obs, _, sub_test = self.get_climatic_hist()
        list_year = random.sample(list(sub_obs.keys()), size)
        print(f"\n Here are the {size} randomly sampled years: \n {list_year}")
        s_ob, sub_T = {}, {}
        for yy_ in list_year:
            s_ob[yy_] = sub_obs[yy_]
            sub_T[yy_] = sub_test[yy_]
        return s_ob, sub_T

