#!/usr/bin/env python3
# coding: utf-8

"""Supplied rising-flank c-GC implementation.

This source is retained as the canonical estimator implementation. The edits
in this module repair invocation and optional-import defects required to run
the supplied ``fit_rising`` path from the package adapter.
"""

import logging
from typing import Optional

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress reporting is optional
    def tqdm(iterable, **_kwargs):
        return iterable


#####################################################
#####################################################
### Rising Flanks
#####################################################
#####################################################

class RisingFlanks:
    """
        Implementation of Granger causality from causal Bayesian network
        perspective using only the rising flanks of the calcium imaging data.
        Methods:
            fit(self, data: np.ndarray)
            get_connectivity_matrix(self)
            plot_conn_mat_on_topography(self)
        """

    corr_: Optional[np.ndarray]
    pVal_corr_: Optional[np.ndarray]
    inv_corr_: Optional[np.ndarray]
    pVal_inv_corr_: Optional[np.ndarray]


    def __init__(self, n_perm: int, n_pasts: int, n_lags: int,
                                        f_s: float, seg_len: int):
        """
        Args:
            n_perm: number of permutations (default = 1000)
            n_pasts: number of past states
            n_lags: maximum allowable lags
            temporal: defines if data is time series or iid
        """

        logging.basicConfig(level=logging.INFO)
        self.logger = None  # Initialize logger when needed

        self.n_neur = None
        self.n_perm = n_perm
        self.n_pasts = n_pasts
        self.n_lags = n_lags
        self.seg_len = seg_len
        self.fs = f_s

        self.data = None
        self.shifted_data = None
        self.topography = None

        self.corr_ = None
        self.pVal_corr_ = None
        self.inv_corr_ = None
        self.pVal_inv_corr_ = None

    def moving_avg(self, data, win_size):
        """

        Args:
            data:
            win_size:

        Returns:

        """
        self.data = data.copy()
        i, moving_avg = 0, []
        while i < (len(data) - win_size + 1):
            window = data[i:i + win_size]
            win_avg = round(sum(window) / win_size, 4)
            moving_avg.append(win_avg)
            i += 1
        return moving_avg


    def ideal_lp(self, f_c, M):
        """

        Args:
            f_c:
            M:

        Returns:

        """
        amp = np.ones(M)
        amp[int(f_c * M / self.fs):-int(f_c * M / self.fs)] = 0
        phase = np.zeros(M)
        H_f = amp * np.exp(1j * phase)
        h_n = np.fft.fftshift(np.real(np.fft.ifft(H_f)))
        return h_n


    def shift_data(self, arr: np.ndarray) -> np.ndarray:   # , nn = None
        """
        Creates shifted versions of original data based on the number of pasts required.
        Concatenate the shifted arrays and return the new data.

        Examples: Given original data as X,
            trimmed_arr = __shifted_data(X, n_past = n)
            with trimmed_arr = {X_{t}, X_{t-1}, \\dots, X_{t-n}}^T
        Args:
            arr: original data recorded or obtained from experiments
            n_pasts: number of pasts defining the number of shifts

        Returns:
            Shifted data.
        """
        # if nn != None:
        #     self.n_pasts = nn

        self.data = arr.copy()
        self.n_neur = self.data.shape[0]
        if self.n_pasts == 0:
            return arr

        trimmed_arr = arr[:, self.n_pasts:]
        for i in range(self.n_pasts):
            idx1 = self.n_pasts - 1 - i
            idx2 = -i - 1

            trimmed_arr = np.r_[trimmed_arr, arr[:, idx1:idx2]]
        self.shifted_data = trimmed_arr
        return trimmed_arr



    @staticmethod
    def _absolute_corr(x: np.ndarray, y: np.ndarray) -> float:
        """Return a finite absolute correlation for two selected traces."""

        if x.size < 2 or y.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.abs(np.corrcoef(x, y)[1, 0]))

    def __perm_test_(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes p_value of correlation of two iid generated variables.
        Args:
        x: (array like, vector): Realisation of a variable x
        y: (array like, vector): Realisation of a variable y

        Returns:
            p_value
        """
        if self.n_perm <= 0 or x.size < 2 or y.size < 2:
            return 1.0

        count, corr_1 = 0, self._absolute_corr(x, y)
        val = np.random.randint(1, len(x), self.n_perm)

        for j in range(self.n_perm):
            x_ = np.roll(x, val[j])
            corr_2 = self._absolute_corr(x_, y)

            if corr_2 >= corr_1:
                count += 1

        return count / self.n_perm


    def get_past(self, X: np.ndarray) -> np.ndarray:
        """
            Creates shifted versions of original data based on the number of pasts required.
            Concatenate the shifted arrays and return the new data.
        Args:
            X: np.ndarray of shape (num_vars, num_timesteps)
            n_past: int, number of timelags to consider

        Returns:
            np.ndarray of shape (n_past+1, num_vars, num_timesteps-n_past) with lags
            increasing along the first axis.
        """
        assert len(X.shape) == 2, \
            "X must be a 2-dimensional array"
        if self.n_pasts == 0:
            return X.copy().reshape(1, *X.shape)
        past_matrices = []
        for j in range(self.n_pasts + 1):
            X_past_j = X[:, self.n_pasts - j:X.shape[1] - j]
            past_matrices.append(X_past_j)
        X_past = np.stack(past_matrices)
        return X_past



    def get_conditioning_set(self, i: int, j: int) -> np.ndarray:
        """
        Identifies the variables in the conditioning set for
        Granger causality implementation.
        All indices in range(0, X.shape[0]) that are not `i_ind` are
        considered to be indices of latent variables.

        Args:
            i: (int) index of the cause variable
            j: (int) index of the effect variable

        Returns:
            The conditioning set containing relevant variables of type np.ndarray
            with 2-dimension, where each row represents a variable
            conditioned variable, and each column includes the historical
            values of said variable.

        """
        X = self.data.copy()
        num_vars = self.n_neur


        j_ind = j
        i_ind = i % num_vars
        i_lag = i // num_vars


        X_past = self.get_past(X)
        # get the latent variable indices
        all_indices = np.arange(num_vars)
        ij_mask = np.isin(all_indices, [i_ind, j_ind])
        z_indices = all_indices[~ij_mask]  # everything that isn't i or j is z
        # `X_past` has shape (n_past, num_vars, X.shape[1]-n_past)

        # From the independent variable, we want to return everything before but
        # not including the "current" value at `i_lag`
        i_past = X_past[i_lag + 1:, [i_ind], :]
        # i_past shape (history up to i_lag, 1, X.shape[1]-n_past)

        # For the latent variable, we want to return everything up to and at the
        # same time as the independent variable
        z_past = X_past[i_lag:, z_indices, :]
        # z_past shape (history up to i_lag+1, X.shape[0]-2, X.shape[1]-n_past)

        # For the dependent variable, we return all times in the past but not the
        # current value
        j_past = X_past[1:, [j_ind], :]
        # j_past shape (history up to current time, 1, X.shape[1]-n_past)

        # reshape everything to be compatible shape
        i_past_reshaped = i_past.reshape(-1, i_past.shape[-1])
        j_past_reshaped = j_past.reshape(-1, j_past.shape[-1])
        z_past_reshaped = z_past.reshape(-1, z_past.shape[-1])
        # stack it back into a matrix and return
        return np.vstack([i_past_reshaped, j_past_reshaped, z_past_reshaped])


    def __residual(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:   #private
        """
        Computes the residuals of a variable x by regressing a
        conditioning set z out of it

        Args:
            x: Variable in question to regress out the conditioning set
            z: Conditioning set based on whether c-GC or fc-GC was used.

        return:
            the residual of x conditioned on z
        """

        if z.size == 0:
            return x - np.mean(x)
        design = np.column_stack([np.ones(x.size), z.T])
        coefs, *_ = np.linalg.lstsq(design, x, rcond=None)
        return x - design @ coefs


    def __ideal_lp(self, M):
        amp = np.ones(M)
        amp[int(self.f_c * M / self.f_s):-int(self.f_c * M / self.f_s)] = 0
        phase = np.zeros(M)
        H_f = amp * np.exp(1j * phase)
        h_n = np.fft.fftshift(np.real(np.fft.ifft(H_f)))
        return h_n


    def fit_rising(self, X: np.ndarray, idx: np.ndarray, verbose=1):
        """
        Fits c-gc to data

        Args:
            data: array-like of shape (n_neur, T)
                where `n_neur` is the number of neurons or variables
                and `T` is the time or number of samples

        Returns:
            self: object
                Returns the instance itself
        """
        data = np.asarray(X, dtype=float)
        if data.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if len(idx) != data.shape[0]:
            raise ValueError("idx must contain one selected-frame vector per variable")

        selected_frames = tuple(np.asarray(values, dtype=int) for values in idx)
        if any(np.any((values < 0) | (values >= data.shape[1])) for values in selected_frames):
            raise ValueError("selected-frame indices must lie on the trace time axis")

        self.n_neur = data.shape[0]
        n_rows = self.n_neur * (self.n_pasts + 1)
        corr = np.zeros((n_rows, self.n_neur))
        pVal_corr = np.ones((n_rows, self.n_neur))
        inv_corr = np.zeros_like(corr)
        pVal_inv_corr = np.ones_like(corr)
        iterator = tqdm(range(n_rows), disable=verbose < 1)
        for i in iterator:
            source = i % self.n_neur
            for j in range(self.n_neur):
                common = np.intersect1d(selected_frames[source], selected_frames[j])
                if common.size <= self.n_pasts + 1:
                    continue
                dat = self.shift_data(data[:, common])
                if dat.shape[1] < 2:
                    continue

                x = dat[i]
                y = dat[j]
                corr[i, j] = self._absolute_corr(x, y)
                pVal_corr[i, j] = self.__perm_test_(x, y)

                z = self.get_conditioning_set(i, j)
                x_res = self.__residual(x, z)
                y_res = self.__residual(y, z)
                inv_corr[i, j] = self._absolute_corr(x_res, y_res)
                pVal_inv_corr[i, j] = self.__perm_test_(x_res, y_res)

        self.corr_, self.pVal_corr_ = corr, pVal_corr
        self.inv_corr_, self.pVal_inv_corr_ = inv_corr, pVal_inv_corr

        return self









def cross_corr_(x, y, n_lags):
    return np.abs(np.corrcoef(x[:-n_lags], y[n_lags:])[1,0])



def cutt_mvg(arr,win_len):
    row_ = moving_avg(arr,window_size=5)
    x_ = np.roll(row_,np.random.randint(30,len(arr)-30,1)) # np.random.randint(30,len(arr)-30,1) Used 10 for proper acccessment
    j = np.diff(x_)> np.mean(np.diff(row_)>0) # filtering out small rises that are less than 0
    idx = np.where(j!=0)[0]
    return np.pad(row_,(0,win_len-1),'constant')[idx]



def cutt_lp(arr,f_c,f_s,M):
    h_n = ideal_lp(f_c,f_s,M)
    conv = np.convolve(arr,h_n,'same')
    j = np.diff(conv) > np.mean(np.diff(conv)>0.5)/10     # /5
    idx = np.where(j>0)[0]
    return arr[idx],idx      # conv[idx] is the squashed vector  # arr ==> conv



def idx_check_(row):
    row_list, new, temp = list(row),[],[]
    for i in range(len(row_list)-1):
        if row_list[i]+1 == row_list[i+1]:
        # grow this list
            temp.append(row_list[i])
        else:
        # add the element that is not the same as the next element, then create a new list
            temp.append(row_list[i])
            new.append(temp.copy())
            temp = []
    else:
        # when the for loop finishes, there is one element left over. This else clause will run when the for loop finishes
        temp.append(row_list[-1])
        new.append(temp)
    return new



def combine(list_of_indices):
    new_list_of_indices = []
    for i in range(len(list_of_indices)-1):
        idx = list_of_indices[i]
        new_list_of_indices.append(idx)
        if idx + 2 == list_of_indices[i+1]:
            new_list_of_indices.append(idx+1)
    new_list_of_indices.append(list_of_indices[-1])
    return new_list_of_indices



def diff_(inf_rising,inf):
    mat = inf_rising==inf
    mat = inf_rising.copy()
    mat[inf_rising==inf] = False
    return mat
#####################################################
#####################################################
