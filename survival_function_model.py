import numpy as np
import torch
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score

class survival_function_model:
    def __init__(self):
        self.keys = None
        self.times = None
        self.deltas = None
        pass

    def fit(self, keys, deltas, times):
        sort_idx = np.argsort(times)
        self.keys = torch.tensor(keys[sort_idx], dtype=torch.float32, device="cpu")
        self.times = times[sort_idx]
        self.deltas = deltas[sort_idx]

    def calculate_metics(self, deltas, times, SF):
        with torch.no_grad():
            SF = SF.detach().clone().numpy()
            if SF.ndim == 1:
                SF = SF[np.newaxis, :]
            if deltas.ndim == 1:
                deltas = deltas[np.newaxis, :]
            if times.ndim == 1:
                times = times[np.newaxis, :]
            c_index = self._count_c_index(SF, deltas, times)
            ibs = self._count_ibs(SF, deltas, times)

            return c_index, ibs

    def survival_curve(self, input, is_train=False, eps = 1.e-7, a = 1.2345e-5):
        return [0], [0]

    def integrate_SF(self, SF):
        if not isinstance(self.times, torch.Tensor):
            times_tensor = torch.tensor(self.times, dtype=torch.float32, device=SF.device)
        else:
            times_tensor = self.times.to(SF.device).float()

        zero_tensor = torch.tensor([0.0], device=SF.device)
        times_exp = torch.cat([zero_tensor, times_tensor], dim=0)
        dt = times_exp[1:] - times_exp[:-1]
        weighted_survival = SF * dt.unsqueeze(0)
        integrated_survival = torch.sum(weighted_survival, dim=1)

        return integrated_survival

    def _count_ibs(self, SF, deltas, times):
        if times.ndim > 1:
            times = times.ravel()
        if deltas.ndim > 1:
            deltas = deltas.ravel()

        y_test = np.array(
            [(deltas[i].astype(bool), times[i])
             for i in range(times.shape[0])],
            dtype=[('event', 'bool'), ('time', 'float')]
        )

        min_time = times.min()
        max_time = times.max()
        train_max = self.times.max()

        effective_max = min(max_time, train_max)

        mask = (
                (self.times >= min_time) &
                (self.times < effective_max)
        )

        if not np.any(mask):
            print(f"Warning: no valid evaluation times in range [{min_time}, {effective_max})")
            return np.nan

        eval_times = np.unique(self.times[mask])

        time_to_col = {}
        for i, t in enumerate(self.times):
            if t not in time_to_col:
                time_to_col[t] = i

        selected_cols = [time_to_col[t] for t in eval_times]
        SF_fix = SF[:, selected_cols]

        y_train = np.array(
            [(self.deltas[i].astype(bool), self.times[i])
             for i in range(len(self.times))],
            dtype=[('event', 'bool'), ('time', 'float')]
        )

        ibs = integrated_brier_score(y_train, y_test, SF_fix, eval_times)
        return ibs

    def _count_c_index(self, SF, deltas, times):
        if times.ndim > 1:
            times = times.ravel()
        if deltas.ndim > 1:
            deltas = deltas.ravel()

        y_test = np.array(
            [(deltas[i].astype(bool), times[i])
             for i in range(times.shape[0])],
            dtype=[('event', 'bool'), ('time', 'float')]
        )

        cens, time_field = y_test.dtype.names

        def calc_expected_time(surv, times):
            dt = times[1:] - times[:-1]
            return times[0] + (surv[:, :-1] * dt).sum(1)

        expected_time = calc_expected_time(SF, self.times)

        c_index = concordance_index_censored(
            y_test[cens],
            y_test[time_field],
            -expected_time
        )[0]

        return c_index
