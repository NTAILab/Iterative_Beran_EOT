import numpy as np
import torch

from survival_function_model import survival_function_model

class beran_iterative_k_ti(survival_function_model):
    def __init__(self, tau=0.1, k = 5, reg_pi=1.0):
        super().__init__()
        self.tau = tau
        self.pi = None
        self.k = k
        self.reg_pi = reg_pi

    def fit(self, keys, deltas, times):
        super().fit(keys, deltas, times)

    def train(self, test_input, test_deltas, test_times, epochs):
        train_history = {"c_index": ([], []), "IBS": ([], [])}
        self.pi = np.ones((self.times.shape[0]), dtype=np.float64) / self.times.shape[0]

        for epoch in range(epochs):
            times, SF_test = self.survival_curve(test_input)
            c_index, ibs = self.calculate_metics(test_deltas, test_times, SF_test)
            train_history["c_index"][1].append(c_index)
            train_history["IBS"][1].append(ibs)

            times, SF = self.survival_curve(self.keys, is_train=True)
            c_index, ibs = self.calculate_metics(self.deltas, times, SF)
            train_history["c_index"][0].append(c_index)
            train_history["IBS"][0].append(ibs)

            times, SF = self.survival_curve(self.keys, is_train=False)

            expansion = np.ones(self.keys.shape[0])
            expansion = expansion[:, np.newaxis]
            SF_exp = np.concatenate((expansion, SF), axis=1)
            SF_diff = SF_exp[:,:-1] - SF_exp[:,1:]
            SF_diff = np.clip(SF_diff, 1e-5, SF_diff.max())
            SF_diff_res = np.zeros(self.keys.shape[0])

            for i in range(SF_exp.shape[0]):
                for j in range(SF_exp.shape[0] - 1):
                    SF_diff_res[i] += np.log(SF_diff[i,j])
                SF_diff_res[i] += np.log(self.pi[i])

            self.pi = torch.softmax(torch.tensor(SF_diff_res/1.0), dim=0)
            self.pi = self.pi.clamp(1e-25)
            self.pi = self.pi.detach().cpu().numpy()
            self.pi = self.pi/self.pi.sum()

        times, SF_test = self.survival_curve(test_input)
        c_index, ibs = self.calculate_metics(test_deltas, test_times, SF_test)
        train_history["c_index"][1].append(c_index)
        train_history["IBS"][1].append(ibs)

        times, SF = self.survival_curve(self.keys, is_train=True)
        c_index, ibs = self.calculate_metics(self.deltas, times, SF)
        train_history["c_index"][0].append(c_index)
        train_history["IBS"][0].append(ibs)

        return train_history

    def gauss_kernel(self, query, is_train=False):
        if type(query) == np.ndarray:
            query = torch.tensor(query, dtype=torch.float64, device="cpu")

        query = query.unsqueeze(1)
        keys = self.keys.unsqueeze(0)
        dist = ((query - keys) ** 2).sum(-1)
        kernel = torch.exp(-dist / self.tau) + 1e-31
        kernel = kernel * np.power(self.pi[None,:], self.reg_pi)

        if is_train:
            bs = query.shape[0]
            kernel[range(bs), range(bs)] = 0.

        kernel = kernel / kernel.sum(dim=1, keepdim=True)

        return kernel

    def survival_curve(self, input, is_train=False, eps = 1.e-7, a = 1.2345e-5):
        weights_full = self.gauss_kernel(input, is_train=is_train)
        weights_keys = self.gauss_kernel(self.keys, is_train=True)

        device = weights_full.device
        dtype = weights_full.dtype

        output = torch.ones([input.shape[0], self.times.shape[0]], dtype=dtype, device=device)
        deltas = torch.as_tensor(self.deltas, dtype=dtype, device=device).view(-1)

        for i in range(input.shape[0]):
            if self.k > 0:
                _, indices_k = torch.topk(weights_full[i, :], k=self.k, dim=0, sorted=False)
                weights_k_i = torch.concatenate([weights_full[i, :].unsqueeze(dim=0), weights_keys[indices_k, :]],
                                                dim=0)
            else:
                weights_k_i = weights_full[i, :].unsqueeze(dim=0)

            w_cumsum = torch.cumsum(weights_k_i, dim=1)
            shifted_w_cumsum = w_cumsum - weights_k_i
            number_close_to_one = 1.0 - eps
            shifted_w_cumsum = torch.where(
                shifted_w_cumsum >= number_close_to_one,
                number_close_to_one + a * (shifted_w_cumsum - shifted_w_cumsum.detach()),
                shifted_w_cumsum
            )
            w_cumsum = torch.where(
                w_cumsum >= number_close_to_one,
                number_close_to_one + a * (w_cumsum - w_cumsum.detach()),
                w_cumsum
            )

            xi = torch.log(1.0 - shifted_w_cumsum)
            xi -= torch.log(1.0 - w_cumsum)
            filtered_xi = deltas[None, :].repeat(weights_k_i.shape[0], 1) * xi
            hazards = torch.cumsum(filtered_xi, dim=1)
            surv_func = torch.exp(-hazards)
            output[i, :] = surv_func.mean(dim=0)

        return self.times, output