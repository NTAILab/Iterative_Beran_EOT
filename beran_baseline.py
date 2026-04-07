import numpy as np
import torch

from survival_function_model import survival_function_model

class beran_baseline(survival_function_model):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def fit(self, keys, deltas, times):
        super().fit(keys, deltas, times)

    def gauss_kernel(self, query, is_train=False):
        if type(query) == np.ndarray:
            query = torch.tensor(query, dtype=torch.float64, device="cpu")

        query = query.unsqueeze(1)
        keys = self.keys.unsqueeze(0)
        dist = ((query - keys) ** 2).sum(-1)
        kernel = torch.exp(-dist / self.tau) + 1e-31

        if is_train:
            bs = query.shape[0]
            kernel[range(bs), range(bs)] = 0.

        kernel = kernel / (kernel.sum(dim=1, keepdim=True))

        return kernel

    def survival_curve(self, input, is_train=False, eps = 1.e-7, a = 1.2345e-5):
        weights = self.gauss_kernel(input, is_train=is_train)

        device = weights.device
        dtype = weights.dtype
        eps = 1.e-7
        a = 1.2345e-5

        deltas = torch.as_tensor(self.deltas, dtype=dtype, device=device).view(-1)

        w_cumsum = torch.cumsum(weights, dim=1)
        shifted_w_cumsum = w_cumsum - weights
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
        filtered_xi = deltas[None,:].repeat(weights.shape[0], 1) * xi
        hazards = torch.cumsum(filtered_xi, dim=1)
        surv_func = torch.exp(-hazards)

        return self.times, surv_func