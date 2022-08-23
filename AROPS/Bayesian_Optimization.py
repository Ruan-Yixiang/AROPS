import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, ProbabilityOfImprovement
from botorch.optim import optimize_acqf_mixed, optimize_acqf
from typing import Optional, Union
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal
from Kernel import MixedSingleTaskGP
from sklearn.metrics import r2_score


# generate the candidate conditions by Bayesian Optimization
def get_next_exeps(space, train_x, train_y, q, return_gp=False, only_return_gp=False):
    r"""
    :param space: the design space
    :param train_x: the reaction conditions' feature vector, Tensor
    :param train_y: the objective vector, Tensor
    :param q: Number of output candidate conditions, Int
    :param return_gp: whether return the GP model, Bool
    :param only_return_gp: whether return the GP model, Bool
    :return:
    candidate_sort: the candidate conditions, Tensor
    qpi: list of candidate conditions' PI value
    gp: the GP model
    """
    x_dim = train_x.size()[1]
    if not space.is_real:
        cat_index = []
        n_real = 0
        for j, i in enumerate(space.dimensions):
            para_type = str(type(i)).split('.')[-1][0:-2]
            if para_type == 'Categorical':
                cat_index.append([j, len(i.bounds)])
            elif para_type == 'Real':
                n_real += 1
        n_cat = n_real
        for i in cat_index:
            n_cat += i[1]
        cat_mat = np.eye(cat_index[0][1])
        if len(cat_index) > 1:
            for i in cat_index[1:]:
                cat_mat_temp = np.ones((cat_mat.shape[0] * i[1], cat_mat.shape[1] + i[1]))
                for k, j in enumerate(cat_mat):
                    cat_mat_temp[k * i[1]: (k + 1) * i[1], :] = np.hstack(
                        (np.repeat(np.array([j]), repeats=i[1], axis=0), np.eye(i[1])))
                cat_mat = cat_mat_temp
        fixed_features_list = []
        for m in cat_mat:
            cat_dict = {}
            for o, n in enumerate(range(n_real, n_cat)):
                cat_dict[n] = m[o]
            fixed_features_list.append(cat_dict)
    if space.is_categorical:
        length = 3.0
        sign = -1
        i = 0
        while 1:
            if 0 < length < 6:
                length += i * sign * 0.5
                sign = sign * -1
            else:
                length = (i + 1) * 0.5
            gp = MixedSingleTaskGP(train_x, train_y, cat_dims=list(range(n_real, n_cat)), prior_l=length)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)
            mean = gp(train_x).mean
            train_y_n = train_y.detach().numpy().flatten()
            mean = mean.detach().numpy()
            loss = r2_score(train_y_n, mean)
            i += 1
            if loss > 0.999 or length >= 15:
                break
    elif space.is_real:
        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
    else:
        gp = MixedSingleTaskGP(train_x, train_y, cat_dims=list(range(n_real, n_cat)))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
    sampler = SobolQMCNormalSampler(2048)
    bounds = torch.stack([torch.zeros(x_dim), torch.ones(x_dim)])
    PI = ProbabilityOfImprovementXi(gp, best_f=torch.max(train_y), maximize=True)
    EI = ExpectedImprovement(gp, best_f=torch.max(train_y), maximize=True)
    qEI = qExpectedImprovement(gp, best_f=torch.max(train_y), sampler=sampler)
    if only_return_gp:
        return gp
        pass
    if not space.is_real:
        candidate, acq_value = optimize_acqf_mixed(
            qEI, bounds=bounds, q=q, fixed_features_list=fixed_features_list, num_restarts=20, raw_samples=200)
    else:
        candidate, acq_value = optimize_acqf(qEI, bounds=bounds, q=q, num_restarts=20, raw_samples=200)
    qei = []
    qpi = []
    for i in candidate:
        qei.append(EI(i.resize(1, x_dim)))
    qei = torch.tensor(qei)
    candidate_sort = torch.ones((q, train_x.size()[1]))
    qei_sort = torch.argsort(-qei)
    for i, j in enumerate(qei_sort):
        candidate_sort[i, :] = candidate[j, :]
    for j, i in enumerate(candidate_sort):
        qpi.append(PI(i.resize(1, x_dim)))
    qpi = torch.tensor(qpi)
    if return_gp:
        return candidate_sort, qpi, gp
    else:
        return candidate_sort, qpi


# Rewrite the ProbabilityOfImprovement class in botorch to add the Xi parameter
class ProbabilityOfImprovementXi(ProbabilityOfImprovement):

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:

        super().__init__(model=model, best_f=best_f, objective=objective, maximize=maximize)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor, xi: float = 0.01) -> Tensor:
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        mean, sigma = posterior.mean, posterior.variance.sqrt()
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = mean - self.best_f.expand_as(mean)
        if not self.maximize:
            u = -u
        u = (u - xi) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        return normal.cdf(u)
