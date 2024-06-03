# Author: baichen.bai@alibaba-inc.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np
from itertools import combinations
from abc import ABC, abstractmethod
from scipy import stats
from iccad_contest.functions.design_space import parse_contest_design_space
from iccad_contest.functions.dataset import load_contest_dataset
from iccad_contest.utils.constants import dim_of_objective_values
from iccad_contest.utils.basic_utils import assert_error


class Problem(ABC):
    def __init__(self):
        super(Problem, self).__init__()
        self._design_space = None

    @property
    def design_space(self):
        return self._design_space

    @design_space.setter
    def design_space(self, design_space):
        self._design_space = design_space

    @design_space.deleter
    def design_space(self):
        del self._design_space

    @abstractmethod
    def evaluate(self, points):
        raise NotImplementedError()


class PPAT(object):
    def __init__(self, performance, power, area, time_of_vlsi_flow):
        super(PPAT, self).__init__()
        self._performance = performance
        self._power = power
        self._area = area
        self._time_of_vlsi_flow = time_of_vlsi_flow

    @property
    def performance(self):
        return self._performance

    @property
    def power(self):
        return self._power

    @property
    def area(self):
        return self._area

    @property
    def time_of_vlsi_flow(self):
        return self._time_of_vlsi_flow

    def get_objective_values(self):
        return [
            self.performance, self.power, self.area
        ]


class DesignSpaceExplorationProblem(Problem):
    def __init__(self):
        super(DesignSpaceExplorationProblem, self).__init__()
        self.design_space = parse_contest_design_space()
        self.dataset = load_contest_dataset()
        self.dim_of_objective_values = dim_of_objective_values
        self.microarchitecture_embedding_set = self.dataset[
            :, :-(dim_of_objective_values + 1)
        ]
        self.ppa = stats.zscore(self.dataset[:, -4:-1])
        self.performance = self.ppa[:, -3]
        self.power = self.ppa[:, -2]
        self.area = self.ppa[:, -1]
        self.time_of_vlsi_flow = self.dataset[:, -1]
        self.pareto_frontier = self.get_pareto_frontier()
        self.reference_point = self.calc_reference_point()

    def evaluate(self, microarchitecture_embedding):
        idx = [
            all(x) \
                for x in np.equal(
                    self.microarchitecture_embedding_set,
                    microarchitecture_embedding
                )
        ].index(True)
        return PPAT(
            self.performance[idx],
            self.power[idx],
            self.area[idx],
            self.time_of_vlsi_flow[idx]
        )

    def get_pareto_frontier(self):
        return get_pareto_frontier(torch.Tensor(self.ppa))

    def calc_reference_point(self):
        # NOTICE: each vale of the reference point should be the worst.
        return [
            # retrieve the minimal PPA value w.r.t. the Pareto frontier
            # however, in the calculation of hypervolume, we assume minimization,
            # so we negate each value
            -np.min(self.performance),
            -np.min(self.power),
            -np.min(self.area)
        ]


def _get_non_dominated(Y, maximize=True):
    is_efficient = torch.ones(
        *Y.shape[:-1],
        dtype=bool,
        device=Y.device
    )
    for i in range(Y.shape[-2]):
        i_is_efficient = is_efficient[..., i]
        if i_is_efficient.any():
            vals = Y[..., i : i + 1, :]
            if maximize:
                update = (Y > vals).any(dim=-1)
            else:
                update = (Y < vals).any(dim=-1)
            update[..., i] = i_is_efficient.clone()
            is_efficient2 = is_efficient.clone()
            if Y.ndim > 2:
                is_efficient2[~i_is_efficient] = False
            is_efficient[is_efficient2] = update[is_efficient2]
    return is_efficient


def get_non_dominated(Y, deduplicate=True):
    MAX_BYTES = 5e6
    n = Y.shape[-2]
    if n == 0:
        return torch.zeros(
            Y.shape[:-1],
            dtype=torch.bool,
            device=Y.device
        )
    el_size = 64 if Y.dtype == torch.double else 32
    if n > 1000 or \
        n ** 2 * Y.shape[:-2].numel() * el_size / 8 > MAX_BYTES:
        return _get_non_dominated(Y)

    Y1 = Y.unsqueeze(-3)
    Y2 = Y.unsqueeze(-2)
    dominates = (Y1 >= Y2).all(dim=-1) & (Y1 > Y2).any(dim=-1)
    nd_mask = ~(dominates.any(dim=-1))
    if deduplicate:
        indices = (Y1 == Y2).all(dim=-1).long().argmax(dim=-1)
        keep = torch.zeros_like(nd_mask)
        keep.scatter_(dim=-1, index=indices, value=1.0)
        return nd_mask & keep
    return nd_mask


def get_pareto_frontier(objective_values):
    """
        objective_values: <torch.Tensor>
        NOTICE: `get_pareto_frontier` assumes maximization.
    """
    assert isinstance(objective_values, torch.Tensor), \
        assert_error("please convert the input to 'torch.Tensor.")
    # NOTICE: we negate the normalized power and area values
    # to calculate the Pareto frontier since a better design
    # has higher performance and lower power and area values.
    for i in [1, 2]:
        objective_values[:, i] = -objective_values[:, i]
    return objective_values[get_non_dominated(objective_values)]


def get_adrs(reference_pareto_frontier, predict_pareto_frontier):
    """
        reference_pareto_frontier: <torch.Tensor>
        predict_pareto_frontier: <torch.Tensor>
    """
    # calculate average distance to the `reference_pareto_frontier` set
    assert isinstance(reference_pareto_frontier, torch.Tensor) and \
        isinstance(reference_pareto_frontier, torch.Tensor), \
            assert_error("please convert the input to 'torch.Tensor.")
    adrs = 0
    reference_pareto_frontier = reference_pareto_frontier.cpu()
    predict_pareto_frontier = predict_pareto_frontier.cpu()
    for omega in reference_pareto_frontier:
        mini = float('inf')
        for gamma in predict_pareto_frontier:
            mini = min(mini, np.linalg.norm(omega - gamma))
        adrs += mini
    adrs = adrs / len(reference_pareto_frontier)
    return adrs
