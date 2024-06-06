"""
    gaussian process optimizer constructs a gaussian process model
    for design space exploration.
    it updates the model progressively and picks the most valuable design to evaluate,
    hoping to reduce the total running time.
    a command to test "gp-optimizer.py":
    ``` 
        python gp-optimizer.py \
            -o [your experiment outputs directory] \
            -q [the number of your queries] \
            -s /path/to/gp-configs.json
    ```
    `optimizer`, `random_state`, etc. are provided with `gp-configs.json`, making you
    develop your optimizer conveniently when you tune your solution.
    you can specify more options to test your optimizer. please use
    ```
        python gp-optimizer.py -h
    ```
    to check.
    the code is only for example demonstration.
"""


import random
import torch
import sklearn
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier
from scipy.stats import norm




class GaussianProcessRegressorOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space, optimizer, random_state):
        """
            build a wrapper class for an optimizer.

            parameters
            ----------
            design_space: <class "MicroarchitectureDesignSpace">
        """
        AbstractOptimizer.__init__(self, design_space)
        kernel = DotProduct() + WhiteKernel()
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=optimizer,
            random_state=random_state
        )
        self.n_suggestions = 5
        self.x = []
        self.y = []
        self.init = True

    def expected_improvement(self, x_new, y_best, model, xi=0.01):
        """
        Expected Improvement (EI) acquisition function.

        Parameters
        ----------
        x_new: <numpy.ndarray>
            A new point to evaluate the EI at.
        y_best: float
            The best observed performance so far.
        model: <sklearn.gaussian_process.GaussianProcessRegressor>
            The trained Gaussian process model.
        xi: float
            Exploration-exploitation trade-off parameter.

        Returns
        -------
        ei: float
            The expected improvement value at `x_new`.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            mu, sigma = self.model.predict(x_new, return_std=True)
            mu, sigma = mu[0], sigma[0]

            # Ensure positive sigma
            sigma = np.maximum(sigma, 1e-6)

            # Calculate EI
            z = (mu - y_best - xi) / sigma
            ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

        return ei

    def get_best(self):
        # ppa weights
        weights = np.array([1, 1, 1])

        weighted_ppas_with_indices = [(sum(weights * ppa), ppa, idx) for idx, ppa in enumerate(self.y)]
        sorted_ppas_with_indices = sorted(weighted_ppas_with_indices, key=lambda x: x[0], reverse=True)
        best_weighted_ppa, best_ppa, best_index = sorted_ppas_with_indices[0]

        # print("Best PPA:", best_ppa)

        return sorted_ppas_with_indices

    def suggest(self):
        """
            get a suggestion from the optimizer.

            returns
            -------
            next_guess: <list> of <list>
                list of `self.n_suggestions` suggestion(s).
                each suggestion is a microarchitecture embedding.
        """
        
        if self.init:
            self.init = False

            # Normalization
            mu = self.design_space.size / 4
            sigma = self.design_space.size / 20
            x_guess = np.random.normal(mu, sigma, size=5).astype(int)

            # mu = self.design_space.size / 4 +self.design_space.size / 2
            # sigma = self.design_space.size / 20
            # x_guess = np.concatenate((x_guess,  np.random.normal(mu, sigma, size=5).astype(int)))
           
            mu = self.design_space.size / 2
            sigma = self.design_space.size / 20
            x_guess = np.concatenate((x_guess, np.random.normal(mu, sigma, size=10).astype(int)))

            # Random
            # x_guess = np.concatenate((x_guess, random.sample(
            #     range(1, self.design_space.size + 1), k=5
            # ).astype(int)))

            x_guess = np.clip(x_guess, 1, self.design_space.size)
            potential_suggest =  [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in x_guess
            ]
            return potential_suggest
        
        else:
            
            sorted_ppas_with_indices = self.get_best()
            
            # x_guess = random.sample(
            #     range(sorted_ppas_with_indices[0][-1], sorted_ppas_with_indices[1][-1]), k=self.n_suggestions
            # )
            # lower_bound = min(sorted_ppas_with_indices[1][-1] , sorted_ppas_with_indices[0][-1] )  
            # upper_bound = max(sorted_ppas_with_indices[1][-1] , sorted_ppas_with_indices[0][-1] )  
            # x_guess =  np.array(random.sample(range(lower_bound, upper_bound), k=1)).astype(int)

            # lower_bound = min(sorted_ppas_with_indices[2][-1] , sorted_ppas_with_indices[0][-1] )  
            # upper_bound = max(sorted_ppas_with_indices[2][-1] , sorted_ppas_with_indices[0][-1] )  
            # x_guess = np.concatenate((x_guess, np.array(random.sample(range(lower_bound, upper_bound), k=1)).astype(int)))

            # lower_bound = min(sorted_ppas_with_indices[3][-1] , sorted_ppas_with_indices[0][-1] )  
            # upper_bound = max(sorted_ppas_with_indices[3][-1] , sorted_ppas_with_indices[0][-1] )  
            # x_guess = np.concatenate((x_guess, np.array(random.sample(range(lower_bound, upper_bound), k=1)).astype(int)))

            # x_guess = np.concatenate((x_guess, np.array(random.sample(range(1, self.design_space.size+1), k=1)).astype(int)))

            lower_bound = min(sorted_ppas_with_indices[1][-1] , sorted_ppas_with_indices[0][-1] )  
            upper_bound = max(sorted_ppas_with_indices[1][-1] , sorted_ppas_with_indices[0][-1] )  
            x_guess =  np.random.sample(range(lower_bound, upper_bound), k=1).astype(int)

            lower_bound = min(sorted_ppas_with_indices[2][-1] , sorted_ppas_with_indices[0][-1] )  
            upper_bound = max(sorted_ppas_with_indices[2][-1] , sorted_ppas_with_indices[0][-1] )  
            x_guess = np.concatenate((x_guess, np.random.sample(range(lower_bound, upper_bound), k=1)).astype(int))

            lower_bound = min(sorted_ppas_with_indices[3][-1] , sorted_ppas_with_indices[0][-1] )  
            upper_bound = max(sorted_ppas_with_indices[3][-1] , sorted_ppas_with_indices[0][-1] )  
            x_guess = np.concatenate((x_guess, np.random.sample(range(lower_bound, upper_bound), k=1)).astype(int))

            x_guess = np.concatenate((x_guess, np.random.sample(range(1, self.design_space.size+1), k=1)).astype(int))

            potential_suggest =  [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in x_guess
            ]

        try:
            # NOTICE: we can also use the model to sweep the design space if
            # the design space is not quite large.
            # NOTICE: we only use a very naive way to pick up the design just for demonstration only.
            
            ppa = torch.Tensor(self.model.predict(np.array(potential_suggest)))
            # print(ppa)
            potential_parteo_frontier = get_pareto_frontier(ppa)
            _potential_suggest = []
            for point in potential_parteo_frontier:
                index = torch.all(ppa == point.unsqueeze(0), axis=1)
                _potential_suggest.append(
                    torch.Tensor(potential_suggest)[
                        torch.all(ppa == point.unsqueeze(0), axis=1)
                    ].tolist()[0]
                )
            return _potential_suggest
        except sklearn.exceptions.NotFittedError:
            return potential_suggest

        

    def observe(self, x, y):
        """
            send an observation of a suggestion back to the optimizer.

            parameters
            ----------
            x: <list> of <list>
                the output of `suggest`.
            y: <list> of <list>
                corresponding values where each `x` is mapped to.
        """
        for _x in x:
            self.x.append(_x)
        for _y in y:
            self.y.append(_y)
        # print("y:")
        # print(self.y)
        self.model.fit(np.array(self.x), np.array(self.y))


if __name__ == "__main__":
    experiment(GaussianProcessRegressorOptimizer)
