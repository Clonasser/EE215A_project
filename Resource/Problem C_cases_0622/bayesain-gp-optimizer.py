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

import sys, os
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
        self.x_idx = []
        self.x = []
        self.y = []
        self.init = True

    def get_best(self):
        # ppa weights
        weights = np.array([1, 1, 1])
        weighted_ppas_with_indices = [(sum(weights * ppa), ppa, idx) for idx, ppa in enumerate(self.y)]
        sorted_ppas_with_indices = sorted(weighted_ppas_with_indices, key=lambda x: x[0], reverse=True)
        best_weighted_ppa, best_ppa, best_index = sorted_ppas_with_indices[0]

        # print("Best PPA:", best_ppa)
        # print(self.x)
        # print(sorted_ppas_with_indices[0][-1])
        # print(self.x_idx[sorted_ppas_with_indices[0][-1]])
        # # print(self.design_space.vec_to_idx(sorted_ppas_with_indices[0][-1]))
        # exit(0)
        return sorted_ppas_with_indices
    def take2abs4(self, elem):
        # l1 norm
        return abs(elem[1])+abs(elem[2])+abs(elem[3])
        # l2 norm
        # return np.sqrt(elem[1]**2+elem[2]**2+elem[3]**2)
        # abs weight
        # return abs(elem[1]*elem[2]*elem[3])

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
            sigma = self.design_space.size / 10
            x_guess = np.concatenate((x_guess, np.random.normal(mu, sigma, size=10).astype(int)))

            # Random
            # x_guess = np.concatenate((x_guess, random.sample(
            #     range(1, self.design_space.size + 1), k=5
            # ).astype(int)))

            x_guess = np.clip(x_guess, 1, self.design_space.size-1)
            # print(x_guess.tolist())
            self.x_idx.extend(x_guess.tolist())
            potential_suggest =  [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in x_guess
            ]
            
            return potential_suggest
        
        else:
            # What happened?
            # if return none, it will samplpe a point randomly
            # return 

            
            # Random selection
            # x_guess = random.sample(
            #     range(sorted_ppas_with_indices[0][-1], sorted_ppas_with_indices[1][-1]), k=self.n_suggestions
            # )

            # Find bayesian optimization bound
            sorted_ppas_with_indices = self.get_best()
            lower_bound = abs(min(self.x_idx[sorted_ppas_with_indices[1][-1]], 
                              self.x_idx[sorted_ppas_with_indices[0][-1]]))
            upper_bound = abs(max(self.x_idx[sorted_ppas_with_indices[1][-1]], 
                              self.x_idx[sorted_ppas_with_indices[0][-1]]))

            #Find linear opportunity
            mirror_bound = np.clip(( 2 * self.x_idx[sorted_ppas_with_indices[0][-1]] - self.x_idx[sorted_ppas_with_indices[1][-1]] ),
                                   1, self.design_space.size)
            linear_lower_bound = abs(min(mirror_bound, 
                              self.x_idx[sorted_ppas_with_indices[0][-1]]))
            linear_upper_bound = abs(max(mirror_bound, 
                              self.x_idx[sorted_ppas_with_indices[0][-1]]))

            # sota
            # x_guess =  np.array(random.sample(range(lower_bound, upper_bound), k=5)).astype(int)
            # advoid too small bound
            if upper_bound - lower_bound > 2:
                sample_num  = 5 if upper_bound - lower_bound > 5 else upper_bound - lower_bound
                x_guess =  np.array(random.sample(range(lower_bound, upper_bound), k=sample_num)).astype(int)          
            # jump out of the optimization trap
            else:
                x_guess = np.array(random.sample(range(1, self.design_space.size+1), k=1)).astype(int)
                self.x_idx.extend(x_guess.tolist())
                potential_suggest =  [
                    self.design_space.vec_to_microarchitecture_embedding(
                        self.design_space.idx_to_vec(_x_guess)
                    ) for _x_guess in x_guess
                ]
                return potential_suggest

            # Find 1st derivative opportunity
            if linear_upper_bound -linear_lower_bound > 200:
                x_guess =  np.concatenate(x_guess, np.array(random.sample(range(linear_lower_bound, linear_upper_bound), k=1)).astype(int)) 
            # Add some random points
            if x_guess.size < 5:
                sample_num = 5 - x_guess.size
                x_guess = np.concatenate((x_guess, np.array(random.sample(range(1, self.design_space.size+1), k=sample_num)).astype(int)))

            # lower_bound = min(self.x_idx[sorted_ppas_with_indices[2][-1]], 
            #                   self.x_idx[sorted_ppas_with_indices[0][-1]])
            # upper_bound = max(self.x_idx[sorted_ppas_with_indices[2][-1]], 
            #                   self.x_idx[sorted_ppas_with_indices[0][-1]])
            # x_guess = np.concatenate((x_guess, np.array(random.sample(range(lower_bound, upper_bound), k=5)).astype(int)))

            # lower_bound = min(self.x_idx[sorted_ppas_with_indices[1][-1]], 
            #                   self.x_idx[sorted_ppas_with_indices[2][-1]])
            # upper_bound = max(self.x_idx[sorted_ppas_with_indices[1][-1]], 
            #                   self.x_idx[sorted_ppas_with_indices[2][-1]])
            # x_guess = np.concatenate((x_guess, np.array(random.sample(range(lower_bound, upper_bound), k=5)).astype(int)))

            # Add a random point
            # x_guess = np.concatenate((x_guess, np.array(random.sample(range(1, self.design_space.size+1), k=10)).astype(int)))
            
            # Clip
            x_guess = np.clip(x_guess, 1, self.design_space.size+1)
            self.x_idx.extend(x_guess.tolist())
            potential_suggest =  [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in x_guess
            ]
            # print(len(potential_suggest))
        try:
            # NOTICE: we can also use the model to sweep the design space if
            # the design space is not quite large.
            # NOTICE: we only use a very naive way to pick up the design just for demonstration only.
            ppa = torch.Tensor(self.model.predict(np.array(potential_suggest)))
            # print(ppa)
            potential_parteo_frontier = get_pareto_frontier(ppa)
            # print(potential_parteo_frontier)

            # if empty, randomly select a point
            if potential_parteo_frontier.numel() == 0:
                # Add a random point
                x_guess = np.concatenate((x_guess, np.array(random.sample(range(1, self.design_space.size), k=1)).astype(int)))
                # Clip
                x_guess = np.clip(x_guess, 1, self.design_space.size-1)
                potential_suggest =  [
                    self.design_space.vec_to_microarchitecture_embedding(
                        self.design_space.idx_to_vec(_x_guess)
                    ) for _x_guess in x_guess
                ]
                return potential_suggest

            mask_value = []
            for i in range(len(potential_parteo_frontier)):
                mask_value.append([i]+[ele for ele in potential_parteo_frontier[i]])
            mask_value.sort(key=self.take2abs4, reverse=True)

            # print(mask_value[1][0])
            
            # select one or two best point(s)
            mask = []
            for i in range(len(potential_parteo_frontier)):
                if len(mask_value) > 1:
                    if i == mask_value[0][0] or i == mask_value[1][0]:
                        mask.append([True, True, True])
                    else:
                        mask.append([False, False, False])
                else:
                    if i == mask_value[0][0]:
                        mask.append([True, True, True])
                    else:
                        mask.append([False, False, False])     
                         
                        
            mask = torch.tensor(mask)
            # print(mask) 
            # print(potential_parteo_frontier)
            potential_parteo_frontier = torch.masked_select(potential_parteo_frontier, mask)
            # num_selected = potential_parteo_frontier.size(0)
            original_width = 3
            potential_parteo_frontier = potential_parteo_frontier.view(-1, original_width)
            # print(potential_parteo_frontier)
            # exit(0)
             
            # print(potential_parteo_frontier)
            # if potential_parteo_frontier.numel() == 0:
            #     exit(0)

            _potential_suggest = []
            # for point in potential_parteo_frontier:
            for point in potential_parteo_frontier:
                index = torch.all(ppa == point.unsqueeze(0), axis=1)
                _potential_suggest.append(
                    torch.Tensor(potential_suggest)[
                        torch.all(ppa == point.unsqueeze(0), axis=1)
                    ].tolist()[0]
                )
            # print("i'm here")    
            # exit(0)
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
