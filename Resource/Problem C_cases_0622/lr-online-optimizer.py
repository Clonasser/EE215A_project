"""
    linear regression (online) optimizer constructs a linear regression model
    for design space exploration.
    it updates the model progressively and picks the most valuable design to evaluate,
    hoping to reduce the total running time.
    a command to test "lr-offline-optimizer.py":
    ``` 
        python lr-online-optimizer.py \
            -o [your experiment outputs directory] \
            -q [the number of your queries]
    ```
    you can specify more options to test your optimizer. please use
    ```
        python lr-online-optimizer.py -h
    ```
    to check.
    the code is only for example demonstration.
"""


import random
import torch
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier


class OnlineLinearRegressionOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space):
        """
            build a wrapper class for an optimizer.

            parameters
            ----------
            design_space: <class "MicroarchitectureDesignSpace">
        """
        AbstractOptimizer.__init__(self, design_space)
        self.model = LinearRegression()
        self.n_suggestions = 5
        self.x = []
        self.y = []

    def suggest(self):
        """
            get a suggestion from the optimizer.

            returns
            -------
            next_guess: <list> of <list>
                list of `self.n_suggestions` suggestion(s).
                each suggestion is a microarchitecture embedding.
        """
        x_guess = random.sample(
            range(1, self.design_space.size + 1),
            k=self.n_suggestions
        )
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
        self.model.fit(np.array(self.x), np.array(self.y))


if __name__ == "__main__":
    experiment(OnlineLinearRegressionOptimizer)
