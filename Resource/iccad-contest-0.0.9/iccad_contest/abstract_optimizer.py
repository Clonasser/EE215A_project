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


"""
    abstract base class for the optimizers.
    it creates a common API across all packages.
"""


from abc import ABC, abstractmethod
from importlib_metadata import version


class AbstractOptimizer(ABC):
    """
        abstract base class for the optimizers.
        it creates a common API across all packages.
    """

    # every implementation package needs to specify `primary_import`,
    # e.g.,
    # primary_import = opentuner
    primary_import = None

    def __init__(self, design_space, **kwargs):
        """
            build a wrapper class for an optimizer.

            parameters
            ----------
            design_space: <class "MicroarchitectureDesignSpace">
        """
        self.design_space = design_space
        # early stopping criterion
        self.early_stopping = False

    @classmethod
    def get_version(cls):
        """
            get the version for the optimizer.

            returns
            -------
            version_str: <str>
                the version number of an optimizer.
                usually, it is equivalent to `package.__version__`.
        """
        assert (cls.primary_import is None) or isinstance(cls.primary_import, str)
        # verstion "x.x.x" is used as a default version
        version_str = "x.x.x" \
            if cls.primary_import is None else version(cls.primary_import)
        return version_str

    @abstractmethod
    def suggest(self):
        """
            get a suggestion from the optimizer.

            returns
            -------
            next_guess: <list> of <list>
                list of suggestions.
                each suggestion is a microarchitecture embedding.
        """
        pass

    @abstractmethod
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
        pass
