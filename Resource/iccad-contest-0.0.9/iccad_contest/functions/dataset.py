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


import os
import numpy as np
import pandas as pd
from iccad_contest.utils.constants import contest_design_space, contest_dataset
from iccad_contest.utils.basic_utils import if_exist, info, warn, assert_error


def load_txt(path, fmt=int):
    if if_exist(path):
        info("loading from %s" % path)
        return np.loadtxt(path, dtype=fmt)
    else:
        warn("cannot load %s" % path)


def load_excel(path, sheet_name=0):
    """
        path: <str>
        sheet_name: <int> | <str>
    """
    if_exist(path, strict=True)
    data = pd.read_excel(path, sheet_name=sheet_name)
    info("read the sheet {} of excel from {}".format(sheet_name, path))
    return data


def load_contest_dataset():
	def validate_contest_dataset(dataset):
		assert dataset.ndim == 2, \
			assert_error(
				"{} dimensions are expected, but the platform gets {}.".format(dataset.ndim)
			)
		assert np.all(np.isfinite(dataset)), \
			assert_error(
				"dataset contains infinity."
			)
	dataset = load_txt(contest_dataset, fmt=float)
	validate_contest_dataset(dataset)
	return dataset


def load_contest_design_space():
	microarchitecture_design_space_sheet = load_excel(
		contest_design_space,
		sheet_name="Microarchitecture Design Space"
	)
	components_sheet = load_excel(
		contest_design_space,
		sheet_name="Components"
	)
	return microarchitecture_design_space_sheet, components_sheet
