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


# A serialization abstraction layer (SAL) to save and load experimental results.\
# All IO experimental results should go through this module.
# This makes changing the backend (between different databases) transparent to the benchmark codes.


import json
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from tempfile import mkdtemp
import xarray as xr
from iccad_contest.utils.constants import format_of_exp_output_root, list_of_exp_output_sub_root, \
	suffix_of_log, log_root, summary_root, suffix_of_summary
from iccad_contest.utils.path_utils import validate_file_path, validate_file_name, \
	path_join_for_write_with_create


class Serializer(ABC):
	@staticmethod
	@abstractmethod
	def init_contest_output_data(output_path):
		raise NotImplementedError

	@staticmethod
	@abstractmethod
	def logging_path(output_path, uuid):
		raise NotImplementedError

	@staticmethod
	@abstractmethod
	def save(data):
		raise NotImplementedError

	@staticmethod
	@abstractmethod
	def load(path):
		raise NotImplementedError


class ContestSerializer(Serializer):

	def init_contest_output_data(output_path):
		validate_file_path(output_path)

		prefix_of_exp_output_root = datetime.utcnow().strftime(format_of_exp_output_root)
		exp_output_root = mkdtemp(prefix=prefix_of_exp_output_root, dir=output_path)
		for exp_output_sub_root in list_of_exp_output_sub_root:
			mkdir(exp_output_sub_root)
		return os.path.basename(exp_output_root)

	def logging_path(output_path, uuid):
		validate_file_path(output_path)
		return path_join_for_write_with_create(output_path, log_root, uuid.hex + suffix_of_log)

	def load(path):
		validate_file_path(path)
		with open(path, 'r') as f:
			data = json.load(f)
		return data

	def save(output_path, uuid, data):
		validate_file_path(output_path)
		with open(
			path_join_for_write_with_create(output_path, summary_root, uuid.hex + suffix_of_summary),
			'w',
			encoding='utf-8'
		) as f:
			json.dump(data, f, indent=4)
		return path_join_for_write_with_create(output_path, summary_root, uuid.hex + suffix_of_summary)
