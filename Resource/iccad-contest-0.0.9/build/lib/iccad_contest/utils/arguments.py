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
import argparse
import json
import uuid
from enum import IntEnum, auto
from pathvalidate.argparse import validate_filepath
from iccad_contest.utils.constants import solution_settings
from iccad_contest.utils.path_utils import open_with_abs_path
from iccad_contest.utils.basic_utils import error, assert_error


def type_of_file_path(val):
	validate_filepath(val, platform="auto")
	return val


def type_of_uuid(val):
	def generate_uuid(val):
		_uuid = uuid.UUID(hex=str.lower(val))
		assert _uuid.hex == val, \
			assert_error("error in generating UUID.")
		return _uuid
	return generate_uuid(str.lower(val))


def type_of_join(val):
	return val


def type_of_positive_int(val):
	val = int(val)
	if val <= 0:
		error(
			"a positive integer is expected, but the platform gets {}.".format(val)
		)
	return val


def base_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument(
		"-o",
		"--output-path",
		default=os.path.abspath(
			os.path.join(
				os.getcwd(),
				"outputs"
			)
		),
		type=type_of_file_path,
		help="contest output path specification"
	)
	return parser


def experiment_parser(description):
	parser = argparse.ArgumentParser(
		description=description,
		parents=[base_parser()]
	)
	parser.add_argument(
		"-u",
		"--uuid",
		type=type_of_uuid,
		default=uuid.uuid4().hex,
		help="universally unique identifier (UUID) specification"
	)
	parser.add_argument(
		"-s",
		"--solution-settings",
		type=type_of_join,
		help="solution submission specification"
	)
	parser.add_argument(
		"-q",
		"--num-of-queries",
		default=5,
		type=type_of_positive_int,
		help="the number of queries specification"
	)
	return parser


def args_to_dict(args):
	return dict(vars(args))


def parse_args(parser):
	"""
		arguments parser.
	"""
	return args_to_dict(parser.parse_args())


def load_solution_settings(solution_settings):
	try:
		with open_with_abs_path(
			os.path.join(
				os.path.abspath(solution_settings)
			),
			'r'
		) as f:
			settings = json.load(f)
	except Exception as e:
		settings = {}
	return settings
