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
import logging


def info(msg):
    """
        msg: <str>
    """
    print("[INFO]: {}".format(msg))


def test(msg):
    """
        msg: <str>
    """
    print("[TEST]: {}".format(msg))


def warn(msg):
    """
        msg: <str>
    """
    print("[WARN]: {}".format(msg))


def error(msg):
    """
        msg: <str>
    """
    print("[ERROR]: {}".format(msg))
    exit(1)


def assert_error(msg):
	"""
		msg: <str>
	"""
	return "[ERROR]: {}".format(msg)


def if_exist(path, strict=False):
    try:
        if os.path.exists(path):
            return True
        else:
            raise FileNotFoundError(path)
    except FileNotFoundError as e:
        warn(e)
        if not strict:
            return False
        else:
            exit(1)


def mkdir(path):
    if not if_exist(path):
        info("create directory: %s" % path)
        os.makedirs(path, exist_ok=True)


def remove(path):
    if if_exist(path):
        if os.path.isfile(path):
            os.remove(path)
            info("remove %s" % path)
        elif os.path.isdir(path):
            if not os.listdir(path):
                # empty directory
                os.rmdir(path)
            else:
                shutil.rmtree(path)
            info("remove %s" % path)


def create_logger(log_file):
    logging.basicConfig(filename=log_file, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger
