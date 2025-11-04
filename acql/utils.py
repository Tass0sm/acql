import os
from os.path import abspath, dirname

import acql

PROJ_DIR = dirname(dirname(abspath(acql.__file__)))
DATA_DIR = os.path.join(PROJ_DIR, "data")
