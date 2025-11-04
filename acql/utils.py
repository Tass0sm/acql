import os
from os.path import abspath, dirname

import achql

PROJ_DIR = dirname(dirname(abspath(achql.__file__)))
DATA_DIR = os.path.join(PROJ_DIR, "data")
