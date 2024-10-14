import os
from os.path import abspath, dirname

import task_aware_skill_composition

PROJ_DIR = dirname(dirname(abspath(task_aware_skill_composition.__file__)))
DATA_DIR = os.path.join(PROJ_DIR, "data")
