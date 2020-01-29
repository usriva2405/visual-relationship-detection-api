# -*- coding: utf-8 -*-
"""

"""

__version__ = '0.1'
__author__ = 'Utkarsh Srivastava'


import glob
import math
import os
import random
import shutil
from pathlib import Path


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

