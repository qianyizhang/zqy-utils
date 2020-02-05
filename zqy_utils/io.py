# -*- coding: utf-8 -*-

import time
import os
import os.path as osp

__all__ = ["make_dir"]


def make_dir(*args):
    """
    the one-liner directory creator
    """
    path = osp.join(*[arg.strip(" ") for arg in args])
    if not osp.isdir(path):
        from random import random
        time.sleep(random() * 0.001)
        if not osp.isdir(path):
            os.makedirs(path)
    return path
