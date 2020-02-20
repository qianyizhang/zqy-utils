# -*- coding: utf-8 -*-

import numpy as np
import itertools


def flat_nested_list(nested_list):
    return list(itertools.chain(*nested_list))


def get_bounding_box(edge_list, dim=2):
    """
    given a (nested) list of points, return its bounding box
    args:
        edge_list (list[points])
        dim (int): point dimension, default is 2
        if the list is empty, all values are defaulted to -1
    return:
        bounding_box: (2d np.array with shape 2xdim): [top_left, bot_right]
    """
    edge_list_np = np.array(flat_nested_list(edge_list))
    try:
        edge_list_np = edge_list_np.reshape(-1, dim)
        min_point = edge_list_np.min(axis=0)
        max_point = edge_list_np.max(axis=0)
        bounding_box = np.vstack((min_point, max_point))
    except Exception:
        print(edge_list)
        dtype = edge_list_np.dtype
        print(dtype)
        bounding_box = np.ones((2, dim), dtype=dtype) * -1
    return bounding_box


__all__ = [k for k in globals().keys() if not k.startswith("_")]
