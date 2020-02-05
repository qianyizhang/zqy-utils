# -*- coding: utf-8 -*-

import numpy as np

__all__ = ["get_bounding_box"]


def get_bounding_box(edge_list):
    """
    given a list of points, return its bounding box
    args:
        edge_list: {[[x,y],...]}
        if the list is empty, all values are defaulted to -1
    return:
        bounding_box: {2d np.array} (top_left, bot_right)
    """
    edge_list_np = np.array(edge_list)
    try:
        top_left = edge_list_np.min(axis=0)
        bot_right = edge_list_np.max(axis=0)
    except Exception:
        print(edge_list)
        print(edge_list_np.dtype)
        top_left = (-1, -1)
        bot_right = (-1, -1)
    return np.vstack((top_left, bot_right))
