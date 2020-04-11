# -*- coding: utf-8 -*-

import itertools

import numpy as np


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


def psf(pts, kernel=0):
    """
    point spread function
    Args:
        pts (list[float]): K-dim point
        kenerl (int) : 0 = center, 1=8 points, 2=27 points
    Return:
        pts_list (tuple(1d array)): K array with N points

    Note: the N points count is using 3-d input as reference.
    """
    if kernel == 0:
        pts = np.array(pts).round().astype(int)
    elif kernel == 1:
        dim_list = [(int(pt), int(pt+1)) for pt in pts]
        pts = np.meshgrid(*tuple(dim_list))
    elif kernel == 2:
        neighbor_pts = np.meshgrid(*[(-1, 0, 1)]*3)
        pts = [(n+round(pt)).astype(int) for n, pt in zip(neighbor_pts, pts)]
    return tuple(pts)


def _test_psf():
    volume = np.zeros((10, 10, 10))
    pt = np.random.random(3)
    pt = pt + [3, 4, 5]

    volume[psf(pt, 0)] = 1
    assert volume.sum() == 1
    volume[psf(pt, 1)] = 2
    assert volume.sum() == 2 * 8
    volume[psf(pt, 2)] = 3
    assert volume.sum() == 3 * 27


def union_merge(merge_mat):
    """
    return group_indices based on merge_mat
    Args:
        merge_mat (NxN np.array or torch.Tensor): merging criteria
    Return:
        group_indices (list[indices])
    """
    N = len(merge_mat)
    if N == 0:
        return []
    else:
        item_id = np.arange(N)
        group_indices = [[i] for i in range(N)]
        for id1 in range(N):
            for id2 in range(N):
                if not merge_mat[id1, id2]:
                    continue
                min_id = min(item_id[id1], item_id[id2])
                for cur_id in (item_id[id1], item_id[id2]):
                    if cur_id == min_id:
                        continue
                    group_indices[min_id].extend(group_indices[cur_id])
                    group_indices[cur_id] = []
                    item_id[group_indices[min_id]] = min_id

    group_indices = [i for i in group_indices if i != []]
    return group_indices


def get_pts_merge_mat(pts_list, ratio=0.25, criteria="min"):
    """
    get the merge_mat for points list
    Args:
        pts_list (list[np.ndarray]):
            each item in pts_list is an array of points
    Return:
        merge_mat (N x N np.ndarray): binary mat
    """
    N = len(pts_list)
    merge_mat = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            if i >= j:
                merge_mat[i, j] = merge_mat[j, i]
                continue
            pts1, pts2 = pts_list[i], pts_list[j]
            num1, num2 = len(pts1), len(pts2)
            # get shared pts
            pts_all = np.concatenate([pts1, pts2])
            num_union = len(np.unique(pts_all, axis=0))
            divident = num1 + num2 - num_union
            if criteria == "min":
                divisor = min(num1, num2)
            elif criteria == "iou":
                divident = num_union
            else:
                raise NotImplementedError(f"unkown criteria {criteria}")
            merge_mat[i, j] = divident * 1.0 / divisor
    return merge_mat > ratio


def get_rounded_pts(pts_list, index_range, stride=1.0, as_unique=False):
    """
    given a list of points, cast them to int
    """
    start, end = index_range
    assert end >= start, f"invalud index_range {index_range}"
    pts = np.array(pts_list)
    pts = (pts[start:end + 1] / stride).round() * stride
    pts = pts.astype(int)
    if as_unique:
        return np.unique(pts, axis=0)
    else:
        return pts


def _test_union_merge():
    def has_match(pts):
        return (merge_fn(pts) - np.eye(len(pts))).max(0)
    pts = np.random.rand(10)

    def merge_fn(pts):
        return abs(pts[None] - pts[:, None]) < 0.1

    merge_mat = merge_fn(pts)
    indices_group = union_merge(merge_mat)
    valid = True
    for indices in indices_group:
        if len(indices) == 1:
            continue
        if not has_match(pts[indices]).all():
            print("pts within group is not connected to others")
            print(merge_mat, indices_group, indices)
            valid = False
            break

    for _ in range(10):
        indices = [np.random.choice(i) for i in indices_group]
        if has_match(pts[indices]).any():
            print("pts between groups is connected")
            print(merge_mat, indices_group, indices)
            valid = False
            break
    return valid


__all__ = [k for k in globals().keys() if not k.startswith("_")]
