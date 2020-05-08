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


def psf(pts, kernel=0, size=None, as_tuple=True):
    """
    point spread function
    Args:
        pts (list[float]): N points with K dim
        kenerl (int) : 0 = center, 1=8 points, 2=27 points
        size (img.size): K int
        as_tuple (bool): if output as tuple
    Return:
        pts_list:
            if as_tuple=True: (array, ) x K
            if as_tuple=False: N x K array

    Note: the kernel points count is using 3-d input as reference.
    """
    if kernel == 1:
        pts = np.array(pts).astype(int)
    else:
        pts = np.array(pts).round().astype(int)

    if pts.size == 0:
        return pts

    if len(pts.shape) == 1:
        # dim -> 1 x dim
        pts = pts[None]

    if kernel > 0:
        dim = pts.shape[-1]
        if kernel == 1:
            neighbor_pts = np.stack(np.meshgrid(*[(0, 1)] * dim))
        elif kernel == 2:
            neighbor_pts = np.stack(np.meshgrid(*[(-1, 0, 1)] * dim))
        # N x dim x 1 + dim x 27 -> N x dim x 27
        pts = pts[..., None] + neighbor_pts.reshape(dim, -1)
        # N x dim x 27 -> N*27 x dim
        pts = pts.transpose(0, 2, 1).reshape(-1, dim)

        size = None if size is None else np.array(size) - 1
        pts = pts.clip(0, size)

    if as_tuple:
        pts = tuple(pts.T)
    return pts


def _test_psf():
    import boxx
    s = 10
    size = (s, s, s)
    pts_list = [(i, i, i) for i in range(s)]
    for kernel in range(3):
        img = np.zeros(size)
        pts = psf(pts_list, kernel, size=size, as_tuple=True)
        img[pts] = 1
        boxx.show(img)


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


def get_num_union(pts1, pts2):
    pts_all = np.concatenate([pts1, pts2])
    num_union = len(np.unique(pts_all, axis=0))
    return num_union


def get_pts_merge_mat(pts_list, pts_list2=None, ratio=0.25, criteria="min"):
    """
    get the merge_mat for points list
    Args:
        pts_list (list[np.ndarray]):
            each item in pts_list is an array of points
        pts_list2 (list[np.ndarray]):
            secondary list, if is not given, assuming it's self merge
    Return:
        merge_mat (N x N np.ndarray): binary mat
    """

    if pts_list2 is None:
        replicate = True
        pts_list2 = pts_list
    else:
        replicate = False
    M = len(pts_list)
    N = len(pts_list2)

    merge_mat = np.ones((M, N))
    for i in range(M):
        for j in range(N):
            if i >= j and replicate:
                merge_mat[i, j] = merge_mat[j, i]
                continue
            pts1, pts2 = pts_list[i], pts_list2[j]
            num1, num2 = len(pts1), len(pts2)
            # get shared pts
            num_union = get_num_union(pts1, pts2)
            divident = num1 + num2 - num_union
            if criteria == "min":
                divisor = min(num1, num2)
            elif criteria == "iou":
                divisor = num_union
            elif criteria == "self":
                divisor = num1
            elif criteria == "ref":
                divisor = num2
            else:
                raise NotImplementedError(f"unkown criteria {criteria}")
            merge_mat[i, j] = divident * 1.0 / divisor
    return merge_mat > ratio


def get_rounded_pts(pts_list, index_range=(0, None), stride=1.0,
                    as_unique=False):
    """
    given a list of points, cast them to int
    """
    start, end = index_range
    if end is None:
        end = len(pts_list) - 1
    assert end >= start, f"invalid index_range {index_range}"
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
