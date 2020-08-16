import heapq
import numbers
from itertools import product
from queue import deque

import numpy as np

############
# bft
############


def get_neighbor_pattern(dim, size=1, distance_limit=0.0, spacing=1.0, ord=1):
    neighbors = list(product(range(-size, size+1), repeat=dim))
    neighbors.remove((0,) * dim)
    if distance_limit > 0.0:
        if isinstance(spacing, numbers.Number):
            spacing = np.ones(dim) * spacing
        else:
            spacing = np.array(spacing).ravel()
        assert spacing.size == dim, "spacing, dimension size mismatch"
        neighbors = [n for n in neighbors if np.linalg.norm(
            spacing * n, ord=ord) <= distance_limit]
    return neighbors


class bftBase(object):
    def get_root(self):
        raise NotImplementedError("get_root has not been implemented")

    def get_successors(self, othernode):
        raise NotImplementedError("get_successors has not been implemented")

    def is_goal(self, subnode):
        return False


class bftTree(bftBase):

    def __init__(self, pts, size=1, distance_limit=0):
        pts = np.array(pts)
        dim = pts.shape[1]
        self.neighbors = get_neighbor_pattern(
            dim, size=size, distance_limit=distance_limit)
        self.dtype = pts.dtype
        self.casted_dtype = [('f{i}'.format(i=i), pts.dtype)
                             for i in range(dim)]
        self.pts = pts.view(self.casted_dtype)

    def get_root(self):
        return tuple(self.pts[0, 0])

    def get_successors(self, pos):
        successors = []
#         print("get_successors ", pos)
        for n in self.neighbors:
            _pos1 = np.array(pos)
            _pos = _pos1+n
            _pos = _pos.view(self.casted_dtype)
            if _pos in self.pts:
                dis = np.linalg.norm(n)
                successors.append((tuple(_pos[0]), dis))
        print("successors", len(successors))
        return successors


class Skeleton(bftBase):
    def __init__(self, skeleton=[], spacing=1.0, distance_limit=0):
        self.set_skeleton(skeleton)
        dim = self.skeleton.ndim
        self.neighbors = get_neighbor_pattern(dim, distance_limit)
        if isinstance(spacing, numbers.Number):
            spacing = np.ones(dim) * spacing
        else:
            spacing = np.array(spacing).ravel()

    def get_skeleton(self):
        return self._skeleton

    def set_skeleton(self, skeleton):
        _s = np.array(skeleton)
        _s[_s < 0] = 0
        _s[_s > 0] = 1
        self._skeleton = np.array(_s)
        if np.sum(_s) > 0:
            self.valid = True
        else:
            self.valid = False

    skeleton = property(get_skeleton, set_skeleton)

    def get_root(self, axis=0, reverse=True):
        if not self.valid:
            raise ValueError("the skeleton is not valid")
        dim = self._skeleton.ndim
        if axis >= dim:
            axis = 0
        slices = [slice(None)] * self._skeleton.ndim
        ranges = range(self._skeleton.shape[axis])
        if reverse:
            ranges = ranges[::-1]
        for index in ranges:
            slices[axis] = index
            plane = self._skeleton[slices]
            if np.sum(plane) > 0:
                _w = np.where(plane > 0)
                pos = [i[0] for i in _w]
                pos.insert(axis, index)
                return tuple(pos)

    def get_successors(self, pos):
        successors = []
        for n in self.neighbors:
            _pos = tuple(np.array(pos) + n)
            if self._is_valid_pos(_pos) and self._skeleton[_pos] > 0:
                dis = np.linalg.norm(self.spacing*n)
                successors.append((_pos, dis))
        return successors

    def _is_valid_pos(self, pos):
        _pos = np.array(pos)
        return np.all(0 <= _pos) and np.all(_pos < self._skeleton.shape)


def breadth_first_search(skeleton):
    # a FIFO open_set
    open_set = deque()
    # an empty set to maintain visited nodes
    closed_set = set()
    # a dictionary to maintain meta information (used for path formation)
    meta = dict()  # key -> (parent state, action to reach child)

    # initialize
    root = skeleton.get_root(reverse=False)
    meta[root] = (None, 0, 0)  # root, distance, branch_order
    open_set.append(root)
    bifurcation_list = []
    while len(open_set) > 0:
        subtree_root = open_set.popleft()
#        print(subtree_root)
        _, distance, order = meta[subtree_root]
        _temp_list = []
        for (child, d) in skeleton.get_successors(subtree_root):
            if child in closed_set:
                continue

            if child not in open_set:
                _temp_list.append((child, d))
                open_set.append(child)

#        p/len(_temp_list)
        if len(_temp_list) > 1:
            order += 1
            bifurcation_list.append((subtree_root, order))
        for child, d in _temp_list:
            #            print(order, child, subtree_root)
            meta[child] = (subtree_root, distance + d, order)

        closed_set.add(subtree_root)
    parent_list = [p for _, (p, _, _) in meta.items()]
    leaf_list = [p for p in meta if p not in parent_list]
    return meta, bifurcation_list, leaf_list


def dijkstra(matrix, start=None, end=None):
    """
    Implementation of Dijkstra algorithm to find the (s,t)-shortest path between top-left and bottom-right nodes
    on a nxn grid graph (with 8-neighbourhood).
    NOTE: This is an vertex variant of the problem, i.e. nodes carry weights, not edges.
    :param matrix (np.ndarray [grid_dim, grid_dim]): Matrix of node-costs.
    :return: matrix (np.ndarray [grid_dim, grid_dim]), indicator matrix of nodes on the shortest path.
    """
    if start is None:
        start = (0, 0)

    def neighbors_func(pos):
        pos = np.array(pos)
        neighbors = get_neighbor_pattern(dim=2)
        for off in neighbors:
            new_pos = pos+off
            if np.all(new_pos > 0) and np.all(new_pos < matrix.shape):
                yield new_pos

    costs = np.full_like(matrix, 1.0e10)
    costs[start] = matrix[start]

    priority_queue = [(matrix[0][0], start)]
    certain = set()
    transitions = dict()

    while priority_queue:
        _, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)

        certain.add((cur_x, cur_y))

    if end is None:
        return transitions
    # retrieve the path
    cur_x, cur_y = end
    on_path = np.zeros_like(matrix)
    on_path[-1][-1] = 1
    while (cur_x, cur_y) != start:
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0
    return on_path
