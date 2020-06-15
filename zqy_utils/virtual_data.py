# -*- coding: utf-8 -*-
import os.path as osp
from collections import Iterable

import numpy as np


META_TOKEN = "__virtual_meta"


def valid_tile_size(shape, tile_size):
    if isinstance(tile_size, Iterable):
        assert len(tile_size) == len(shape), \
            "tile_size and shape must have same dimension"
        tile_size = np.array(tile_size).astype(np.int)
        assert np.all(tile_size > 0), \
            f"tile_size must be non-negative {tile_size}"
    elif isinstance(tile_size, int):
        assert tile_size > 0, f"tile_size must be non-negative {tile_size}"
        tile_size = np.array([tile_size] * len(shape), dtype=np.int)
    else:
        raise ValueError(f"invalid tile_size {tile_size}")
    return tile_size


def get_indexing_mat(mat):
    return np.arange(mat.size, dtype=np.int).reshape(mat.shape)


def norm_index(nd_index, tile_size):
    """
    """
    normed_index = [i for i in np.index_exp[nd_index] if i is not None]
    tiled_index = []
    current_index = 0

    def norm(_index, is_stop=False):
        if _index is None:
            return _index
        i = _index // tile
        if is_stop:
            # fast way to increase end by 1 to cover the range
            i += 1
            if i == 0:
                i = None
        return i

    for ind in normed_index:
        if np.array_equal(ind, []):
            continue
        if ind is Ellipsis:
            current_index += len(tile_size) - len(normed_index) + 1
            tiled_index.append(...)
            continue
        tile = tile_size[current_index]
        if isinstance(ind, int):
            tiled_index.append(norm(ind))
        elif isinstance(ind, np.ndarray) and np.issubdtype(ind.dtype, np.signedinteger):
            tiled_index.append(ind//tile)
        elif isinstance(ind, slice):
            if ind.step is None or ind.step > 0:
                start = norm(ind.start)
                stop = norm(ind.stop, is_stop=True)
            else:
                start = norm(ind.stop)
                stop = norm(ind.start, is_stop=True)
            tiled_index.append(slice(start, stop))
        else:
            raise NotImplementedError(
                f"unknown index {ind}, {normed_index}, {nd_index}")
        current_index += 1
    return tuple(tiled_index)


def index_to_slice(nd_index, tile_size, shape=None):
    if shape is None:
        shape = tile_size
    slices = [slice(i * t, i * t + s)
              for i, t, s in zip(nd_index, tile_size, shape)]
    return tuple(slices)


class VirtualData():
    def __init__(self, shape, tile_size, data_handler, name):
        self.shape = shape
        self.tile_size = valid_tile_size(shape, tile_size)
        self.data_handler = data_handler
        assert not name.startswith("__"), "'__' prefix is prohibited"
        self.name = name
        self.tile_num = np.ceil(shape / self.tile_size).astype(np.int)
        self.empty_tiles = np.ones(self.tile_num, dtype=np.bool)
        self.tile_index = get_indexing_mat(self.empty_tiles)
        self._data = None

    @property
    def data(self):
        return self._data

    @staticmethod
    def indexed_name(name, index):
        return f"__{name}_{index}"

    @staticmethod
    def make_tile(data, tile_size, name=None):
        tile_size = valid_tile_size(data.shape, tile_size)
        tile_num = tuple(np.ceil(data.shape / tile_size).astype(np.int))
        result = {} if name else []
        for index, nd_index in enumerate(np.ndindex(tile_num)):
            indexing = index_to_slice(nd_index, tile_size)
            v = data[indexing]
            if name:
                indexed_name = VirtualData.indexed_name(name, index)
                result[indexed_name] = v
            else:
                result.append(v)
        return result

    def __getitem__(self, inds):
        t_inds = norm_index(inds, self.tile_size)
        load_inds = self.tile_index[t_inds][self.empty_tiles[t_inds]]
        for ind in np.unique(load_inds):
            self.load(ind)

        if self._data is None:
            return None

        return self._data[inds]

    def load(self, index):
        load_name = self.indexed_name(self.name, index)
        assert load_name in self.data_handler.files, \
            f"{load_name} does not exist"
        v = self.data_handler.get(load_name)
        if self._data is None:
            self._data = np.zeros((self.shape), dtype=v.dtype)
        nd_index = tuple([i[0]
                          for i in np.where(self.tile_index == index)])
        if not self.empty_tiles[nd_index]:
            print(f"{index}, {nd_index} was loaded")
        slices = index_to_slice(nd_index, self.tile_size, v.shape)
        self._data[slices] = v
        self.empty_tiles[nd_index] = False


def load_virtual_data(path, variable_list=None):
    success = True
    data_dict = {}
    if not osp.exists(path):
        return False, {}
    try:
        handler = np.load(path, allow_pickle=True)
        if META_TOKEN in handler:
            virtual_meta = handler[META_TOKEN].item()
        else:
            virtual_meta = []

        if variable_list is None:
            # loading all variables
            variable_list = [k for k in handler.files
                             if not k.startswith("__")]
            variable_list.extend(virtual_meta.keys())

        for key in variable_list:
            if key in virtual_meta:
                # virtual data
                meta = virtual_meta[key]
                shape = meta["shape"]
                tile_size = meta["tile_size"]
                obj = VirtualData(shape, tile_size, handler, key)
            elif key in handler:
                # regular data
                obj = handler[key]
                if obj.shape == ():
                    obj = obj.item()
            else:
                # no data
                success = False
                continue
            data_dict[key] = obj

    except Exception:
        success = False
    return success, data_dict


def save_virtual_data(data_dict, path, virtual_meta_dict=(), compress=False):
    new_data_dict = {}
    new_meta_dict = {}
    for key, data in data_dict.items():
        if key in virtual_meta_dict:
            meta = virtual_meta_dict[key]
            tile_size = meta["tile_size"]
            tiled_data_dict = VirtualData.make_tile(data, tile_size, key)
            new_data_dict.update(tiled_data_dict)
            meta["shape"] = data.shape
            new_meta_dict[key] = meta
        else:
            new_data_dict[key] = data
    new_data_dict[META_TOKEN] = new_meta_dict
    if compress:
        np.savez_compressed(path, **new_data_dict)
    else:
        np.savez(path, **new_data_dict)
    return new_data_dict


__all__ = [k for k in globals().keys() if not k.startswith("_")]
