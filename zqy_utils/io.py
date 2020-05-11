# -*- coding: utf-8 -*-

import gzip
import json
import os
import os.path as osp
import pickle
import shutil
import tarfile
import time
from collections import Callable

import numpy as np
from PIL import Image

from .dicom import sitk, sitk_read_image, is_valid_file

PARSER_EXT_DICT = {"txt": "txt",
                   "pickle": "pkl",
                   "json": "json",
                   "torch": "pth",
                   "sitk": ["dicom", "dcm", "nii", "nii.gz"],
                   "image": ["png", "jpg", "jpeg", "bmp"],
                   "numpy": ["npy", "npz"]}


def _inverse_dict(d):
    inv_d = {}
    for k, v in d.items():
        if isinstance(v, list):
            inv_d.update({_v: k for _v in v})
        else:
            inv_d[v] = k
    return inv_d


EXT_TO_PARSER_DICT = _inverse_dict(PARSER_EXT_DICT)


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


def read_str(string, default_out=None):
    """
    the one-liner string parser
    """
    def invalid_entry(value):
        return value is None or value == ""

    def invalid_type(value):
        raise ValueError()

    if invalid_entry(string):
        return default_out

    if isinstance(string, str):
        parser = json.loads
    elif isinstance(string, bytes):
        parser = pickle.loads
    else:
        parser = invalid_type
    try:
        out = parser(string)
    except Exception:
        out = default_out
        print(string)
    return out


def load(filename, file_type="auto", **kwargs):
    """
    the one-liner loader
    Args:
        filename (str)
        file_type (str): support types: PARSER_EXT_DICT
    """

    if not is_valid_file(filename):
        return None

    if file_type == "auto":
        # check ext reversely
        # eg: a.b.c.d -> ["d", "c.d", "b.c.d", "a.b.c.d"]
        ext = ""
        for token in filename.lower().split(".")[::-1]:
            ext = f"{token}.{ext}".strip(".")
            if ext in EXT_TO_PARSER_DICT:
                file_type = EXT_TO_PARSER_DICT[ext]
                break
        else:
            file_type = "unknown"

    if file_type == "txt":
        with open(filename, "r") as f:
            result = f.readlines()
    elif file_type == "json":
        with open(filename, "r") as f:
            result = json.load(f, **kwargs)
    elif file_type == "pickle":
        with open(filename, "rb") as f:
            result = pickle.load(f, **kwargs)
    elif file_type == "torch":
        import torch
        result = torch.load(filename, map_location=torch.device("cpu"))
    elif file_type == "sitk":
        result = sitk_read_image(filename, **kwargs)
    elif file_type == "image":
        result = Image.open(filename)
        as_np = kwargs.get("as_np", False)
        if as_np:
            result = np.array(result)
    elif file_type == "numpy":
        o = np.load(filename, allow_pickle=kwargs.get("allow_pickle", False))
        if kwargs.get("lazy", False):
            # if its lazy loading, simply return the object
            return o
        if len(o.files) == 1:
            # if only 1 array, return as it is
            return o.get(o.files[0])
        else:
            # asuming is multiple array and load sequentially
            result = {}
            for k in o.files:
                v = o.get(k)
                if v.dtype == "O":
                    v = v.item()
                result[k] = v
            return result
    elif file_type == "unknown":
        raise ValueError(f"Unknown ext {filename}")
    else:
        raise NotImplementedError(f"Unknown file_type {file_type}")

    return result


def save(to_be_saved, filename, file_type="auto", **kwargs):
    """
    the one-liner saver
    Args:
        to_be_saved (any obj)
        filename (str)
        file_type (str): support types: PARSER_EXT_DICT
    """
    if not isinstance(filename, str):
        return None

    if file_type == "auto":
        ext = filename.rpartition(".")[-1]
        file_type = EXT_TO_PARSER_DICT.get(ext, "unknown")

    if file_type == "txt":
        with open(filename, "w") as f:
            f.write(to_be_saved)
    elif file_type == "json":
        with open(filename, "w") as f:
            json.dump(to_be_saved, f)
    elif file_type == "pickle":
        with open(filename, "wb") as f:
            pickle.dump(to_be_saved, f)
    elif file_type == "torch":
        import torch
        torch.save(to_be_saved, filename)
    elif file_type == "sitk":
        sitk.WriteImage(to_be_saved, filename)
    elif file_type == "numpy":
        saver = np.savez_compressed if kwargs.get(
            "compressed", False) else np.savez
        if isinstance(to_be_saved, dict):
            saver(filename, **to_be_saved)
        else:
            saver(filename, to_be_saved)
    elif file_type == "unknown":
        raise ValueError(f"Unknown ext {filename}")
    else:
        raise NotImplementedError(f"Unknown file_type {file_type}")


def recursive_copy(src, dst, softlink=False, overwrite=True, filter_fn=None):
    """
    recursively update dst root files with src root files
    Args:
        src (str): source root path
        dst (str): destination root path
        softlink (bool): Default False, if True, using os.symlink instead of copy
        overwrite (bool): Default True, if overwrite when file already exists
        filter_fn (function): given basename of src path, return True/False
    """
    src_file_list = []

    for root, dirs, files in os.walk(src):
        for filename in files:
            if isinstance(filter_fn, Callable) and not filter_fn(filename):
                continue
            relative_root = root.rpartition(src)[-1].lstrip("/ ")
            src_file_list.append((relative_root, filename))

    make_dir(dst)

    dst_file_list = []
    for root, dirs, files in os.walk(dst):
        for filename in files:
            relative_root = root.rpartition(src)[-1].lstrip("/ ")
            dst_file_list.append((relative_root, filename))

    for f in src_file_list:
        if f in dst_file_list:
            continue
        relative_root = f[0]
        make_dir(dst, relative_root)
        src_path = osp.join(src, *f)
        dst_path = osp.join(dst, *f)
        if osp.exists(dst_path):
            if overwrite:
                if osp.islink(dst_path):
                    if osp.isdir(dst_path):
                        os.rmdir(dst_path)
                    else:
                        os.unlink(dst_path)
                else:
                    os.remove(dst_path)
                print(f"{dst_path} exists, overwrite")
            else:
                print(f"{dst_path} exists, skip")
                continue
        if softlink:
            os.symlink(src_path, dst_path)
        else:
            shutil.copyfile(src_path, dst_path)


def unzip(src_path, dst_path):
    """
    the one-liner unzip function, currently support "gz" and "tgz"
    """
    if osp.isdir(dst_path):
        filename = osp.basename(src_path).rpartition(".")[0]
        dst_path = osp.join(dst_path, filename)

    if src_path.endswith(".gz"):
        with gzip.open(src_path, 'rb') as f_in:
            with open(dst_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    elif src_path.endswith(".tgz"):
        with tarfile.open(src_path, "r:gz") as tar:
            tar.extractall(path=dst_path)

    else:
        raise NotImplementedError(f"unrecognized zip format {src_path}")

    return dst_path


def get_folder_size(path):
    """
    credit to:
    https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python/1392549#1392549
    """
    nbytes = sum(osp.getsize(f)
                 for f in os.listdir(path) if osp.isfile(f))
    return nbytes


__all__ = [k for k in globals().keys() if not k.startswith("_")]
