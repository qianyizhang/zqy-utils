# -*- coding: utf-8 -*-
import os
import os.path as osp
import shutil

import numpy as np
from psutil import disk_partitions, disk_usage

from joblib import Parallel, delayed
from zqy_utils import filesize_to_str, make_dir


"""
utility functions to copy from remote disk to multiple local disks
"""


def get_avaliable_disks(min_size=1024, ignored_disks=("/", "/boot")):
    disks_list = []
    for disk in disk_partitions():
        path = disk.mountpoint
        if path in ignored_disks:
            continue
        if disk_usage(path).free > min_size:
            disks_list.append(path)
    return disks_list


def get_ready_disks(disks_list, total=1024):
    for disk in disks_list[:]:
        if not os.access(disk, os.W_OK):
            print(f"cant make dir in {disk}, maybe you dont have right to ")
            disks_list.remove(disk)
    size_list = sorted([disk_usage(path).free for path in disks_list])
    disks_list = sorted(disks_list, key=lambda path: disk_usage(path).free)

    for i, size in enumerate(size_list):
        if size > total/(len(size_list) - 1) + 1:
            break
    else:
        return []
    return disks_list[i:]


def copy_and_link(src_root, relative_path, files_list, target_root, overwrite=True):
    target_dir = make_dir(target_root, relative_path)
    username = os.environ["USER"]
    for path in files_list:
        filename = osp.basename(path)
        src_dir = make_dir(src_root, username, relative_path)
        src_file = osp.join(src_dir, filename)
        dst_file = osp.join(target_dir, filename)
        if osp.exists(dst_file):
            if overwrite:
                os.remove(dst_file)
            else:
                raise ValueError(f"{dst_file} already exist")
        shutil.copy2(path, src_file)
        os.symlink(src_file, dst_file)


def make_copies(src_root_list, target_root, relative_path,
                disks_list=None,
                random=False):
    """
    if relative_path is Noner
    final_root = {target_root}/{relative_path}
    """
    final_root = make_dir(target_root, relative_path)

    if isinstance(src_root_list, str):
        src_root_list = [src_root_list]
    file_list = []
    size_list = []
    for src_root in src_root_list:
        if osp.isfile(src_root):
            file_list.append(src_root)
            size_list.append(osp.getsize(src_root))
        elif osp.isdir(src_root):
            for root, dirs, files in os.walk(src_root):
                for f in files:
                    path = osp.join(root, f)
                    file_list.append(path)
                    size_list.append(osp.getsize(path))
        else:
            raise ValueError(f"unsupported type: {src_root}")

    total_size = sum(size_list)
    total_size_str = filesize_to_str(total_size)
    print(f"copying {len(file_list)}({total_size_str}) to {final_root}")
    if disks_list is None:
        disks_list = get_avaliable_disks()
    # from small to large
    disks_list = get_ready_disks(disks_list)
    # todo: make balanced loaders
    indices = range(len(file_list))
    if random:
        indices = np.random.permutation(indices)
    size = 0
    files = []
    thresh = total_size/len(disks_list)
    mapping_dict = {}
    for index in indices:
        files.append(file_list[index])
        size += size_list[index]
        if size > thresh:
            for disk in disks_list[:]:
                if disk_usage(disk).free > size:
                    mapping_dict[disk] = files
                    print(
                        f"will copy {len(files)}({filesize_to_str(size)}) files to {disk}")
                    files = []
                    size = 0
                    disks_list.remove(disk)
                    break
            else:
                raise ValueError(f"no disk larger than {size}")
    disk = disks_list[0]
    assert len(disks_list) == 1 and disk_usage(
        disk).free > size, "soemthing wrong"
    mapping_dict[disk] = files
    print(f"will copy {len(files)}({filesize_to_str(size)}) files to {disk}")

    param_list = [[disk, relative_path, file_list, target_root]
                  for disk, file_list in mapping_dict.items()]
    Parallel(n_jobs=len(mapping_dict))(delayed(copy_and_link)(*param)
                                       for param in param_list)

    return mapping_dict


__all__ = [k for k in globals().keys() if not k.startswith("_")]
