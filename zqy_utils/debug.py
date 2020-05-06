# -*- coding: utf-8 -*-
import time
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np


def sync_cuda():
    try:
        import torch
        for idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(idx)
    except Exception:
        pass


def list_to_str(float_list, decimal_count=3, with_bracket=True):
    """
    formatting list of floats
    """
    _string = ", ".join(["{:.%df}" % decimal_count] * len(float_list))
    if with_bracket:
        _string = "[" + _string + "]"
    return _string.format(*float_list)


def search_dir(module, key=""):
    """
    the one-liner dir
    """
    return [i for i in dir(module) if key in i.lower()]


class TimeCounter(object):
    def __init__(self, sync=False, print_toc=False):
        self.clocks = OrderedDict()
        self.sync = sync
        self.print_toc = print_toc

    def tic(self, name):
        if self.sync:
            sync_cuda()
        if name not in self.clocks:
            self.clocks[name] = {'times': [], 'last_clock': 0}
        self.clocks[name]['last_clock'] = time.time()
        self._last_name = name

    def toc(self, name=None):
        name = self._last_name if name is None else name
        if self.sync:
            sync_cuda()
        if name in self.clocks:
            time_spend = time.time() - self.clocks[name]['last_clock']
            if self.print_toc:
                print(f"[{name}] {time_spend:.3f}s")
            self.clocks[name]['times'].append(time_spend)

    @contextmanager
    def timeit(self, name):
        self.tic(name)
        yield
        self.toc(name)

    def get_time(self, name, mode="mean"):
        if name not in self.clocks:
            return -1
        times = self.clocks[name]['times']
        if len(times) == 0:
            return -1
        if mode == "mean":
            return np.float32(times).mean()
        elif mode == "last":
            return times[-1]
        elif mode == "sum":
            return np.float32(times).sum()
        elif mode == "raw":
            return times
        elif mode == "count":
            return len(times)
        else:
            return ValueError(f"unknwon mode {mode}")

    def get_keys(self):
        return self.clocks.keys()

    def get_str(self, mode="mean"):
        def _str(name):
            times = self.get_time(name, mode=mode)
            count = self.get_time(name, mode="count")
            return f"[{name}] {times:.3f}s / {count} runs"
        return "\n".join([_str(name) for name in self.clocks])

    def __repr__(self):
        return self.get_str(mode="mean")


__all__ = [k for k in globals().keys() if not k.startswith("_")]
