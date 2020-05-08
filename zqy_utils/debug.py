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


def search_dir(module, key=""):
    """
    the one-liner dir
    """
    return [i for i in dir(module) if key in i.lower()]


class TimeCounter(object):
    def __init__(self, sync=False, verbose=False):
        self.clocks = OrderedDict()
        self.sync = sync
        self.verbose = verbose
        self._last_name = None

    def tic(self, name):
        if self.sync:
            sync_cuda()
        if name not in self.clocks:
            self.clocks[name] = {"times": [], "last_clock": 0}
        self.clocks[name]["last_clock"] = time.time()
        self._last_name = name

    def toc(self, name=None):
        name = self._last_name if name is None else name
        if self.sync:
            sync_cuda()
        if name in self.clocks:
            time_spend = time.time() - self.clocks[name]["last_clock"]
            if self.verbose:
                print(f"[{name}] {time_spend:.3f}s")
            self.clocks[name]["times"].append(time_spend)

    def toctic(self, name):
        if self._last_name is None:
            # no entry yet
            self.tic("all")
        else:
            self.toc()
        self.tic(name)

    @contextmanager
    def timeit(self, name):
        self.tic(name)
        yield
        self.toc(name)

    def get_time(self, name, mode="mean"):
        if name not in self.clocks:
            return -1
        times = self.clocks[name]["times"]
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

    def get_str(self, mode="mean", deliminator="\n", with_runs=True):
        def _str(name):
            times = self.get_time(name, mode=mode)
            if with_runs:
                count = self.get_time(name, mode="count")
                return f"[{name}] {times:.3f}s/{count}r"
            else:
                return f"[{name}] {times:.3f}s"
        return deliminator.join([_str(name) for name in self.clocks])

    def __repr__(self):
        for name, info in self.clocks.items():
            if len(info["times"]) == 0:
                if self.verbose:
                    print(f"toc on [{name}] for closure")
                self.toc(name)
        return self.get_str(mode="mean", deliminator=" | ", with_runs=True)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
