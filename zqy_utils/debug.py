# -*- coding: utf-8 -*-
__all__ = ["list_to_str", "dir"]


def list_to_str(float_list, decimal_count=3, with_bracket=True):
    """
    formatting list of floats
    """
    _string = ", ".join(["{:.%df}" % decimal_count] * len(float_list))
    if with_bracket:
        _string = "[" + _string + "]"
    return _string.format(*float_list)


def dir(module, key=""):
    """
    the one-liner dir
    """
    return [i for i in dir(module) if key in i.lower()]
