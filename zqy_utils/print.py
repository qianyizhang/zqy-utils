# -*- coding: utf-8 -*-
import numbers
units = ["Byte", "Kb", "Mb", "Gb", "Tb", "Pb"]


def list_to_str(float_list, decimal_count=3, with_bracket=True):
    """
    formatting list of floats
    """
    _string = ", ".join(["{:.%df}" % decimal_count] * len(float_list))
    if with_bracket:
        _string = "[" + _string + "]"
    return _string.format(*float_list)


def filesize_to_str(filesize, decimal=3):
    index = 0
    while filesize > 1 and index < len(units) - 1:
        if filesize < 500:
            break
        filesize /= 1024
        index += 1
    return f"{filesize:.{decimal}f}[{units[index]}]"


def filesize_to_byte(filesize, unit="Mb"):
    if unit not in units:
        raise ValueError(f"unkown unit {unit}, suported units {units}")
    if isinstance(filesize, numbers.Number):
        return filesize * (1024 ** units.index(unit))
    elif isinstance(filesize, str):
        filesize, _, unit = filesize.partition("[")
        return filesize_to_byte(float(filesize), unit.rstrip("]"))


__all__ = [k for k in globals().keys() if not k.startswith("_")]
