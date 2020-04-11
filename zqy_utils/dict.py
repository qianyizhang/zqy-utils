# -*- coding: utf-8 -*-

"""
the common functionalities
"""


def get_value_from_dict_safe(d, key, default=None):
    """
    get the value from dict
    args:
        d: {dict}
        key: {a hashable key, or a list of hashable key}
            if key is a list, then it can be assumed the d is a nested dict
        default: return value if the key is not reachable, default is None
    return:
        value
    """
    assert isinstance(d, dict), f"only supports dict input, {type(d)} is given"
    if isinstance(key, (list, tuple)):
        for _k in key[:-1]:
            if _k in d and isinstance(d[_k], dict):
                d = d[_k]
            else:
                return default
        key = key[-1]
    return d.get(key, default)


def set_value_to_dict_safe(d, key, value, append=False):
    """
    set the value to dict
    args:
        d: {dict}
        key: {a hashable key, or a list of hashable key}
            if key is a list, then it can be assumed the d is a nested dict
        value: value to be set
        append: if the value is appended to the list, default is False
    return:
        bool: if the value is succesfully set
    """
    assert isinstance(d, dict), f"only supports dict input, {type(d)} is given"
    if isinstance(key, (list, tuple)):
        for _k in key[:-1]:
            if _k in d:
                if isinstance(d[_k], dict):
                    d = d[_k]
                else:
                    return False
            else:
                d[_k] = dict()
                d = d[_k]
        key = key[-1]
    if append:
        if key not in d:
            d[key] = [value]
        elif isinstance(d[key], list):
            d[key].append(value)
        else:
            return False
    else:
        d[key] = value
    return True


def visualize_dict(d, indent="--", level=0):
    """
    print out the the dict strctures, unwraping list by 1-depth
    """
    prefix = indent * level
    if prefix:
        prefix = " |" + prefix
    if isinstance(d, (list, tuple)):
        print(prefix, f"[{type(d[0])} * {len(d)}]")
        return
    elif not isinstance(d, dict):
        print(prefix, f"{type(d)}")
        return

    for k, v in d.items():
        if isinstance(v, (dict, )):
            print(prefix, k)
            visualize_dict(v, indent=indent, level=level + 1)
        elif isinstance(v, (list, tuple)):
            print(prefix, k)
            if isinstance(v[0], (dict, )):
                for i in v:
                    visualize_dict(i, indent=indent, level=level + 1)
            else:
                visualize_dict(v, indent=indent, level=level + 1)
        else:
            print(prefix, k, v)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
