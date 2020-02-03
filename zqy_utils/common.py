# -*- coding: utf-8 -*-
"""
the common functionalities
"""
import json
import os
import time


def get_value_from_dict_safe(dict_, key, default=None):
    """
    get the value from dict
    args:
        dict_: {dict}
        key: {a hashable key, or a list of hashable key}
            if key is a list, then it can be assumed the input d is a nested dict
        default: return value if the key is not reachable, default is None
    return:
        value
    """
    assert isinstance(dict_,
                      dict), "only supports dict input, {} is given".format(
                          type(dict_))
    if isinstance(key, list):
        for _k in key[:-1]:
            if _k in dict_ and isinstance(dict_[_k], dict):
                dict_ = dict_[_k]
            else:
                return default
        key = key[-1]
    return dict_.get(key, default)


def set_value_to_dict_safe(dict_, key, value, append=False):
    """
    set the value to dict
    args:
        dict_: {dict}
        key: {a hashable key, or a list of hashable key}
            if key is a list, then it can be assumed the input d is a nested dict
        value: value to be set
        append: if the value is appended to the list, default is False
    return:
        bool: if the value is succesfully set
    """
    assert isinstance(dict_,
                      dict), "only supports dict input, {} is given".format(
                          type(dict_))
    if isinstance(key, list):
        for _k in key[:-1]:
            if _k in dict_:
                if isinstance(dict_[_k], dict):
                    dict_ = dict_[_k]
                else:
                    return False
            else:
                dict_[_k] = dict()
                dict_ = dict_[_k]
        key = key[-1]
    if append:
        if key not in dict_:
            dict_[key] = [value]
        if isinstance(dict_[key], list):
            dict_[key].append(value)
        else:
            return False
    else:
        dict_[key] = value
    return True


def _make_dir(*args):
    """
    the one-liner directory creator
    """
    path = os.path.join(*[arg.strip(" ") for arg in args])
    if not os.path.isdir(path):
        from random import random
        time.sleep(random() * 0.001)
        if not os.path.isdir(path):
            os.makedirs(path)
    return path


def _json_read_str(string):
    """
    the one-liner json string parser
    """

    def invalid_entry(value):
        return value is None or value == ""

    if invalid_entry(string):
        return {}
    if isinstance(string, (str)):
        try:
            out = json.loads(string)
        except Exception:
            out = {}
            print(string)
        return out
    else:
        return {}


def _load_json(filename):
    """
    the one-liner json loader
    """
    if not isinstance(filename, str):
        return None
    with open(filename, 'r') as file_:
        result = json.load(file_)
    return result


def _save_json(to_be_saved, filename):
    """
    the one-liner json saver
    """
    if not isinstance(filename, str):
        return
    with open(filename, 'w') as file_:
        json.dump(to_be_saved, file_)


def _list_to_string(float_list, decimal_count=3, with_bracket=True):
    """
    formatting list of floats
    """
    _string = ", ".join(["{:.%df}" % decimal_count] * len(float_list))
    if with_bracket:
        _string = "[" + _string + "]"
    return _string.format(*float_list)


def _dir(module, key=""):
    """
    the one-liner dir
    """
    return [i for i in dir(module) if key in i.lower()]
