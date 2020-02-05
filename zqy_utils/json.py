# -*- coding: utf-8 -*-

import json

__all__ = ["json_read_str", "load_json", "save_json"]


def json_read_str(string):
    """
    the one-liner json string parser
    """

    def invalid_entry(value):
        return value is None or value == ""

    if invalid_entry(string):
        return {}

    if isinstance(string, (str, )):
        try:
            out = json.loads(string)
        except Exception:
            out = {}
            print(string)
        return out
    else:
        return {}


def load_json(filename):
    """
    the one-liner json loader
    """
    if not isinstance(filename, str):
        return None
    with open(filename, 'r') as f:
        result = json.load(f)
    return result


def save_json(to_be_saved, filename):
    """
    the one-liner json saver
    """
    if not isinstance(filename, str):
        return
    with open(filename, 'w') as file_:
        json.dump(to_be_saved, file_)
