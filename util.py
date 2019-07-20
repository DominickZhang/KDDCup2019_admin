import os
import pickle
import time
from typing import Any

import CONSTANT

nesting_level = 0
is_start = None

class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

def show_dataframe(df):
    if len(df) <= 30:
        print(f"content=\n"
              f"{df}")
    else:
        print(f"dataframe is too large to show the content, over {len(df)} rows")

    if len(df.dtypes) <= 100:
        print(f"types=\n"
              f"{df.dtypes}\n")
    else:
        print(f"dataframe is too wide to show the dtypes, over {len(df.dtypes)} columns")


class Config:
    def __init__(self, info):
        self.data = {
            ## ZJN: Fix number cover bug.
            **info,
            "start_time": time.time()
        }
        self.data["tables"] = {}
        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    @staticmethod
    def aggregate_op(col):
        import numpy as np

        def my_nunique(x):
            return x.nunique()

        def collect_list(group):
            results = []
            for x in list(group):
                if x != 0:
                    results.append(x)
            return results

        my_nunique.__name__ = 'nunique'
        ops = {
            CONSTANT.NUMERICAL_TYPE: ["mean", "std", "sum"],
            # CONSTANT.CATEGORY_TYPE: ["mean", "std"],
            # CONSTANT.CATEGORY_TYPE: ["count", my_nunique],
            CONSTANT.CATEGORY_TYPE: ["count", "mean", "sum", "var"],#collect_list],
            # CONSTANT.CATEGORY_TYPE: [my_nunique]
            #  TIME_TYPE: ["max"],
            # CONSTANT.MULTI_CAT_TYPE: ["count"],
        }
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):
            return ops[CONSTANT.NUMERICAL_TYPE]
        if col.startswith(CONSTANT.CATEGORY_PREFIX):
            return ops[CONSTANT.CATEGORY_TYPE]
        if col.startswith(CONSTANT.MULTI_CAT_PREFIX):
            assert False, f"MultiCategory type feature's aggregate op are not supported."
            #return ops[CONSTANT.MULTI_CAT_TYPE]
        if col.startswith(CONSTANT.TIME_PREFIX):
            assert False, f"Time type feature's aggregate op are not implemented."
        #assert False, f"Unknown col type {col}"

    def time_left(self):
        return self["time_budget"] - (time.time() - self["start_time"])

    def is_enough_time(self):
        return ((time.time() - self["start_time"]) < 0.8*self["time_budget"])

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)


def reset_index(X):
    table_data["index"] = table_data.index

