import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

import copy
import numpy as np
import pandas as pd

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME
from merge import merge_table
from preprocess import clean_df, clean_tables, feature_engineer
from util import Config, log, show_dataframe, timeit

from feature_expansion import cat_value_counts


class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        self.y = None

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)
        self.y = y
        # clean_tables(Xs)
        # X = merge_table(Xs, self.config)
        # clean_df(X)
        # feature_engineer(X, self.config)
        #train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)
        feature_engineer(X, self.config)

        cat_features = []

        for col in X.columns:
            if "c_" in col and "ROLLING" not in col and "cnt" not in col:
                cat_features.append(col)


        X, _ = cat_value_counts(X, cat_features)



        X_train = X[X.index.str.startswith("train")]



        X = X[X.index.str.startswith("test")]
        X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)

        result = None
        for i in range(0,3):
            train(X_train, self.y, self.config)
            tmp = predict(X, self.config)
            if result == None:
                result = tmp
                continue
            else:
                result = result + tmp

        result = result/float(3)

        return pd.Series(result)
