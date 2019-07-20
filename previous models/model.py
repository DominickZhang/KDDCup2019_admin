import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install sklearn")
os.system("pip3 install pandas==0.24.2")
#os.system("pip install pandarallel")

#from pandarallel import pandarallel
#pandarallel.initialize()

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
        self.one_hot_features = []
        self.one_hot_models = []
        self.mlbs = []

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)
        self.y = copy.deepcopy(y)
        #from data_sample import data_sample
        #Xs, y = data_sample(Xs, y)

        # clean_tables(Xs)
        # X = merge_table(Xs, self.config)
        #
        # clean_df(X)
        # feature_engineer(X, self.config)
        #
        # ###------------------- mul onehot feature -----------------###
        # m_features = []
        #
        # print(X.columns)
        # for col in X.columns:
        #     if ("ROLLING" not in col) and ("mul_feature_" in col):
        #         m_features.append(col)
        #
        # one_hot_features = None
        # one_hot_models = None
        # mlbs = None
        # from feature_expansion import onehot_feature_selection_m
        #
        # if len(m_features) > 0 and int(self.config["time_budget"]) > 100:
        #     self.one_hot_features, self.one_hot_models, self.mlbs = onehot_feature_selection_m(X, y, m_features,
        #                                                                         feature_num_everyiter=len(m_features))






        # clean_tables(Xs)
        # X = merge_table(Xs, self.config)
        # clean_df(X)
        # feature_engineer(X, self.config)
        # train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):


        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]#.iloc[0:4000]
        #X_test = X_test#.iloc[0:4000]
        #self.y = self.y#.iloc[0:4000]
        if int(self.config["time_budget"]) > 2000:
            from data_sample import data_sample
            main_table, self.y = data_sample(main_table, self.y,ratio=1)
            # main_table = Xs[MAIN_TABLE_NAME].iloc[-1000000:]
            # self.y = self.y.iloc[-1000000:]

        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)

        X = merge_table(Xs, self.config)

        clean_df(X)
        feature_engineer(X, self.config)

        ###-------------------- cat feature -----------------------###
        cat_features = []

        for col in X.columns:
            if "ROLLING" not in col and "c_" in col:
                cat_features.append(col)

        X, _ = cat_value_counts(X, cat_features)
        ###--------------------------------------------------------###

        ###------------------- data sample ------------------###

        if int(self.config["time_budget"]) <= 300 :

            X_train = X[X.index.str.startswith("train")]
            X_test = X[X.index.str.startswith("test")]
            from data_sample import data_sample
            X_train, self.y = data_sample(X_train, self.y,flag=True)

            X = pd.concat([X_train, X_test], keys=['train', 'test'])
        elif int(self.config["time_budget"]) < 2000 :
            X_train = X[X.index.str.startswith("train")]
            X_test = X[X.index.str.startswith("test")]
            from data_sample import data_sample
            X_train, self.y = data_sample(X_train, self.y)

            X = pd.concat([X_train, X_test], keys=['train', 'test'])

        #X.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")




        ###------------------- mul onehot feature -----------------###
        m_features = []

        for col in X.columns:
            if ("ROLLING" not in col) and ("mul_feature_" in col):
                m_features.append(col)

        # if len(self.mlbs)>0 or  self.mlbs is not None:
        #     m_features = list(self.mlbs.keys())
        # else:
        #     m_features = []



        one_hot_features = None
        one_hot_models = None
        mlbs = None

        one_hot_features_m = None

        from feature_expansion import onehot_feature_selection_m

        if len(m_features) > 0 and int(self.config["time_budget"]) > 100:
            one_hot_features_m, one_hot_models, mlbs = onehot_feature_selection_m(X, self.y, m_features,
                                                                                feature_num_everyiter=len(m_features),selection=True)
            X.drop(m_features, inplace=True, axis=1)

        elif len(m_features) > 0:
            X.drop(m_features, inplace=True, axis=1)

        ###-------------------------------------------------###



        ###------------------- onehot encoder ------------------###

        from feature_expansion import onehot_feature_selection
        one_hot_features = None
        if len(cat_features) > 0 and int(self.config["time_budget"]) > 4000:
            one_hot_features, one_hot_models, mlbs = onehot_feature_selection(X, self.y, cat_features,
                                                                                  feature_num_everyiter=len(
                                                                                      cat_features),
                                                                                  selection=True)
            for cat_col in cat_features:
                if cat_col not in mlbs:
                    X.drop(cat_col, inplace=True, axis=1)




        ###-----------------------concat--------------------###

        from scipy.sparse import hstack, csr_matrix
        X = csr_matrix(X)
        if one_hot_features is not None:
            X = hstack([X, one_hot_features]).tocsr()

        if one_hot_features_m is not None:
            X = hstack([X, one_hot_features_m]).tocsr()






        ###-------------------------------------------------###


        # ###------------------drop mul_feature---------------###
        # m_features = []
        # for feature in X.columns:
        #     if "mul_feature_" in feature:
        #         m_features.append(feature)
        #
        # X.drop(m_features,inplace=True,axis=1)
        # ###-------------------------------------------------###

        X_train = X[0:self.y.shape[0]]
        X = X[self.y.shape[0]:]

        result = None

        if int(self.config["time_budget"]) < 2000 and int(self.config["time_budget"]) > 300:
            for i in range(0, 3):
                train(X_train, self.y, self.config)
                tmp = predict(X, self.config)
                if result is None:
                    result = tmp
                    continue
                else:
                    result = result + tmp

            result = result / float(3)
        else:
            train(X_train, self.y, self.config)
            result = predict(X, self.config)

        return pd.Series(result)
