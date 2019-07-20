import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")
os.system("pip3 install category_encoders")
os.system("pip3 install scikit-learn")

#os.system("pip3 install xgboost")

import copy
import numpy as np
import pandas as pd
#from xgboost import XGBClassifier

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME

from util import Config, log, show_dataframe, timeit
from feature_selection import feature_selection
from data_sample import data_sample



from feature_expansion import baseline_features,timestamp_features, cat_value_counts

class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None

        self.time_window_config = {}
        self.feature_selection_models = []

        self.feature_selection_window = 40
        self.Xs=None
        self.y=None
        self.cat_dict_counts = []
        self.one_hot_models = None
        self.one_hot_features = None

    @timeit
    def fit(self, Xs, y, time_ramain):
        #self.tables = copy.deepcopy(Xs)
        # Xs, self.y = data_sample(Xs,y)
        # self.Xs = copy.deepcopy(Xs)
        self.Xs = copy.deepcopy(Xs)

        X, y, feature_names, cat_feature_map, stampcol,self.one_hot_features, self.one_hot_models, self.m_features ,self.mlbs = baseline_features(Xs, y, self.config)


        train(X, y, self.config)
    #
    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.Xs

        from feature_for_test import baseline_features_test

        if self.one_hot_features is not None:
            X_test = baseline_features_test(Xs,X_test,self.config,self.m_features,self.mlbs,self.one_hot_models)
        else:
            X_test = baseline_features_test(Xs, X_test, self.config, [], None, None)

        result = predict(X_test, self.config)

        return pd.Series(result)

