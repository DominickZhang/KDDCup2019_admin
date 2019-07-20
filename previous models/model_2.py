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



from feature_expansion import baseline_features,timestamp_features, cat_value_counts,onehot_feature_selection

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

    @timeit
    def fit(self, Xs, y, time_ramain):
        #self.tables = copy.deepcopy(Xs)
        Xs, self.y = data_sample(Xs,y)
        self.Xs = copy.deepcopy(Xs)

        X, y, feature_names, cat_feature_map, stampcol= baseline_features(Xs, y, self.config)

        features_from_base, self.feature_selection_models = feature_selection(X, y, int(len(X.columns)/5), feature_names, cat_feature_map)

        X, self.cat_dict_counts = cat_value_counts(X,list(cat_feature_map.keys()))



        X = pd.concat([X, features_from_base], axis=1)


        #
        #
        # one_hot_feature,models = onehot_feature_selection(X, y, cat_feature_map.keys(), feature_num_everyiter=1)
        #
        # one_hot_feature = pd.DataFrame(one_hot_feature,columns=["one_hot_feature"])
        #
        # print(X.shape)
        #
        # X = pd.concat([X,one_hot_feature],axis=1)
        #
        # print(X.shape)


        #features_from_base,self.feature_selection_models = feature_selection(X, y ,20,feature_names, cat_feature_map)

        #
        #timestamp_features(X, y, features_from_base, cat_feature_map,  self.config,stampcol)



        # X=polyfeatures(X)
        # # model=XGBClassifier()
        # # model.fit(X, y)
        # # print(model.feature_importances_)

        train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.Xs

        from feature_for_test import baseline_features_test,cat_value_counts,feature_selection_test

        X_test = baseline_features_test(Xs,X_test,self.config)

        features_from_base = feature_selection_test(X_test,self.feature_selection_models, int(len(X_test.columns)/5))

        X_test = cat_value_counts(X_test,self.cat_dict_counts)

        X_test.index = features_from_base.index

        X_test = pd.concat([X_test, features_from_base], axis=1)




        result = predict(X_test, self.config)



        return pd.Series(result)

