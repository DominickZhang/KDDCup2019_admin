import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")
os.system("pip3 install --upgrade scikit-learn==0.20")
os.system('pip3 install tensorflow')
os.system('pip3 install keras')

#os.system("pip3 install fastFM")

import copy
import numpy as np
import pandas as pd
import time

from automl import predict, train, validate,oneHotEncoding, oneHotEncodingCSRMatrix, train_and_predict, train_fm_keras, train_fm_keras_batch
from CONSTANT import MAIN_TABLE_NAME
import CONSTANT
from merge import merge_table
from preprocess import clean_df, clean_tables, feature_engineer
from util import Config, log, show_dataframe, timeit
import tensorflow as tf
from sklearn import metrics

from feature_expansion import cat_value_counts


class Model:
    def __init__(self, info):
        #print("Time before init: %s"%str(time.time()))
        self.config = Config(info)
        #print(self.config["start_time"])
        #print("Time after init: %s"%str(time.time()))
        #print(self.config.time)
        #input()
        self.tables = None
        self.y = None

    @timeit
    def fit(self, Xs, y, time_remain):
        self.tables = copy.deepcopy(Xs)
        self.y = y
        # clean_tables(Xs)
        # X = merge_table(Xs, self.config)
        # clean_df(X)
        # feature_engineer(X, self.config)
        #train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):

        ### calculate time range
        '''Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        print(main_table.columns)
        input()
        min_train_time = np.min(main_table[[c for c in main_table.columns if c.startswith(CONSTANT.TIME_PREFIX)]])
        max_train_time = np.max(main_table[[c for c in main_table.columns if c.startswith(CONSTANT.TIME_PREFIX)]])
        min_test_time = np.min(X_test[[c for c in X_test.columns if c.startswith(CONSTANT.TIME_PREFIX)]])
        max_test_time = np.max(X_test[[c for c in X_test.columns if c.startswith(CONSTANT.TIME_PREFIX)]])

        print("minimum time in training dataset %s"%str(min_train_time))
        print("maximum time in training dataset %s"%str(max_train_time))
        print("minimum time in testing dataset %s"%str(min_test_time))
        print("maximum time in testing dataset %s"%str(max_test_time))
        return None'''


        ### test concept drift
        '''Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        #main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        #main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        main_table = pd.concat([main_table, self.y], axis=1)
        time_feature = [c for c in main_table.columns if c.startswith(CONSTANT.TIME_PREFIX)]
        main_table = main_table.sort_values(time_feature)
        number_test = int(main_table.shape[0]*0.2)
        X_test = main_table.tail(number_test)
        X_test.index = range(X_test.shape[0])
        main_table = main_table.head(main_table.shape[0] - number_test)
        main_table.index = range(main_table.shape[0])


        min_train_time = np.min(main_table[time_feature])
        max_train_time = np.max(main_table[time_feature])
        min_test_time = np.min(X_test[time_feature])
        max_test_time = np.max(X_test[time_feature])

        print("minimum time in training dataset %s"%str(min_train_time))
        print("maximum time in training dataset %s"%str(max_train_time))
        print("minimum time in testing dataset %s"%str(min_test_time))
        print("maximum time in testing dataset %s"%str(max_test_time))

        y_test = X_test[X_test.columns[-1]]
        X_test = X_test[X_test.columns[0:-1]]
        y_train = main_table[main_table.columns[-1]]
        main_table = main_table[main_table.columns[0:-1]]

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
        X_test = X[X.index.str.startswith("test")]

        train(X_train, y_train, self.config)
        result = predict(X_test, self.config)

        fpr, tpr, thresholds=metrics.roc_curve(y_test.values, result, pos_label=1)
        print("test auc is %.4f"%(metrics.auc(fpr, tpr)))
        return None'''

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)
        feature_engineer(X, self.config)

        diff = X.max() - X.min()
        threshold = 1e-6
        X = X[X.columns[diff>threshold]]
        print("There are %d columns of trivial features"%(diff.shape[0] - X.shape[1]))



        '''cat_features = []

        for col in X.columns:
            if "c_" in col and "ROLLING" not in col and "cnt" not in col:
                cat_features.append(col)'''


        #X, _ = cat_value_counts(X, cat_features)

        #X = pd.get_dummies(X, columns = X.columns, sparse=True)
        #cumulative_shift, X = oneHotEncoding(X)
        #self.config["cumulative_shift"] = cumulative_shift


        X_train, X, one_hot_features, all_features = oneHotEncodingCSRMatrix(X)
        #cumulative_shift = X.shape[1]
        self.config["cumulative_shift"] = all_features
        y = self.y.values
        result=None

        #X_train = X[X.index.str.startswith("train")]
        #train(X_train, y, self.config)

        #X = X[X.index.str.startswith("test")]
        #X.index = X.index.map(lambda x: int(x.split('_')[1]))
        #X.sort_index(inplace=True)
        #result = predict(X, self.config)

        #result = train_fm_keras(X_train, X, y, self.config, one_hot_features)
        #input()
        result = train_fm_keras_batch(X_train, X, y, self.config, one_hot_features)


        
        #result = train_and_predict(X_train, y, X, self.config, one_hot_features)

        '''tf.reset_default_graph()
        from tensorflow.python.summary.writer import writer_cache
        #print(writer_cache.FileWriterCache.get('./models/eval'))
        writer_cache.FileWriterCache.clear()

        input()

        os.system("rm -r ./models/*")'''



        '''os.system("rm -r ./models/model.*")
        os.system("rm -r ./models/check*")
        os.system("rm -r ./models/graph.*")
        os.system("rm -r ./models/eval/*")'''

        return pd.Series(result)
