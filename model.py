import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")
os.system("pip3 install --upgrade scikit-learn==0.20")
os.system('pip3 install tensorflow')
os.system('pip3 install keras')
#os.system('pip3 install auto-sklearn')
os.system('pip3 install fastFM2')

import copy
import numpy as np
import pandas as pd

from automl import train_fastfm_batch, train_and_predict_with_time_control_basic,lightgbm_predict_by_split,train_lightgbm
from CONSTANT import MAIN_TABLE_NAME
from merge import merge_table, get_tfidf_vector
from preprocess import clean_df, clean_tables, feature_engineer, remove_trivial_features
from automl import normalize_categorical_features, oneHotEncodingForFastFM
from util import Config, log, show_dataframe, timeit
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from feature_expansion import min_max_func
from feature_expansion import cat_value_counts,onehot_feature_selection, onehot_encoding_without_fit
import CONSTANT
import warnings
warnings.filterwarnings("ignore")
import time


class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        self.y = None

        self.test_mode = False

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

        ##--------Calculate sample size----------
        '''main_table=self.tables[MAIN_TABLE_NAME]
        print(main_table.shape[0])
        print(X_test.shape[0])
        return None'''

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        ## -------------Sort X_train according to the timestamp-------------------
        time_col = self.config['time_col']
        '''print(main_table[time_col])
        print(main_table[time_col].dtype)
        input()'''
        main_table.sort_values(time_col, inplace=True)
        index = main_table.index
        self.y = self.y.reindex(index)
        main_table.reset_index(inplace=True, drop=True)
        self.y.reset_index(inplace=True, drop=True)
        #print(main_table.index)
        #print(self.y.index)
        #input()
        #print(main_table.columns)
        #print(self.y.columns)
        #input()

        if self.test_mode:
            train_ratio = 0.8
            train_size = int(train_ratio * main_table.shape[0])
            X_test = main_table[train_size:]
            main_table = main_table[0:train_size]
            y_test = self.y[train_size:]
            self.y = self.y[0:train_size]

        ##----------concat and merge tables------------------------

        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        ## Clean tables
        clean_tables(Xs)
        #remove_trivial_features_in_tables(Xs)

        ## Merge tables and remove trivial features
        print(main_table.index)
        X = merge_table(Xs, self.config)
        train_index = X.index.str.startswith("train")
        test_index = X.index.str.startswith("test")

        ##------convert m_ features to c_---------##
        new_columns = []
        mul_features = []
        for col in X.columns:
            if "m_" in col and "mul_" not in col:
                new_columns.append("c_"+col)
            elif "mul_" in col:
                mul_features.append(col)
                new_columns.append(col)
            else:
                new_columns.append(col)
        X.columns=new_columns

        print(X.columns)
        clean_df(X)
        print(X.shape)


        ##-------------- Add number frequency feature ----------------
        cat_features = []
        for col in X.columns:
            #if "c_" in col and "ROLLING" not in col and "cnt" not in col:
            if col.split('_')[0] == 'c':
                cat_features.append(col)
        print("cat_features",cat_features)
        X, _ = cat_value_counts(X, cat_features)
        print(X.shape)

        #print(X.dtypes)

        # ##-------------- Reserve multi-cat features ------------------
        # all_features = X.columns
        # tmp_c = None
        # mul_features = []
        #
        # for c in all_features:
        #     if c.split('_')[0] == 'm':
        #         tmp_c = X[c].copy()
        #         tmp_c.fillna("0",inplace=True)
        #         tmp_c = tmp_c.apply(lambda x: str(x).split(","))
        #         X["mul_"+tmp_c.name] = tmp_c
        #         mul_features.append("mul_"+tmp_c.name)
        #         tmp_c = tmp_c.apply(lambda x: int(x[0]))
        #         tmp_c.name = f"{CONSTANT.CATEGORY_PREFIX}{tmp_c.name}"
        #         X = pd.concat([X, tmp_c], axis=1)
        #         print(c)
        #
        # #input()
        # if not (tmp_c is None):
        #     del tmp_c
        # #print(X.columns)
        # print(X.shape)
        # #input()



        ###--------------Change data type------------------
        X.drop([self.config['time_col']], axis=1, inplace=True)
        X = normalize_categorical_features(X)
        for c in X.columns:
            if c.split('_')[0] == 'n':
                X[c] = X[c].astype('float32')
            elif c.split('_')[0] == 'c':
                #X[c] = X[c].apply(lambda x: int(x))
                X[c] = X[c].astype('int32')
            elif c.split('_')[0] == 't':
                X[c] = X[c].values.astype('float32')
            elif c.split('_')[0] == 'm':
                continue
            elif c.split('_')[0] == 'mul':
                continue
            else:
                raise ValueError('Undefined column type: %s'%c)
        print(X.shape)

        ##---------------features split--------------------

        main_features = []
        for feature in X.columns:
            if "mul_" not in feature:
                main_features.append(feature)
        print(main_features)

        print(X.shape)
        ##--------------Remove trivial features-------------------
        remove_trivial_features(X[main_features])
        print(X.shape)





        # ##---------------Multi-cat features--------------------
        # multi_cat_features=[]
        # for c in X.columns:
        #     if c.split('_')[0] == 'm':
        #         multi_cat_features.append(c)
        #
        # if len(multi_cat_features) > 0:
        #     X_multi_cat = X[multi_cat_features]
        #     X.drop(multi_cat_features, axis=1, inplace=True)





        ###-------------Train lightgbm to get an initial result-------------
        result = None
        num_trails = 0
        skip_multi_cat = False

        selection = False
        one_hot_features_m = None

        mlbs_m = None
        mlbs = None

        for i in range(1000):
            random_state=np.random.RandomState(i)

            num_trails = num_trails + 1
            if self.config.time_left()<50:
                num_trails = num_trails-1
                break

            X_train = X[X.index.str.startswith("train")]
            X_test = X[X.index.str.startswith("test")]


            ##-------------data sample ----------------#
            from data_sample import data_sample

            if i==0:

                X_train_sample, y_train_sample = data_sample(X_train, self.y, p_n_ratio=1, ratio=1, random_state_seed=i)

                mul_size = 0
                for mul_feature in mul_features:
                    mul_count_data_tmp = X_train_sample[mul_feature].apply(lambda x:len(x))
                    #print(mul_feature,mul_count_data_tmp.sum(axis=0))
                    mul_size = mul_size + mul_count_data_tmp.sum(axis=0)




                size_fo_train = 60000000

                train_size = csr_matrix(X_train_sample[main_features]).nonzero()[0].size + mul_size
                p_n_ratio = size_fo_train / train_size



            X_train_sample, y_train_sample = data_sample(X_train, self.y, random_state_seed=i, p_n_ratio=p_n_ratio, ratio=1)
            #print(y_train_sample)
            #input()

            if self.config.time_left()<50:
                num_trails = num_trails-1
                break

            print(X_train_sample.shape)
            X_tmp = pd.concat([X_train_sample, X_test])



            print("train test_train shape:train", X_tmp.shape)
            print("train test_train shape:test",X_tmp[0:y_train_sample.shape[0]].shape)
            print("train test_train shape:y",X_tmp[y_train_sample.shape[0]:].shape)



            from feature_expansion import onehot_feature_selection_m

            main_body_start_time = time.time()

            if i>=0:

                if len(mul_features) > 0 :
                    # if i>1 :
                    #     selection=False
                    #     one_hot_features_m, _, mlbs_m = onehot_feature_selection_m(X_tmp, y_train_sample, mlbs_m.keys(),
                    #                                                                             feature_num_everyiter=len(
                    #                                                                                 mlbs_m.keys()),
                    #                                                                             selection=selection)
                    #
                    # else:
                    #     selection = False
                    if i == 0:
                        first_iter_flag = True
                    else:
                        first_iter_flag = False

                    if skip_multi_cat:
                        one_hot_features_m, mul_features = onehot_encoding_without_fit(X_tmp, mul_features, mlbs_m,
                                                                            config = self.config
                                                                            )
                    else:
                        one_hot_features_m, one_hot_models, mlbs_m, mul_features = onehot_feature_selection_m(X_tmp, y_train_sample, mul_features,
                                                                                        config = self.config,
                                                                                        is_first_iter = first_iter_flag,
                                                                                        feature_num_everyiter=len(mul_features),
                                                                                        selection=selection)


                    if self.config.time_left() < 50:
                        num_trails = num_trails-1
                        break

                    if one_hot_features_m is not None:
                        one_hot_features_m = csr_matrix(one_hot_features_m,dtype=np.float32)
                        one_hot_features_m = one_hot_features_m[0:y_train_sample.shape[0],:]

                        print("mul_features shape sparse:",one_hot_features_m.shape)


            ###--------------------data concat--------------------###

            X_train_sample = X_tmp[0:y_train_sample.shape[0]]
            X_test = X_tmp[y_train_sample.shape[0]:]
            X_train_sample = X_train_sample[main_features]
            X_train_sample = csr_matrix(X_train_sample)
            y_train_sample = y_train_sample.values


            if one_hot_features_m is not None:
                X_train_sample = hstack([X_train_sample, one_hot_features_m]).tocsr()

            #if self.config.time_left() < 60:
            #    num_trails = num_trails - 1
            #    break
            if result is None:
                #model = train_and_predict_with_time_control_basic(X_train_sample, X_test, y_train_sample, self.config, random_state=random_state)
                model = train_lightgbm(X_train_sample, y_train_sample, self.config, random_state=random_state, mode = "timestamp")
                del X_train_sample
                result = lightgbm_predict_by_split(model, X_test, main_features, mlbs_m,mul_features,self.config)

                whole_process_time = time.time() - main_body_start_time
                print("Time for the whole process time is: %.4f"%whole_process_time)
                if result is None:
                    mul_features = list()

            else:
                #model= train_and_predict_with_time_control_basic(X_train_sample, X_test, y_train_sample, self.config, random_state=random_state)
                model = train_lightgbm(X_train_sample, y_train_sample, self.config, random_state=random_state, mode="timestamp")
                del X_train_sample
                print("timeleft",self.config.time_left())
                if self.config.time_left() < 50 or model is None:
                    num_trails = num_trails - 1
                    break
                result_tmp = lightgbm_predict_by_split(model, X_test, main_features, mlbs_m,mul_features,self.config)

                if self.config.time_left() < 50 or result_tmp is None:
                    num_trails = num_trails - 1
                    break

                result = result_tmp + result

            '''
            if i>0:
                ###-------------- get sparse training and testing matrix---------
                numeric_table = X_tmp[main_features]
                numeric_table = min_max_func(numeric_table)
                numeric_table = csr_matrix(numeric_table)
                X_train_sample = numeric_table[0:y_train_sample.shape[0],:]

                if len(cat_features) > 0 :

                    if i > 1:
                        selection = False
                        one_hot_features, _, mlbs, le_models = onehot_feature_selection(X_tmp, y_train_sample,
                                                                                        mlbs.keys(),
                                                                                        feature_num_everyiter=len(
                                                                                            mlbs.keys()),
                                                                                        selection=selection)
                    else:

                        one_hot_features, one_hot_models, mlbs, le_models = onehot_feature_selection(X_tmp, y_train_sample,
                                                                                                     cat_features,
                                                                                                     feature_num_everyiter=len(
                                                                                                         cat_features),
                                                                                                     selection=selection)


                if one_hot_features is not None:
                    X_train_sample = hstack([X_train_sample, one_hot_features]).tocsr()

                if one_hot_features_m is not None:
                    X_train_sample = hstack([X_train_sample, one_hot_features_m]).tocsr()

                train_fastfm_batch(X_train_sample, y_train_sample, self.config)

            '''
            ####---------------If there is not enough time for the whole process, skip multi-cat-------------
            if self.config.time_left() < whole_process_time:
                if skip_multi_cat:
                    print("Time is not enough even using basic features!")
                    break
                else:
                    skip_multi_cat = True

            print("Time remaining: %.4f"%self.config.time_left())
            print("-"*50)
            if self.config.time_left()<50:
                break

        if result is None:
            print("Time is not enough for a complete iteration!")
            result = np.zeros(X_test.shape[0])
        else:
            result = result/num_trails
            print(result)
        #result /= num_trails


        ###-------------Ensemble-------------------------
        flag = False
        if flag:

            ###--------------Multi-cat features processing-----------
            if len(multi_cat_features) > 0:
                X_multi_cat_sparse = None
                for c in multi_cat_features:
                    sparse_out = get_tfidf_vector(X_multi_cat[c], max_features=100, sparse=True)
                    if X_multi_cat_sparse is None:
                        X_multi_cat_sparse = sparse_out
                    else:
                        X_multi_cat_sparse = hstack([X_multi_cat_sparse, sparse_out]).tocsr()

                    ## Warning appears here, but there is no effect.
                    X_multi_cat.drop([c], axis=1, inplace=True)
            print("Time remaining: %.4f"%self.config.time_left())
            print("-"*50)

            ###-------------- get sparse training and testing matrix---------
            numeric_table = X[[c for c in X.columns if c.startswith(CONSTANT.NUMERICAL_PREFIX) or c.startswith(CONSTANT.TIME_PREFIX)]]
            X = X[[c for c in X.columns if not (c.startswith(CONSTANT.NUMERICAL_PREFIX) or c.startswith(CONSTANT.TIME_PREFIX))]]
            numeric_table = (numeric_table - numeric_table.min())/(numeric_table.max() - numeric_table.min())
            enc = OneHotEncoder(sparse=True, dtype=np.float32, categories="auto")
            X = enc.fit_transform(X)
            X = hstack((X, numeric_table.values), dtype=np.float32).tocsr()

            del numeric_table
            del enc

            if len(multi_cat_features) > 0:
                X = hstack([X, X_multi_cat_sparse]).tocsr()
                del X_multi_cat_sparse
            print("Time remaining: %.4f"%self.config.time_left())
            print("-"*50)

            ###--------------- train FM and merge result -------------------
            weight = 1.0
            result *= weight
            #result += (1 - weight)*train_fastfm_batch(X[train_index], X[test_index], self.y.values, self.config)


        ## Training process
        #train_with_time_control(X_train, self.y, self.config)

        ## Testing process
        #result = predict(X, self.config)

        ###-------------Train and Predict--------------------
        #result = train_and_predict(X, self.y, self.config)
        #result = train_and_predict_with_concept_drift_zhiqiang(X, self.y, self.config)

        # Can not install autosklearn
        '''from autosklearn import classification
        import sklearn
        X, X_test, y, y_test = sklearn.model_selection.train_test_split(X, self.y, test_size=0.2, random_state=1)
        automl = classification.AutoSklearnClassifier()
        automl.fit(X, y)
        pred = automl.predict(X_test)
        score = roc_auc_score(y_test.values, pred)
        print("Test auc on hold-out dataset: %.4f"%score)
        result=None'''

        if self.test_mode:
            score = roc_auc_score(y_test.values, result)
            print("Test auc on hold-out dataset: %.4f"%score)
            result=None

        return pd.Series(result)
