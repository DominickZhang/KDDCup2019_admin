from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, STATUS_FAIL, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import hstack
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)
import keras
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
K = keras.backend
import time
from fastFM2 import sgd

from fm_keras import FMLayer
import CONSTANT
from util import Config, log, timeit
from callbacks import early_stopping_with_time_budget

import json



@timeit
def train_fastfm_batch(X_train, y_train, config, test=False, random_state=np.random.RandomState(0)):
    max_trails = 1

    if test:
        training_size = int(X_train.shape[0]*0.8)
        try:
            #X_test = X_train[training_size:]
            X_train = X_train[0:training_size]
            y_train = y_train[0:training_size]
        except:
            X_train = X_train.tocsr()
            #X_test = X_train[training_size:]
            X_train = X_train[0:training_size]
            y_train = y_train[0:training_size]

    try:
        test_coo = X_train[0]
    except:
        X_train = X_train.tocsr()
        #X_test = X_test.tocsr()

    #result = np.zeros(X_test.shape[0])
    continue_training_flag = True
    count = 0
    model=None
    for i in range(max_trails):
        print('-'*50)
        print('Starting the %d-th trail...'%(i+1))
        print('-'*50)
        if continue_training_flag and (config.time_left() > 30):
            batch_size = int(X_train.shape[0]*1)
            batch_index = random_state.choice(X_train.shape[0], batch_size, replace=False)
            batch_index.sort()
            model = train_process_fastfm(X_train[batch_index], y_train[batch_index], config)

            # pred, continue_training_flag = train_process_fastfm(X_train[batch_index], y_train[batch_index], config)
            # if continue_training_flag:
            #     result += np.squeeze(pred)
            #     count += 1
        else:
            break

    # if count > 0:
    #     result /= float(count)
    #
    # return np.squeeze(result)
    return model

def sample(X, y,random_state, sample_size=30000, n_p_ratio=None, same_ratio=True, is_sort=False):
    pos_index = np.where(y.values > np.mean(y.values))[0]
    pos_num = len(pos_index)
    print("The number of positive labels is: %d"%pos_num)
    neg_index = np.where(y.values < np.mean(y.values))[0]
    neg_num = len(neg_index)
    original_ratio = float(pos_num)/neg_num
    print("The number of negative labels is: %d"%neg_num)
    print("The ratio is: %.4f"%(original_ratio))

    total_num = X.shape[0]
    if n_p_ratio is None:
        if same_ratio:
            pos_sample_num = int(sample_size*original_ratio)
            neg_sample_num = sample_size - pos_sample_num
            pos_sample_index = random_state.choice(pos_index, pos_sample_num, replace=False)
            neg_sample_index = random_state.choice(neg_index, sample_size - pos_sample_num, replace=False)
            print("The number of selected positive labels is :%d"%pos_sample_num)
            print("The number of selected negative labels is :%d"%neg_sample_num)
            print("The ratio is: %.4f"%(float(pos_sample_num)/neg_sample_num))
            sample_index = np.append(pos_sample_index, neg_sample_index)
        else:
            sample_index = random_state.choice(X.shape[0], sample_size, replace=False)
            pos_sample_num = len(np.intersect1d(pos_index, sample_index))
            neg_sample_num = len(np.intersect1d(neg_index, sample_index))
            print("The number of selected positive labels is :%d"%pos_sample_num)
            print("The number of selected negative labels is :%d"%neg_sample_num)
            print("The ratio is: %.4f"%(float(pos_sample_num)/neg_sample_num))

    elif np.max([float(pos_num)/neg_num, float(neg_num)/pos_num]) < n_p_ratio:
        sample_index = random_state.choice(X.shape[0], sample_size, replace=False)
        pos_sample_num = len(np.intersect1d(pos_index, sample_index))
        neg_sample_num = len(np.intersect1d(neg_index, sample_index))
        print("The number of selected positive labels is :%d"%pos_sample_num)
        print("The number of selected negative labels is :%d"%neg_sample_num)
        print("The ratio is: %.4f"%(float(pos_sample_num)/neg_sample_num))

    else:
        if pos_num < neg_num:   
            pos_ratio = 1.0/(n_p_ratio + 1.0)
            pos_sample_num = int(sample_size*pos_ratio)
            if pos_sample_num > pos_num:
                pos_sample_num = pos_num
            pos_sample_index = random_state.choice(pos_index, pos_sample_num, replace=False)
            neg_sample_index = random_state.choice(neg_index, sample_size - pos_sample_num, replace=False)
            sample_index = np.append(pos_sample_index, neg_sample_index)
        else:
            neg_ratio = 1.0/(n_p_ratio + 1.0)
            neg_sample_num = int(sample_size*pos_ratio)
            if neg_sample_num > neg_num:
                neg_sample_num = neg_num
            neg_sample_index = random_state.choice(neg_index, neg_sample_num, replace=False)
            pos_sample_index = random_state.choice(pos_index, sample_size - neg_sample_num, replace=False)
            sample_index = np.append(pos_sample_index, neg_sample_index)

    if is_sort:
        sample_index.sort()
    return sample_index

@timeit
def train_process_fastfm(X_train, y_train, config, random_state=np.random.RandomState(0)):

    continue_training_flag = True

    sample_size = 30000
    ###-------Here is a bug, X_train is not Pandas table
    #sample_index = sample(X_train, y_train)
    #hyperparams = find_best_hyperparams(X_train[sample_index,:], y_train[sample_index], X_train.shape[1], one_hot_features, config["cumulative_shift"])
    hyperparams = dict()
    hyperparams["batch_size"] = X_train.shape[0]
    hyperparams["step_size"] = 5e-5
    hyperparams["rank"] = 50
    
    batch_size = hyperparams["batch_size"]
    step_size = hyperparams["step_size"]
    rank = hyperparams["rank"]

    X_train, X_val, y_train, y_val = numpy_data_split(X_train, y_train, test_size=0.1)
    y_train[y_train<np.mean(y_train)] = -1

    best_validation_metric = -1e5

    step_count = 0
    iteration_count = 0
    iteration_num = 1
    sample_num = X_train.shape[0]
    max_step = 100
    step_default_duration = 1.0 # Default duration is 1s
    step_start = None
    step_duration = None
    step_times = 3

    index = np.arange(X_train.shape[0])

    count = 0
    fm = sgd.FMClassification(n_iter=batch_size, init_stdev=0.1, l2_reg_w=1e-6, l2_reg_V=1e-6, rank=rank, step_size=step_size, random_state=123)
    print("Time remaining: %.4f"%config.time_left())
    print("-"*50)
    while step_count < max_step:
        if (iteration_count == 0) and (step_count > 0):
            random_state.shuffle(index)
            X_train = X_train[index]
            y_train = y_train[index]

        if step_count == 1:
            step_start = time.time()

        if step_count > 0:
            fm.warm_start = True

        #fm.fit(X_train[iteration_count*batch_size: np.min([sample_num, (iteration_count+1)*batch_size])],
        #    y_train[iteration_count*batch_size: np.min([sample_num, (iteration_count+1)*batch_size])])
        fm.fit(X_train, y_train)

        val_result = fm.predict_proba(X_val)
        try:
            val_auc = roc_auc_score(y_val, val_result)
        except:
            print(val_result)
            print(fm.w0_)
            print(fm.w_)
            print(fm.V_)
            input()
        print("#%d: Current val auc is: %.4f, time left: %.2f"%(step_count, val_auc, config.time_left()))

        if val_auc > best_validation_metric:
            best_validation_metric = val_auc
        else:
            print("Auc is not increasing. Stop training...")
            break
            '''if count > 100:
                print("Auc is not increasing. Stop training...")
                break
            else:
                count += 1'''
            


        if (step_duration is None) and not (step_start is None):
            step_duration = time.time() - step_start
            print("*"*20)
            print("The duration of each step is %.2f"%step_duration)
            print("*"*20)

        

        if step_duration is None:
            if config.time_left() < step_times*step_default_duration:
                #model.load_weights("./models/model.h5")
                print("Time is not enough for training FM model!")
                #return np.zeros(X_test.shape[0]), False

        else:
            if config.time_left() < step_times*step_duration:
                continue_training_flag = False
                print("No enough time available, stop training...")
                break

        step_count += 1
        iteration_count += 1
        iteration_count = iteration_count%iteration_num

    #result = fm.predict_proba(X_test)

    #return result, continue_training_flag
    return fm


@timeit
def normalize_categorical_features(X):
    for c in X.columns:
        if c.split('_')[0] == 'c':
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c])
    return X



@timeit
def train_and_predict_with_time_control_basic(X_train: pd.DataFrame, X_test:pd.DataFrame, y:pd.Series, config:Config, random_state, test=False):
    if test:
        X_train, X_test, y, y_test = data_split_by_time(X_train, y, test_size=0.2)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4
    }
    result = None

    ##-------------Generate random index--------------
    #index = random_state.choice(X_train.shape[0], 30000, replace=False)
    #index.sort()
    ### If assign the n_p_ratio, the HPO process will become very slow, which
    ### indicates low convergence speed.
    #index = sample(X_train, y, sample_size=30000, n_p_ratio=20, is_sort=True)
    index = sample(X_train, y, sample_size=30000, is_sort=True, random_state=random_state, same_ratio=False)
    X_sample = X_train.iloc[index]
    y_sample = y.iloc[index]

    hyperparams = hyperopt_lightgbm_basic(X_sample, y_sample, params, config)

    X_train, X_val, y, y_val = data_split_by_time(X_train, y, test_size=0.1)
    train_data = lgb.Dataset(X_train, y)
    val_data = lgb.Dataset(X_val, y_val)

    model = lgb.train({**params, **hyperparams}, train_data, 10000,
                          val_data, early_stopping_rounds=30, verbose_eval=100)


    result = model.predict(X_test)

    return result




def lightgbm_predict_by_split(model,X_test,main_features,mlbs_m,mul_features,config):
    result_list = []
    num_division = 5
    sub_length = int(X_test.shape[0] / num_division)
    for u in range(0, num_division):
        if config.time_left() < 30:
            return None
        if u < num_division-1:
            X_tmp = X_test[main_features].iloc[u * sub_length:(u + 1) * sub_length, :]
            X_tmp_tmp = X_test.iloc[u * sub_length:(u + 1) * sub_length, :]

        else:
            X_tmp = X_test[main_features].iloc[u * sub_length:, :]
            X_tmp_tmp = X_test.iloc[u * sub_length:, :]

        from scipy.sparse import hstack, csr_matrix, vstack
        from feature_expansion import onehot_feature_transform_m

        X_tmp = csr_matrix(X_tmp)

        # if mlbs is not None:
        #     print(cat_features)
        #     print(le_models.keys())
        #     print(mlbs.keys())
        #
        #     one_hot_features = onehot_feature_transform(X_tmp_tmp,mlbs.keys(),mlbs,le_models)
        #     one_hot_features = csr_matrix(one_hot_features)
        #     print("one_hot_features", one_hot_features.shape)
        #     X_tmp = hstack([X_tmp, one_hot_features[:, :]]).tocsr()

        if len(mul_features) > 0:
            try:
                try:
                    one_hot_features_m = onehot_feature_transform_m(X_tmp_tmp, mul_features, mlbs_m)
                    one_hot_features_m = csr_matrix(one_hot_features_m)
                    print("one_hot_features_m", one_hot_features_m.shape)
                    X_tmp = hstack([X_tmp, one_hot_features_m[:, :]]).tocsr()

                    # ZJN: test Memory error
                    #raise MemoryError

                except MemoryError as err:
                    print("*"*50)
                    print("Memory error occurred, try more divisions...")
                    sub_length_tmp = int(X_tmp_tmp.shape[0]/num_division)
                    one_hot_features_m=None
                    for u in range(num_division):
                        if u<num_division-1:
                            X_tmp_tmp_tmp = X_tmp_tmp.iloc[u*sub_length_tmp: (u+1)*sub_length_tmp]
                        else:
                            X_tmp_tmp_tmp = X_tmp_tmp.iloc[u*sub_length_tmp:]
                        one_hot_features_tmp = onehot_feature_transform_m(X_tmp_tmp_tmp, mul_features, mlbs_m)
                        one_hot_features_tmp = csr_matrix(one_hot_features_tmp)
                        print("one_hot_features_m", one_hot_features_tmp.shape)
                        if one_hot_features_m is None:
                            one_hot_features_m = one_hot_features_tmp
                        else:
                            one_hot_features_m = vstack([one_hot_features_m, one_hot_features_tmp]).tocsr()
                    X_tmp = hstack([X_tmp, one_hot_features_m]).tocsr()
            except:
                print("*"*50)
                print("Memory error still occurs, try skipping...")
                return None

        print("test size", X_tmp.shape)
        tmp = model.predict(X_tmp)

        result_list.append(tmp)

    result = np.hstack(result_list)
    print(result.shape)
    return result



@timeit
def hyperopt_lightgbm_basic(X, y, params, config, max_evals=50):
    X_train, X_test, y_train, y_test = data_split_by_time(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = data_split_by_time(X, y, test_size=0.3)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        #"forgetting_factor": hp.loguniform("forgetting_factor", 0.01, 0.1)
        #"max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "max_depth": hp.choice("max_depth", [1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 100,
                        val_data, early_stopping_rounds=30, verbose_eval=0)
        pred = model.predict(X_test)
        score = roc_auc_score(y_test, pred)
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=max_evals, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams



@timeit
def oneHotEncodingForFastFM(X: pd.DataFrame):
    numeric_table = X[[c for c in X.columns if c.startswith(CONSTANT.NUMERICAL_PREFIX) or c.startswith(CONSTANT.TIME_PREFIX)]]
    X = X[[c for c in X.columns if not (c.startswith(CONSTANT.NUMERICAL_PREFIX) or c.startswith(CONSTANT.TIME_PREFIX))]]

    numeric_table = (numeric_table - numeric_table.min())/(numeric_table.max() - numeric_table.min())

    enc = OneHotEncoder(sparse=True, dtype=np.float32, categories="auto")

    X = enc.fit_transform(X)

    X = hstack((X, numeric_table.values), dtype=np.float32).tocsr()

    return X


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config,mode="timestamp"):

        #X = X.sample(frac=1)
        train_lightgbm(X, y, config,mode)



@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    preds = predict_lightgbm(X, config)
    return preds


@timeit
def validate(preds, y_path) -> np.float64:
    score = roc_auc_score(pd.read_csv(y_path)['label'].values, preds)
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config, random_state, mode="timestamp",test=False):

    if test:
        X_train, X_test, y, y_test = data_split_by_time(X, y, test_size=0.2)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4,
        # "learning_rate":0.1,
        # 'feature_fraction': 0.8,
        # 'bagging_fraction': 0.7,
        # "max_depth":5,
        # "num_leaves":50

    }

    sample_flag = True
    while sample_flag:
        X_sample, y_sample = data_sample_automl(X, y, random_state=random_state, nrows=30000)
        if np.mean(y_sample) == 0:
            print("Sample failed!")
            input()
            sample_flag = True
        else:
            sample_flag = False

    ### ----------- print the ratio of pos and neg samples -------------
    print(y.shape)
    print(y_sample.shape)
    pos_index = np.where(y_sample > np.mean(y_sample))[0]
    pos_num = len(pos_index)
    print("The number of positive labels is: %d"%pos_num)
    neg_index = np.where(y_sample < np.mean(y_sample))[0]
    neg_num = len(neg_index)
    print("The number of negative labels is: %d"%neg_num)
    original_ratio = float(pos_num)/neg_num
    print("The ratio is: %.4f"%(original_ratio))
    #input()


    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

    if config.time_left() < 50:
        return None

    #X, y = data_sample_by_time(X, y, 30000)

    #X_train, X_val, y_train, y_val = data_split_by_time(X, y, 0.1)



    print(X.shape)
    print(y.shape)

    #X_train, X_val, y_train, y_val = data_split_by_time(X, y, 0.1)
    X_train, X_val, y_train, y_val = data_split(X, y, 0.1)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)
    #
    # hyperparams = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'metric': {'binary_logloss','auc'},
    #     'num_leaves': 50,
    #     'max_depth': 5,
    #     'feature_fraction': 0.8,
    #     'bagging_fraction': 0.8,
    #     #'bagging_freq': 5,
    #     'learning_rate':0.1
    # }


    model = lgb.train({**params,**hyperparams},#{**hyperparams},
                                train_data,
                                1000,
                                valid_data,
                                early_stopping_rounds=30,
                                verbose_eval=100,
                                callbacks=[early_stopping_with_time_budget(config, reserved_time=50)])

    #rint(config["model"].dump_model());

    return model







@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    return config["model"].predict(X)


@timeit
def hyperopt_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: Dict, config: Config, max_evals=10):
    X_train, X_test, y_train, y_test = data_split_by_time(X_train, y_train, test_size=0.2)
    X_train, X_val, y_train, y_val = data_split_by_time(X_train, y_train, test_size=0.3)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        #"max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "max_depth": hp.choice("max_depth", [1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        if config.time_left() < 50:
            return {'status': STATUS_FAIL}
        else:
            model = lgb.train({**params, **hyperparams}, train_data, 100,
                          valid_data, early_stopping_rounds=10, verbose_eval=0)
            pred = model.predict(X_test)
            score = roc_auc_score(y_test, pred)

            #score = model.best_score["valid_0"][params["metric"]]

            # in classification, less is better
            return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=max_evals, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)

def numpy_data_split(X, y, test_size):
    split_num = int(X.shape[0]*(1-test_size))
    try:
        return X[0:split_num], X[split_num:], y[0:split_num], y[split_num:]
    except:
        X = X.tocsr()
        return X[0:split_num], X[split_num:], y[0:split_num], y[split_num:]


def data_split_by_time(X, y, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    #split_num = int(len(X)*(1-test_size))
    split_num = int(X.shape[0] * (1 - test_size))
    print(type(X))
    if "sparse" in str(type(X)):
        return X[0:split_num], X[split_num:], y[0:split_num], y[split_num:]
    else:
        return X.iloc[0:split_num],X.iloc[split_num:],y.iloc[0:split_num],y.iloc[split_num:]


def data_sample_automl(X: pd.DataFrame, y: pd.Series, random_state, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    if "sparse" in str(type(X)):
        if X.shape[0] > nrows:
            #import random
            index = list(range(0, X.shape[0]))
            #random.shuffle(index)
            random_state.shuffle(index)
            X_sample = X[index[0:nrows]]
            # X_sample = X.sample(nrows, random_state=1)
            y_sample = y[index[0:nrows]]
        else:
            X_sample = X
            y_sample = y
    elif len(X) > nrows:
        index = list(range(X.shape[0]))
        random_state.shuffle(index)
        X_sample = X.iloc[index]
        y_sample = y.iloc[index]
    else:
        X_sample = X
        y_sample = y
    return X_sample, y_sample



def data_sample_sparse(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):

    if X.shape[0] > nrows:
        import random
        index = list(range(0,X.shape[0]))
        random.shuffle(index)
        X_sample = X[index[0:nrows]]
        #X_sample = X.sample(nrows, random_state=1)
        y_sample = y[index[0:nrows]]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample


def data_sample_by_time(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    datasize = len(X)

    if datasize > nrows:
        X_sample = X.iloc[datasize-nrows:]
        y_sample = y.iloc[datasize-nrows:]
        #y_sample = y[X_sample.index]

    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
