import pandas as pd
import lightgbm as lgb
from automl import data_split_by_time,data_split


from InferenceLightGBM import InferenceLightGBM
from util import timeit
import scipy


def feature_selection(X,y,feature_num_everyiter=10,feature_names=None, cat_feature_map=None):
    columns = list(X.columns)
    if "timestamp" in columns:
        X.drop(["timestamp"],inplace=True)
    elif "label" in columns:
        X.drop(["label"], inplace=True)
    features = []
    features_tmp = []
    models = []


    len_col = len(columns)

    for i in range(0,int(len_col),feature_num_everyiter):
        if i >= feature_num_everyiter:
            #X_tmp = features_tmp
            cols = X.columns[i:i + feature_num_everyiter]
            X_tmp = pd.concat([X[cols], features_tmp], axis=1)
            cols = list(cols) + ["new_feature"]

        elif i > len(columns) - feature_num_everyiter:
            #X_tmp = features_tmp
            cols = X.columns[i:]
            X_tmp = pd.concat([X[cols], features_tmp], axis=1)
            cols = list(cols)+["new_feature"]

        else:
            cols = X.columns[i:i + feature_num_everyiter]
            X_tmp = X[cols]
            cols = list(cols)

        categorical_feature = []
        for col in cols:
            if "c_" in col:
                categorical_feature.append(col)

        features_tmp,model = train_lightgbm_for_feature_selection(X_tmp,y,cols, categorical_feature)
        models.append(model)
    return features_tmp,models







@timeit
def train_lightgbm_for_feature_selection(X: pd.DataFrame, y: pd.Series,feature_names=None, cat_feature_map=None):
    #num_leaves = int(X.shape[0]/1000)

    params = {
        "learning_rate":0.01,
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4,
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 100,
        'max_depth': 5,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
    }

    # X_sample, y_sample = data_sample_by_time(X, y, 30000)
    #
    # hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)
    #
    # X, y = data_sample_by_time(X, y, 30000)

    # X, y = data_sample_by_time(X, y, 100000)
    # X, y = data_sample_by_time(X, y, 30000)

    X_train = X[0:y.shape[0]]

    print(X_train.shape)
    print(y.shape)

    X_train, X_val, y_train, y_val = data_split(X_train, y, 0.5)





    # X_train_sp = scipy.sparse.csr_matrix(X_train)
    # X_val_sp = scipy.sparse.csr_matrix(X_val)
    #print(X_train_sp)
    #y_train = y_train.values

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)


    #train_data = lgb.Dataset(X_train, label=y_train,feature_name=feature_names,categorical_feature=cat_feature_map)
    #valid_data = lgb.Dataset(X_val, label=y_val,feature_name=feature_names,categorical_feature=cat_feature_map)

    tmpmodel = lgb.train(params,
                                train_data,
                                500,
                                valid_data,
                                early_stopping_rounds=20,
                                #verbose_train=1,
                                verbose_eval=100)

    features = tmpmodel.predict(X_val)

    from sklearn.metrics import roc_auc_score
    import numpy as np
    y_val = np.array(y_val)


    auc = roc_auc_score(y_val,features)


    return auc,tmpmodel
    #inflgb = InferenceLightGBM(tmpconfig.dump_model(),feature_names, cat_feature_map)
    #features = inflgb.get_node_id_feature_sparse(X)
    #return features

