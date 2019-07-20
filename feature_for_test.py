


import pandas as pd
import copy
from CONSTANT import MAIN_TABLE_NAME
import itertools

from merge import merge_table
from preprocess import clean_df, clean_tables, feature_engineer, polyfeatures

from feature_selection import train_lightgbm_for_feature_selection
from util import timeit
from multiprocessing import Pool
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder




@timeit
def baseline_features_test(Xs,X_test,config,m_features,mlbs,one_hot_model):

    main_table = Xs[MAIN_TABLE_NAME]
    main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
    main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")

    Xs[MAIN_TABLE_NAME] = main_table
    clean_tables(Xs)


    X = merge_table(Xs, config)

    clean_df(X)


    from feature_for_test import multi_features_for_test
    X = X[X.index.str.startswith("test")]

    feature_engineer(X, config)
    new_features=None
    
    if len(m_features)>0 and int(config["time_budget"])>300:
        new_features = multi_features_for_test(X, m_features, mlbs, one_hot_model)
        #new_features.index = X.index
        X.drop(m_features, inplace=True, axis=1)
        X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)
        from scipy.sparse import hstack, csr_matrix
        X = csr_matrix(X)
        X = hstack([X,new_features]).tocsr()
        print("------------------")
        print(X.shape)
        #X = pd.concat([X, new_features], axis=1)

    elif len(m_features)>0:
        X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)
        X.drop(m_features, inplace=True, axis=1)
        from scipy.sparse import hstack, csr_matrix
        X = csr_matrix(X)

    else:
        X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)
        from scipy.sparse import hstack, csr_matrix
        X = csr_matrix(X)




    return X


def feature_selection_test(X,models,feature_num_everyiter=10,feature_names=None, cat_feature_map=None):


    features_tmp = []


    columns = list(X.columns)
    len_col = len(columns)
    k=0
    for i in range(0,int(len_col),feature_num_everyiter):
        model=models[k]
        k=k+1
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

        features_tmp = train_lightgbm_for_feature_generation(X_tmp,model)
        features_tmp.index = X.index

    return features_tmp


@timeit
def train_lightgbm_for_feature_generation(X,model):


    features = model.predict(X)
    features = pd.DataFrame(features,columns=["new_feature"])

    return features



@timeit
def cat_value_counts(X_test,cat_dict):
    result = pd.DataFrame([])
    for col in cat_dict:
        X_test = pd.merge(X_test, cat_dict[col], how='left', left_on=col, right_on=col)
        result = pd.concat([result,X_test[cat_dict[col].columns[1]]],axis=1)
        X_test.drop([col,cat_dict[col].columns[0]],axis=1,inplace=True)

    result.index = X_test.index

    return result

@timeit
def multi_features_for_test(df,columns,mlbs,models):

    new_features = {}
    #from multiprocessing import Pool
    #pool = Pool(processes=len(columns))

    for col in columns:
        if col in mlbs:
            mlb = mlbs[col]
            #model = models[col]
            model = None
            new_features[col] = multi_feature_for_one_col(df[col], mlb, model,col) #pool.apply_async(multi_feature_for_one_col, args=(df[col], mlb, model,col))

    new_features_list = []
    for col in columns:
        if col in new_features:
            new_features_list.append(new_features[col])
    from scipy.sparse import hstack
    new_features = hstack(new_features_list,dtype=float)
    #new_features = pd.concat(new_features_list,axis=1)

    return new_features

@timeit
def multi_feature_for_one_col(df,mlb,model,col):

    import warnings
    warnings.filterwarnings('ignore')
    tmp_x = mlb.transform(df.values)
    from scipy.sparse import csr_matrix
    tmp_x = csr_matrix(tmp_x, dtype="float")
    #tmp_x = model.predict(tmp_x)
    new_feature = tmp_x
    #new_feature = pd.DataFrame(tmp_x, columns=["mul_feature_" + col])
    return new_feature

