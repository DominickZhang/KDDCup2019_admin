import datetime

import CONSTANT
from util import log, timeit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import pandas as pd
import numpy as np

FEATURE_MASK_INDEX=[]

@timeit
def feature_select_predict(X, k=100):
    new_feature = np.zeros((X.shape[0], k))
    feature_num = X.shape[1]

    for i in range(k):
        index = FEATURE_MASK_INDEX[i]
        if index < feature_num:
            new_feature[:, i] = np.array(X[X.columns[index]])
        else:
            index_i = (index - feature_num)//feature_num
            index_j = (index - feature_num)%feature_num
            new_feature[:, i] = np.array(X[X.columns[index_i]])*np.array(X[X.columns[index_j]])

    return pd.DataFrame(new_feature)



@timeit
def cross_feature_and_select_poly2(X, y, k=100):
    scores = list()
    indexes = list()
    feature_num = len(X.columns)
    new_feature = np.zeros((X.shape[0], k))
    ## generate cross features
    # first order feature
    for i in range(feature_num):
        column = X.columns[i]
        F, P = f_classif(np.array(X[column]).reshape(-1, 1), np.array(y))
        if len(scores) < k:
            new_feature[:,len(scores)] = np.array(X[column])
            scores.append(F[0])
            indexes.append(i)
        else:
            if F[0]>min(scores):
                index = scores.index(min(scores))
                new_feature[:, index] = np.array(X[column])
                scores[index] = F[0]
                indexes[index] = i

    ## second order feature
    for i in range(feature_num):
        for j in range(feature_num):
            cross_feature = np.array(X[X.columns[i]])*np.array(X[X.columns[j]])
            F, P = f_classif(cross_feature.reshape(-1, 1), np.array(y))
            if len(scores) < k:
                cross_index = i*feature_num+j+feature_num
                new_feature[:,len(scores)] = cross_feature
                scores.append(F[0])
                indexes.append(cross_index)
            else:
                if F[0]>min(scores):
                    cross_index = i*feature_num+j+feature_num
                    index = scores.index(min(scores))
                    new_feature[:, index] = cross_feature
                    scores[index] = F[0]
                    indexes[index] = cross_index

    global FEATURE_MASK_INDEX
    FEATURE_MASK_INDEX = indexes

    return pd.DataFrame(new_feature)

@timeit
def polyfeatures(X):
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    X = pd.DataFrame(X_poly, columns=poly.get_feature_names())
    return X

@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname])


@timeit
def clean_df(df):
    fillna(df)


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if "mul_feature_" in c]:
        df[c].fillna("0", inplace=True)

@timeit
def remove_trivial_features_in_tables(tables):
    for tname in tables:
        log(f"Processing table {tname}")
        remove_trivial_features(tables[tname])

@timeit
def remove_trivial_features(df):
    count = 0
    for c in df.columns:
        if c.split('_')[0] == 'm':
            continue
        if len(df[c].unique()) == 1:
            count += 1
            df.drop(c, axis=1, inplace=True)
    #diff = df.max() - df.min()
    #df = df[df.columns[diff>threshold]]
    #df.drop(df.columns[diff>threshold], axis=1, inplace=True)
    print("There are %d columns of trivial features"%(count))
    #return df

@timeit
def feature_engineer(df, config):
    transform_categorical_hash(df)
    transform_datetime_jinnian(df, config)
    #transform_datetime(df, config)

@timeit
def transform_datetime_jinnian(df, config):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c] = df[c].values.astype('float32')
        #print(df[c])


@timeit
def transform_datetime(df, config):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

    # for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
    #     df["mul_feature_" + c] = df[c].str.split(",")
    #     df[c] = df["mul_feature_" + c].apply(lambda x: int(x[0]))

