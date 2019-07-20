
import pandas as pd
import copy
import CONSTANT
import itertools

from merge import merge_table
from preprocess import clean_df, clean_tables, feature_engineer, polyfeatures

from feature_selection import train_lightgbm_for_feature_selection
from util import timeit
from multiprocessing import Pool
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MultiLabelBinarizer

from scipy.sparse import csr_matrix


@timeit
def rollByTimestamp(X ,tconfig,
                    window_size,
                    offset,
                    cols,
                    splitnum_list=[100000 ,500000],
                    window_iter_num=1):

    table_data = X[cols+["timestamp","label","index"]]

    # time_col = tconfig['time_col']
    # table_data.sort_values(time_col, inplace=True)
    # table_data['timestamp'] = table_data[time_col].apply(lambda x: int(x.timestamp()))

    timestamp_max = table_data['timestamp'].max()
    timestamp_min = table_data['timestamp'].min()
    timestamp_delta = timestamp_max -timestamp_min

    tmp_data = pd.DataFrame()
    timestamp_cols = []
    splitnum_list = [timestamp_delta]
    for splitnum in splitnum_list:
        if(timestamp_delta >= splitnum):
            split_delta = int(timestamp_delta /splitnum)
            tmp_data[f"{splitnum}_timestamp_{CONSTANT.MAIN_TABLE_NAME}"] = \
                        (table_data['timestamp'] / split_delta).astype(int)
            timestamp_cols.append(f"{splitnum}_timestamp_{CONSTANT.MAIN_TABLE_NAME}")

    table_data = pd.concat([table_data, tmp_data], axis=1)
    #cols = tconfig["tables"][CONSTANT.MAIN_TABLE_NAME]['type']

    for col in cols:

            col_data_set = list(set(table_data[col]))
            if len(col_data_set) * 10000 >= len(table_data):
                continue

            for stamp_col in timestamp_cols:

                tmp_timestamp = list(range(table_data[stamp_col].min(), table_data[stamp_col].max() + 1))

                tmp_pv_table = table_data[[col, stamp_col]].groupby([col, stamp_col]).size()
                tmp_click_table = table_data[[col, stamp_col, "label"]].groupby([col, stamp_col]).sum()

                tmp_window_data = pd.concat([tmp_pv_table, tmp_click_table], axis=1)
                tmp_window_data.columns = [col + "_" + stamp_col + "_pv", col + "_" + stamp_col + "_click"]

                tmp_window_data.reset_index(stamp_col, drop=False, inplace=True)

                result_data_tmp = pd.DataFrame([])
                for col_val in col_data_set:
                    tmp_window_data_2 = pd.DataFrame(itertools.product(tmp_timestamp, [0], [0]))
                    tmp_window_data_2.columns = [stamp_col, col + "_" + stamp_col + "_pv",
                                                 col + "_" + stamp_col + "_click"]

                    tmp_window_data_3 = tmp_window_data.loc[col_val]

                    if tmp_window_data_3.shape[0] < 4:
                        continue

                    tmp_window_data_3 = pd.concat([tmp_window_data_3, tmp_window_data_2])

                    tmp_window_data_3 = tmp_window_data_3.groupby(stamp_col).sum()
                    tmp_window_data_3.sort_values(stamp_col, inplace=True)
                    tmp_window_data_3[col + "_" + stamp_col + "_ctr"] = tmp_window_data_3[
                                                                            col + "_" + stamp_col + "_click"] / (
                                                                                    tmp_window_data_3[
                                                                                        col + "_" + stamp_col + "_pv"] + 1)

                    tmp_window_data_list = [tmp_window_data_3]
                    for i in range(window_iter_num):
                        tmp_window_data_4 = tmp_window_data_list[-1].rolling(window_size).agg(["mean"])
                        # try:
                        tmp_window_data_4.columns = [
                            col + "_" + stamp_col + "_pv_" + str(window_size * (i + 1)) + "_window_mean",
                            col + "_" + stamp_col + "_click_" + str(window_size * (i + 1)) + "_window_mean",
                            col + "_" + stamp_col + "_ctr_" + str(window_size * (i + 1)) + "_window_mean"
                        ]

                        tmp_window_data_list.append(tmp_window_data_4[[
                            col + "_" + stamp_col + "_pv_" + str(window_size * (i + 1)) + "_window_mean",
                            col + "_" + stamp_col + "_click_" + str(window_size * (i + 1)) + "_window_mean",
                            col + "_" + stamp_col + "_ctr_" + str(window_size * (i + 1)) + "_window_mean"
                        ]])

                    if (len(tmp_window_data_list) > 1):
                        tmp_window_data_4 = pd.concat(tmp_window_data_list[1:], axis=1)
                        # print(tmp_window_data_4)
                        tmp_window_data_4.reset_index(stamp_col, drop=False, inplace=True)
                        tmp_window_data_4[stamp_col] = tmp_window_data_4[stamp_col] + offset

                        result_data_tmp_2 = pd.merge(tmp_window_data.loc[col_val], tmp_window_data_4, how='left',
                                                     on=stamp_col, left_index=True)
                        result_data_tmp_2[col] = col_val
                        result_data_tmp_2.drop([col + "_" + stamp_col + "_pv", col + "_" + stamp_col + "_click"],
                                               axis=1, inplace=True)

                        result_data_tmp = pd.concat([result_data_tmp, result_data_tmp_2], axis=0)

                table_data = pd.merge(table_data, result_data_tmp, how='left', on=[col, stamp_col])  # ,left_index=True)
    #print(table_data)
    #table_data.sort_values("index", inplace=True)
    #print(table_data)
    #table_data.index = table_data["index"]
    table_data.drop("index", inplace=True, axis=1)
    y = table_data["label"]
    timestamplist = table_data["timestamp"]
    table_data.drop(["label", "timestamp"], inplace=True, axis=1)
    for stamp_col in timestamp_cols:
        table_data.drop(stamp_col, inplace=True, axis=1)
    for col in cols:
        table_data.drop(col, inplace=True, axis=1)

    table_data.fillna(-1, inplace=True)
    # pca = PCA(n_components=1)
    # table_data = pd.DataFrame(pca.fit_transform(table_data))
    return table_data#, y, timestamplist
    # return result_data


@timeit
def baseline_features(Xs,y,config):

    clean_tables(Xs)
    stampcol = Xs[CONSTANT.MAIN_TABLE_NAME][config["time_col"]].apply(lambda x: int(x.timestamp()))
    main_table = Xs[CONSTANT.MAIN_TABLE_NAME]
    main_table["label"] = y
    main_table["timestamp"] = stampcol
    main_table.sort_values("timestamp", inplace=True)

    tmp_columns = main_table.columns
    main_table = pd.DataFrame(main_table.values)

    main_table.columns = tmp_columns

    #main_table = main_table.iloc[0:40000]
    Xs[CONSTANT.MAIN_TABLE_NAME] = main_table

    y = main_table["label"]
    stampcol = main_table["timestamp"]
    X = merge_table(Xs, config)

    print(X.columns)


    X.drop(["label", "timestamp"], axis=1, inplace=True)

    clean_df(X)
    feature_engineer(X, config)

    cat_feature_map = {}
    for col in X.columns:
        if "c_" in col and "ROLLING" not in col and "cnt" not in col:
            cat_feature_map[col] = set(X[col])

    feature_names = X.columns

    m_features = []
    for feature in feature_names:
        if "mul_feature_" in feature:
            m_features.append(feature)

    one_hot_features=None
    one_hot_models= None
    mlbs=None

    if len(m_features)>0 and int(config["time_budget"])>200000:
        one_hot_features,one_hot_models,mlbs = onehot_feature_selection_m(X, y, m_features, feature_num_everyiter=len(m_features))
        X.drop(m_features,inplace=True,axis=1)
        #X = pd.concat([X, one_hot_features], axis=1)
        from scipy.sparse import hstack,csr_matrix
        X = csr_matrix(X)
        X = hstack([X,one_hot_features]).tocsr()
    elif len(m_features)>0:
        X.drop(m_features, inplace=True, axis=1)
        from scipy.sparse import hstack, csr_matrix
        X = csr_matrix(X)
    else:
        from scipy.sparse import hstack, csr_matrix
        X = csr_matrix(X)

    print("---------------------------------")
    print(X.shape)

    #X.drop(m_features,inplace=True,axis=1)


    # one_hot_features=None
    # one_hot_models = None

    # import random
    # X_tmp = [X]
    # y_tmp = [y]
    # for i in range(5):
    #     cols = list(X.columns)
    #     random.shuffle(cols)
    #     cols_tmp = cols[0:int(len(cols)*0.5)]
    #     X_tmp.append(X[cols_tmp])
    #     y_tmp.append(y)
    #
    # y = pd.concat(y_tmp,axis=0)
    # X = pd.concat(X_tmp, axis=0)



    return X,y,feature_names,cat_feature_map,stampcol,one_hot_features, one_hot_models,m_features,mlbs

@timeit
def timestamp_features(X,y,X_base,cat_feature_map,tconfig,stampcol):

    columns = list(cat_feature_map.keys())
    #X_base["timestamp"] = stampcol
    X["timestamp"] = stampcol
    X["label"] = y
    X["index"] = X.index
    stampcol_max = stampcol.max()
    stampcol_min = stampcol.min()

    offset = int((stampcol_max - stampcol_min) / 1000)
    #import numpy as np
    #X_base = pd.DataFrame(np.ones(X_base.shape),index=X_base.index)

    time_features = None
    for i in range(0,len(columns),3):
        if i<len(columns)-1:
            cols = columns[i:i+3]
            time_features = rollByTimestamp(X, tconfig,
                            10,
                            10,
                            cols,
                            window_iter_num=1)
        else:
            cols = columns[i:]
            time_features = rollByTimestamp(X, tconfig,
                            offset,
                            10,
                            cols,
                            window_iter_num=1)

        time_features = pd.concat([X_base,time_features],axis=1)
        X_base = train_lightgbm_for_feature_selection(time_features, y)

    return X_base


# def __cat_feat(df):
#     if len(df) != 0:
#         cat = _cat_encoder(df.copy())
#         # log = self.cat_gen_feats(df.copy())
#     df.columns = df.columns.map(lambda x: 'CAT_' + str(x))
#     df = __cat_value_counts(df)
#     df = np.concatenate([df, cat], axis=1)
#     return df

#
# def __cat_cnt_feat(df):
#     for col in df.columns:
#         gb = df.groupby(col).size().reset_index()
#         gb.rename(columns={col: col, 0: col + '_cnt'}, inplace=True)
#         df = pd.merge(df, gb, how='left', left_on=col, right_on=col)
#         del df[col]
#     return df

#
#
# def cat_encoder(df,columns):
#     table_data = df
#     df = df[columns]
#     df = df.fillna(0)
#     from category_encoders import OrdinalEncoder
#     enca = OrdinalEncoder().fit(df)
#     cat = enca.transform(df)
#     print(cat)
#     table_data = pd.concat([table_data,cat],axis=1)
#     print(table_data)
#     return table_data



#
#
# @timeit
# def onehot_feature_selection(X,y ,columns,feature_num_everyiter=1):
#     features_tmp = []
#     models = []
#     columns = list(columns)
#     for i in range(0, len(columns), feature_num_everyiter):
#         if i >= feature_num_everyiter:
#
#             cols = columns[i:i+feature_num_everyiter]
#             X_tmp = cat_onehot_encoder(X[cols],cols)
#             from scipy.sparse import hstack
#
#             X_tmp = hstack([X_tmp, features_tmp]).tocsr()
#             cols = list(cols) + ["new_feature"]
#
#         elif i >= len(X.columns)-feature_num_everyiter:
#
#             cols = X.columns[i:]
#             X_tmp = cat_onehot_encoder(X[cols], cols)
#             from scipy.sparse import hstack
#
#             X_tmp = hstack([X_tmp, features_tmp]).tocsr()
#             cols = list(cols) + ["new_feature"]
#
#         else:
#
#             cols = columns[i:i + feature_num_everyiter]
#             X_tmp = cat_onehot_encoder(X[cols], cols)
#             X_tmp = X_tmp
#             cols = list(cols)
#
#         features_tmp, model = train_lightgbm_for_feature_selection(X_tmp, y, cols)
#         models.append(model)
#
#     return features_tmp,models
#
# def cat_onehot_encoder(df,columns):
#     table_data = df
#     df = df[columns]
#     df = df.fillna(1)
#     df = df.abs()
#     new_features = []
#     for col in columns:
#         laencoder = LabelEncoder()
#         onencoder = OneHotEncoder(sparse=True)
#
#         tmp_x = np.array(laencoder.fit_transform(df[col].values.reshape(-1,1)))
#
#         onencoder.fit(tmp_x.reshape(-1, 1))
#
#         new_feature = onencoder.transform(tmp_x.reshape(-1,1))
#         new_features.append(new_feature)
#         # cat = pd.get_dummies(df[col],prefix="onehot_",sparse=True).astype("float")
#         # new_columns = []
#         # for cat_col in cat.columns:
#         #     cat_col = col+"_"+str(cat_col)
#         #     new_columns.append(cat_col)
#         # cat.columns = new_columns
#     from scipy.sparse import hstack
#     new_features = hstack(new_features)
#
#     #new_features = pd.DataFrame(new_features)
#     return new_features



@timeit
def onehot_feature_selection_m(X,y ,columns,config, is_first_iter, feature_num_everyiter=1,selection=True):

    columns = list(columns)
    #new_features, mlbs, models = cat_onehot_encoder_m(X[cols], y)

    new_features = {}
    mlbs = {}
    models = {}

    new_feature_name_list = list()

    #pool = Pool(processes=len(columns))
    count = 0
    try:
        for col in columns:
            # print(col)
            # print(X[col].values)

            if is_first_iter:
                if config.time_left() < 100:
                    print("Remaining time is less than 100 sec")
                    print("There are %d/%d multi-cat features."%(count, len(columns)))
                    break
            else:
                if config.time_left() < 50:
                    print("Remaining time is less than 50 sec")
                    print("There are %d/%d multi-cat features."%(count, len(columns)))
                    break
            count += 1
            tmp_features,tmp_mlb,tmp_model,auc_score = cat_onehot_encoder_m(X[col], y, col,selection)
            if tmp_model is not None:
                if float(auc_score)>0.53:
                    new_features[col], mlbs[col], models[col] = tmp_features,tmp_mlb,tmp_model
            else:

                #features_imp = []
                tmp_features_train = tmp_features[0:y.shape[0], :]
                tmp_features_test = tmp_features[y.shape[0]:, :]

                new_features[col] = tmp_features_train
                mlbs[col] = tmp_mlb
    except:
        print("Error occurred in processing multi-cat features, skipping...")

    new_features_list=[]
    for col in columns:
        if col in new_features:
            new_features_list.append(new_features[col])
            new_feature_name_list.append(col)

    new_features = None
    from scipy.sparse import  hstack
    if len(new_features_list)>0:
        new_features = hstack(new_features_list)

    return new_features,models, mlbs, new_feature_name_list


@timeit
def onehot_encoding_without_fit(X, columns, mlbs, config):
    columns = list(columns)
    one_hot_features_m = None
    new_feature_name_list = list()
    if len(columns) > 0:
        try:
            one_hot_features_m, new_feature_name_list = onehot_feature_transform_m(X, columns, mlbs, config)
            one_hot_features_m = csr_matrix(one_hot_features_m,  dtype=np.float32)
            print("one_hot_features_m", one_hot_features_m.shape)
        except:
            print("Error occurred in processing multi-cat features, skipping...")

    return one_hot_features_m, new_feature_name_list

@timeit
def onehot_feature_transform_m(X,columns,mlbs,config=None):

    columns = list(columns)
    #new_features, mlbs, models = cat_onehot_encoder_m(X[cols], y)

    new_features = {}
    new_feature_name_list = list()

    for col in columns:
        if not (config is None):
            if config.time_left() < 50:
                print("Remaining time is less than 50 sec!!!!!!!")
                break

        df = X[col]
        from scipy.sparse import csr_matrix

        ## ZJN: test memory error
        # raise MemoryError

        features_tmp = mlbs[col].transform(df.values)
        print(features_tmp.shape)
        features_tmp = csr_matrix(features_tmp, dtype=np.float32).tocsr()

        new_features[col] = features_tmp


    new_features_list=[]
    for col in columns:
        if col in new_features:
            new_features_list.append(new_features[col])
            new_feature_name_list.append(col)

    new_features = None
    from scipy.sparse import  hstack
    if len(new_features_list)>0:
        new_features = hstack(new_features_list)

    if config is None:
        return new_features
    else:
        return new_features, new_feature_name_list


'''
@timeit
def onehot_feature_transform_m(X,columns,mlbs):

    columns = list(columns)
    #new_features, mlbs, models = cat_onehot_encoder_m(X[cols], y)

    new_features = {}

    for col in columns:
        df = X[col]
        from scipy.sparse import csr_matrix
        features_tmp = mlbs[col].transform(df.values)
        print(features_tmp.shape)
        features_tmp = csr_matrix(features_tmp, dtype=float).tocsr()

        # features_imp = []
        # X_sum = features_tmp.sum(axis=0)
        # X_sum = X_sum.tolist()[0]
        #
        # for i in range(0, features_tmp.shape[1]):
        #     tmpsum = X_sum[i]
        #     if tmpsum != 0:
        #         features_imp.append(i)
        #
        # features_tmp = features_tmp[:, features_imp]
        # print(features_tmp.shape)
        # print("-------------")
        new_features[col] = features_tmp


    new_features_list=[]
    for col in columns:
        if col in new_features:
            new_features_list.append(new_features[col])

    new_features = None
    from scipy.sparse import  hstack
    if len(new_features_list)>0:
        new_features = hstack(new_features_list)

    #new_features = pd.concat(new_features_list, axis=1)

    return new_features
'''



def cat_onehot_encoder_m(df,y,col,selection=True):
    ## ZJN: test raise memory error
    # raise MemoryError


    mlbs = MultiLabelBinarizer(sparse_output=True).fit(df.values)
    from scipy.sparse import csr_matrix
    features_tmp = mlbs.transform(df.values)
    features_tmp = csr_matrix(features_tmp,dtype=float).tocsr()
    models = None
    auc_score = None
    if selection is True:
        auc_score, models = train_lightgbm_for_feature_selection(features_tmp, y)
        print(col, "auc", auc_score)
    #new_feature = pd.DataFrame(features_tmp,columns=["mul_feature_"+col])
    new_feature = features_tmp
    from scipy.sparse import hstack



    return new_feature,mlbs,models,auc_score



@timeit
def onehot_feature_transform(X ,columns,mlbs,le_models):

    columns = list(columns)


    new_features = {}



    for col in columns:
        df = X[col]
        feat_x = df.values.reshape(-1, 1)
        feat_x = le_models[col].transform(feat_x)
        features_tmp = mlbs[col].transform(feat_x.reshape(-1, 1))
        from scipy.sparse import  csr_matrix
        features_tmp = csr_matrix(features_tmp, dtype=float).tocsr()

        new_features[col] = features_tmp


    new_features_list=[]
    for col in columns:
        if col in new_features:
            new_features_list.append(new_features[col])

    new_features = None
    from scipy.sparse import  hstack
    if len(new_features_list)>0:
        new_features = hstack(new_features_list)


    return new_features

@timeit
def onehot_feature_selection(X,y ,columns,feature_num_everyiter=1,selection=True):

    columns = list(columns)
    #new_features, mlbs, models = cat_onehot_encoder_m(X[cols], y)

    new_features = {}
    mlbs = {}
    models = {}
    le_models = {}

    #pool = Pool(processes=len(columns))

    for col in columns:
        # print(col)
        # print(X[col].values)
        tmp_features,tmp_mlb,tmp_model,auc_score,tmp_le_model = cat_onehot_encoder(X[col], y, col,selection) #pool.apply_async(cat_onehot_encoder_m, args=(X[col], y, col)).get()

        if tmp_model is not None:
            if float(auc_score)>0.53:
                new_features[col], mlbs[col], models[col],le_models[col] = tmp_features,tmp_mlb,tmp_model,tmp_le_model
        else:

            features_imp = []
            tmp_features_train = tmp_features[0:y.shape[0], :]
            tmp_features_test = tmp_features[y.shape[0]:, :]
            X_sum = tmp_features_test.sum(axis=0)

            X_sum = X_sum.tolist()[0]

            #print(X_sum)

            for i in range(0,tmp_features.shape[1]):
                tmpsum=X_sum[i]
                if tmpsum != 0:
                    features_imp.append(i)

            print(tmp_features.shape)

            tmp_features = tmp_features[:,features_imp]

            print(tmp_features.shape)

            new_features[col] = tmp_features[0:y.shape[0], :]
            mlbs[col] = tmp_mlb
            le_models[col] = tmp_le_model


        #print(new_features[col])
        #new_features[col],mlbs[col],models[col] =\

    #pool.close()
    #pool.join()


    new_features_list=[]
    for col in columns:
        if col in new_features:
            new_features_list.append(new_features[col])

    new_features = None
    from scipy.sparse import  hstack
    if len(new_features_list)>0:
        new_features = hstack(new_features_list)
    #new_features = pd.concat(new_features_list, axis=1)

    return new_features,models, mlbs,le_models



def cat_onehot_encoder(df,y,col,selection=True):
    feat_x = df.values.reshape(-1,1)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(feat_x)
    feat_x = le.transform(feat_x)

    mlbs = OneHotEncoder(sparse=True).fit(feat_x.reshape(-1,1))
    from scipy.sparse import csr_matrix
    features_tmp = mlbs.transform(feat_x.reshape(-1,1))
    features_tmp = csr_matrix(features_tmp,dtype=float).tocsr()
    models = None
    auc_score = None
    if selection is True:
        auc_score, models = train_lightgbm_for_feature_selection(features_tmp, y)
        print(col, "auc", auc_score)
    #new_feature = pd.DataFrame(features_tmp,columns=["mul_feature_"+col])
    new_feature = features_tmp




    return new_feature,mlbs,models,auc_score,le




def counts(df, columns):
    df_cnt = pd.DataFrame()
    dict_counts = {}
    for col in columns:
        gb = df.groupby(col).size().reset_index()
        gb.rename(columns={col: col, 0: col+'_cnt'}, inplace=True)
        cnt = pd.merge(df[[col]], gb, how='left', left_on=col, right_on=col)
        df_cnt = pd.concat([df_cnt, cnt[col+'_cnt']], axis=1)
        dict_counts[col] = gb
    return df_cnt,dict_counts


def cat_value_counts_threads(df,cat_feats):
    X = df
    df = df[cat_feats]


    num_threads = 4
    pool = Pool(processes=num_threads)
    col_num = int(np.ceil(df.columns.shape[0] / 4))

    res1,dict_counts1 = pool.apply_async(counts, args=(df, df.columns[:col_num]))
    res2,dict_counts2 = pool.apply_async(counts, args=(df, df.columns[col_num:2 * col_num]))
    res3,dict_counts3 = pool.apply_async(counts, args=(df, df.columns[2 * col_num:3 * col_num]))
    res4,dict_counts4 = pool.apply_async(counts, args=(df, df.columns[3 * col_num:]))
    pool.close()
    pool.join()

    df_counts = pd.concat([X,res1.get(), res2.get(), res3.get(), res4.get()], axis=1)

    dict_counts = dict(dict_counts1.items() + dict_counts2.items() + dict_counts3.items() + dict_counts4.items())

    # df_counts = pd.merge(X, df_counts, how='left', left_on=[cat_feats], right_on=cat_feats)
    return df_counts,dict_counts


def cat_value_counts(df,cat_feats):
    X = df
    df = df[cat_feats]
    df_counts,dict_counts = counts(df, df.columns)
    df_counts.index = X.index
    df_counts = pd.concat([X, df_counts],axis=1)

    return df_counts,dict_counts

#
# def __mv_feat(self, df):
#     if df.__len__() == 0: return pd.DataFrame().values
#
#     def mv_len(x):
#         if isinstance(x, str):
#             return x.split(',').__len__()
#         else:
#             return 0
#
#     # df = pd.concat(tmp, axis=1)
#     df = self.__cat_cnt_feat(df)
#     for cl in df.columns:
#         df.rename(columns={cl: 'multi' + str(cl)}, inplace=True)
#     print('multi cate df *****', df.head())
#     df = df.fillna(0)
#     return df



def z_score_func(df):
    for col in df.columns:
        if col.split("_")[0]=="n":
            if df[col].max()>1:
                df[col] = (df[col] - df[col].mean()) / (df[col].std())
                #df[col] = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    return df




def min_max_func(df):
    for col in df.columns:
        if col.split("_")[0] == "n":
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            df[col] = df[col].astype("float32")
                # df[col] = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    return df




