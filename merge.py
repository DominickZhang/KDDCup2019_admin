import os
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import itertools

import CONSTANT
from util import Config, Timer, log, timeit
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_df

NUM_OP = [np.std, np.mean]

import copy



def get_tfidf_vector(X, key_name=None, max_features=10, sparse=False):
    v = TfidfVectorizer(max_features=max_features, dtype=np.float32)
    sparse_output = v.fit_transform(X)
    if sparse:
        return sparse_output
    #print(sparse_output.shape)
    #print(sparse_output.count_nonzero())
    #input()
    Y = pd.DataFrame(sparse_output.toarray())
    Y.index = X.index
    feature_name = v.get_feature_names()
    col_name_dict = dict()
    for i in range(max_features):
        col_name_dict[i] = 'n_' + key_name + '_' + feature_name[i]
    return Y.rename(col_name_dict, axis="columns")


def get_m2m_features(u):

    columns = []
    print(u.columns)
    for col in u.columns:
        if col.startswith("t_0") or col.startswith("c_0") or col.startswith("index"):
            continue
        columns.append(col)
    print(u)
    data = u.values
    group_index = {}
    result_data = []
    print(data.shape)
    print(u.shape)
    input()
    for i in range(u.shape[0]):
        tmp = data[i]
        print(tmp)
        input()
        tmp_index = tmp[-1]
        if tmp_index[0]=="v":
            group_index[tmp[-2]] = tmp
        else:
            if tmp[-2] in group_index:
                result_data.append(group_index[tmp[-2]])
            else:
                result_data.append(list(np.ones(tmp.shape)*np.nan))

    result_data = pd.DataFrame(np.array(result_data))
    print(result_data)
    input()
    result_data.columns = u.columns

    result_data.drop(["index"],axis=1,inplace=True)

    # for c in [c for c in result_data if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
    #     result_data[c].fillna("0", inplace=True)
    #
    #     result_data[c] = result_data[c].apply(lambda x: int(x))

    for c in [c for c in result_data if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        #result_data.drop([c],axis=1,inplace=True)
        try:

            #print(result_data[c])
            result_data[c].fillna("0", inplace=True)
            result_data[c] = result_data[c].astype(str)
            #print(result_data[c])
            result_data["mul_feature_" + c] = result_data[c].str.split(",")
            #print(result_data["mul_feature_" + c])
            #print("------------")
            #result_data["mul_feature_" + c] = result_data[c]

            result_data[c] = result_data["mul_feature_" + c].apply(lambda x: int(x[0]))

        except:
            result_data.drop([c], axis=1, inplace=True)
            continue


    return result_data

    #     #group_index[tmp[]]
    #     #tmp.rehash_c_01
    # data_dict = dict(list(u))
    # for key in data_dict:
    #     print
    #     print(data_dict[key].values.shape)



    #
    # for hash_key in data_dict:
    #     for key2 in data_dict[hash_key]:
    #         table  = data_dict[hash_key][key2]
    #         print(table)
    #         # for line in table:
    #         #     print(line.index)


    # cols = u.columns
    # columns = []
    # for col in cols:
    #     print(col)
    #     if col.startswith("t_"):
    #         continue
    #     columns.append(col)
    #
    # for hash_key in u.rehash_c_01:
    #
    #     for line in u[hash_key]:
    #         if line.index == "v":
    #             print(line)
    #         else:
    #             print(line)






def bfs(root_name, graph, tconfig):
    tconfig[CONSTANT.MAIN_TABLE_NAME]['depth'] = 0
    queue = deque([root_name])
    while queue:
        u_name = queue.popleft()
        for edge in graph[u_name]:
            v_name = edge['to']
            if 'depth' not in tconfig[v_name]:
                tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                queue.append(v_name)


@timeit
def join(u, v, v_name, key, type_):
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    if type_.split("_")[2] == 'many':
        v_features = v.columns
        for c in v_features:
            if c != key and c.startswith(CONSTANT.CATEGORY_PREFIX):
                v[c] = v[c].apply(lambda x: int(x))
        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                     and not col.startswith(CONSTANT.TIME_PREFIX)
                     and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)
                     and 'mul_feature_' not in col}

        v = v.fillna(0).groupby(key).agg(agg_funcs)
        v.columns = v.columns.map(lambda a:
                f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")

        ## --------------- remove duplicated rolling columns ------------
        c_tmp = None
        count = 0
        for c in v.columns:
            if 'COUNT_ROLLING5' in c:
                if c_tmp is None:
                    c_tmp = v[c]
                else:
                    v.drop(c, axis=1, inplace=True)
                    count += 1
        print("There are %d duplicated columns in temporal join!" % count)

        v.reset_index(0, inplace=True)
        v = v.set_index(key)
    else:
        # for c in [c for c in v if c.startswith(CONSTANT.CATEGORY_PREFIX) and "c_0"]:
        #
        #     v[c] = v[c].apply(lambda x: int(x))

        ###-------------- Multi-cat features will be processed in the main function--------###
        for c in [c for c in v if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
            v[c].fillna("0",inplace=True)
            v["mul_feature_" + c] = v[c].apply(lambda x:str(x).split(","))
            #v["mul_feature_" + c] = v[c].str.split(",")
            #print(v["mul_feature_" + c])
            #v["mul_feature_" + c] = v[c]
            v[c] = v["mul_feature_" + c].apply(lambda x: int(x[0]))
        '''
        for c in [c for c in v if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
            v[c].fillna("0",inplace=True)
            mul_features = get_tfidf_vector(v[c], c)
            v.drop(c, axis=1, inplace=True)
            v = pd.concat([v, mul_features], axis=1)
            '''
            #v["mul_feature_" + c] = v[c].parallel_apply(lambda x:str(x).split(","))
            #v["mul_feature_" + c] = v[c].apply(lambda x:str(x).split(","))
            #v["mul_feature_" + c] = v[c].str.split(",")
            #print(v["mul_feature_" + c])
            #v["mul_feature_" + c] = v[c]
            #v[c] = v["mul_feature_" + c].parallel_apply(lambda x: int(x[0]))
            #v[c] = v["mul_feature_" + c].apply(lambda x: int(x[0]))
        #v[key].astype(str, copy=True)
        v = v.set_index(key)
        v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")
    #u[key].astype(str,copy=True)

    '''print(u.dtypes)
    print("*"*50)
    print(v.dtypes)
    print(v.index)
    input()'''

    return u.join(v, on=key)


@timeit
def temporal_join_jinnian(u, v, v_name, key, time_col, type_):
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    if type_.split("_")[2] == 'many':
        timer = Timer()

        tmp_u = u[[time_col, key]]
        timer.check("select")
        # tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
        # timer.check("concat")

        # tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
        for c in v.columns:
            if c != key and c.startswith(CONSTANT.CATEGORY_PREFIX):
                v[c] = v[c].apply(lambda x: int(x))
        tmp_u = pd.concat([tmp_u, v], sort=False)
        #tmp_u = v
        # print(tmp_u.index)
        # input()
        # print(tmp_u[key].nunique())
        # input()
        timer.check("concat")

        rehash_key = f'rehash_{key}'
        # tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
        timer.check("rehash_key")

        # tmp_u.sort_values(time_col, inplace=True)
        timer.check("sort")

        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                     and not col.startswith(CONSTANT.TIME_PREFIX)
                     and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)
                     and 'mul_feature_' not in col
                     }
        tmp_u = tmp_u.fillna(0).groupby(key).agg(agg_funcs)







        '''agg_funcs_num = {col: Config.aggregate_op(col) for col in v if col != key
                     and col.startswith(CONSTANT.NUMERICAL_PREFIX)
                     }
        agg_funcs_cat = {col: Config.aggregate_op(col) for col in v if col != key
                     and col.startswith(CONSTANT.CATEGORY_PREFIX)
                     }
        num_features = [c for c in tmp_u.columns if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
        if key not in num_features:
            num_features.append(key)
        cat_features = [c for c in tmp_u.columns if c.startswith(CONSTANT.CATEGORY_PREFIX)]
        if key not in cat_features:
            cat_features.append(key)
        #print(num_features)
        #print(cat_features)
        #input()
        tmp_u_num = tmp_u[num_features]
        tmp_u_cat = tmp_u[cat_features]'''
        ##---------------FillNA-----------------
        # tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
        ##tmp_u = tmp_u.fillna(0).groupby(rehash_key).rolling(5).agg(agg_funcs)

        '''if len(num_features) > 1:
            tmp_u_cat = tmp_u_cat.groupby(key).agg(agg_funcs_cat)
            tmp_u_cat.reset_index(0, inplace=True)

        if len()


        tmp_u_num = tmp_u_num.fillna(0).groupby(key).agg(agg_funcs_num)
        tmp_u_num.reset_index(0, inplace=True)
        print(tmp_u_cat.index)
        print(tmp_u_cat.columns)
        print(tmp_u_num.index)
        print(tmp_u_num.columns)
        input()
        tmp_u = pd.merge(tmp_u_cat, tmp_u_num, on=[key])'''

        timer.check("group & rolling & agg")

        # tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
        timer.check("reset_index")

        tmp_u.columns = tmp_u.columns.map(lambda a:
                                          f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")


        # new_columns = []
        # for a in tmp_u.columns:
        #     if "collect_list" == a[1]:
        #         new_columns.append(f"{'mul_'}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")
        #     else:
        #         new_columns.append(f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")
        #
        # tmp_u.columns = new_columns
        # print(tmp_u.columns)

        ##-------------remove duplicated rolling columns---------------
        c_tmp = None
        count = 0
        for c in tmp_u.columns:
            if 'COUNT_ROLLING5' in c:
                if c_tmp is None:
                    c_tmp = tmp_u[c]
                else:
                    tmp_u.drop(c, axis=1, inplace=True)
                    count += 1
        print("There are %d duplicated columns in temporal join!" % count)

        # print(tmp_u.columns)
        # input()

        ##------------check whether all n_COUNT_ROLLING_X are the same---------------
        '''all_columns = tmp_u.columns
        print(all_columns)
        c_tmp = None
        for c in all_columns:
            if 'COUNT_ROLLING5' in c:
                if c_tmp is None:
                    c_tmp = tmp_u[c]
                else:
                    print(c)
                    print([(tmp_u[c]-c_tmp).max(), (tmp_u[c]-c_tmp).min()])
        input()'''

        # print(tmp_u.columns)
        tmp_u.reset_index(0, inplace=True)

        ##-------------check NAN after aggregation functions----------
        '''print(tmp_u.columns)
        #print(tmp_u["n_COUNT_ROLLING5(table_1.c_1)"])
        for c in tmp_u.columns:
            print(c)
            print(tmp_u[c].loc['u'].shape[0])
            print(np.sum(np.isnan(tmp_u[c]).loc['u']))
            print(tmp_u[c].loc['u'])
        #print(tmp_u['n_MEAN_ROLLING5(table_1.n_1)'])
        input()'''

        if tmp_u.empty:
            log("empty tmp_u, return u")
            return u
        # print(u.shape,tmp_u.loc['u'].shape,tmp_u_2.shape)
        # ret = pd.concat([u, tmp_u.loc['u'],#tmp_u_2], axis=1, sort=False)
        # ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
        index_tmp = u.index
        u["index"] = list(range(0, len(index_tmp)))
        ret = pd.merge(u, tmp_u, on=[key])
        ret.sort_values("index", inplace=True)
        ret.index = index_tmp
        ret.drop("index", axis=1, inplace=True)

        timer.check("final concat")

        del tmp_u

        return ret

    else:
        ###------------ Multi-cat features will be processed in the main function --------##
        for c in [c for c in v if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
            v[c].fillna("0", inplace=True)
            v["mul_feature_" + c] = v[c].apply(lambda x: str(x).split(","))
            # v["mul_feature_" + c] = v[c].str.split(",")
            # print(v["mul_feature_" + c])
            # v["mul_feature_" + c] = v[c]
            v[c] = v["mul_feature_" + c].apply(lambda x: int(x[0]))
        '''
        for c in [c for c in v if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
            v[c].fillna("0",inplace=True)
            mul_features = get_tfidf_vector(v[c], c)
            v.drop(c, axis=1, inplace=True)
            v = pd.concat([v, mul_features], axis=1)
            '''
        # tmp_u = u[[time_col, key]]
        # tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
        # print(tmp_u[key].nunique())
        # input()
        # print(u.dtypes)
        # input()
        # u[key] = u[key].astype('int64')
        v = v.set_index(key)
        v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")

        return u.join(v, on=key)



@timeit
def temporal_join(u, v, v_name, key, time_col):
    timer = Timer()

    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    tmp_u = u[[time_col, key]]
    timer.check("select")
    #tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    #timer.check("concat")

    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    timer.check("concat")


    rehash_key = f'rehash_{key}'
    tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
    timer.check("rehash_key")

    tmp_u.sort_values(time_col, inplace=True)
    timer.check("sort")



    agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                 and not col.startswith(CONSTANT.TIME_PREFIX)
                 and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)
                 and 'mul_feature_' not in col
                 }

    tmp_u_2 = tmp_u

    ##---------------FillNA-----------------
    #tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
    # tmp_u_2 = tmp_u

    tmp_u = tmp_u.groupby(key).agg(agg_funcs)

    timer.check("group & rolling & agg")

    # tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
    timer.check("reset_index")

    # tmp_u.columns = tmp_u.columns.map(lambda a:
    #    f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")

    if tmp_u.empty:
        log("empty tmp_u, return u")
        return u
    # print(u.shape,tmp_u.loc['u'].shape,tmp_u_2.shape)
    # ret = pd.concat([u, tmp_u_2], axis=1, sort=False)
    # ret = pd.concat([u, tmp_u.loc['u'],tmp_u_2], axis=1, sort=False)

    # ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
    timer.check("final concat")

    tmp_u.columns = tmp_u.columns.map(lambda a:
                                      f"{v_name}.{a})")

    tmpindex = u.index

    u["index"] = list(range(0, len(tmpindex)))

    ret = pd.merge(u, tmp_u, left_index=True, on=[key])

    ret.sort_values("index", inplace=True)

    # ret.index = ret["index"]
    ret.index = tmpindex
    ret.drop("index", axis=1, inplace=True)

    # u[key] = u[key].apply(int)
    # v[key] = v[key].apply(int)
    # #u = u.join(v,on=key)
    # u = u.merge(v)
    # print(u)
    del tmp_u

    return u

def dfs(u_name, config, tables, graph):
    u = tables[u_name]
    log(f"enter {u_name}")
    for edge in graph[u_name]:
        v_name = edge['to']
        if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
            continue

        v = dfs(v_name, config, tables, graph)
        key = edge['key']
        type_ = edge['type']

        if config['time_col'] not in u and config['time_col'] in v:
            continue

        if config['time_col'] in u and config['time_col'] in v:
            log(f"join {u_name} <--{type_}--t {v_name}")
            ## ZJN: change to my temporal_join function
            #u = temporal_join(u, v, v_name, key, config['time_col'])
            u = temporal_join_jinnian(u, v, v_name, key, config['time_col'], type_)
            ## ZJN: Test many-to-many join
            #u = join(u, v, v_name, key, type_)
        else:
            log(f"join {u_name} <--{type_}--nt {v_name}")
            u = join(u, v, v_name, key, type_)

        del v

    log(f"leave {u_name}")
    return u


@timeit
def merge_table(tables, config):
    graph = defaultdict(list)
    for rel in config['relations']:
        ta = rel['table_A']
        tb = rel['table_B']
        graph[ta].append({
            "to": tb,
            "key": rel['key'],
            "type": rel['type']
        })
        graph[tb].append({
            "to": ta,
            "key": rel['key'],
            "type": '_'.join(rel['type'].split('_')[::-1])
        })
    v = tables[CONSTANT.MAIN_TABLE_NAME]
    # for c in [c for c in v if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
    #     v[c] = v[c].apply(lambda x: int(x))

    ## --------- Process Multi-cat features in the maintable-----------##
    '''
    for c in [c for c in v if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        #tfidfVec = TfidfVectorizer(max_features=10, dtype=np.float32)
        #v["mul_feature_" + c] = v[c]
        v[c].fillna("0",inplace=True)
        mul_features = get_tfidf_vector(v[c], c)
        v.drop(c, axis=1, inplace=True)
        v = pd.concat([v, mul_features], axis=1)
        '''
        #v["mul_feature_" + c] = v[c].parallel_apply(lambda x: str(x).split(","))
        #v["mul_feature_" + c] = v[c].apply(lambda x: str(x).split(","))
        #v["mul_feature_" + c] = v[c].str.split(",")
        #print(v["mul_feature_" + c])
        #v[c] = v["mul_feature_" + c].parallel_apply(lambda x: int(x[0]))
        #v[c] = v["mul_feature_" + c].apply(lambda x: int(x[0]))
    #print(v.columns)
    #print(v.dtypes)
    #input()
    for c in [c for c in v if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        #v["mul_feature_" + c] = v[c]
        v[c].fillna("0",inplace=True)
        v["mul_feature_" + c] = v[c].apply(lambda x: str(x).split(","))
        #v["mul_feature_" + c] = v[c].str.split(",")
        #print(v["mul_feature_" + c])
        v[c] = v["mul_feature_" + c].apply(lambda x: int(x[0]))

    tables[CONSTANT.MAIN_TABLE_NAME] = v
    bfs(CONSTANT.MAIN_TABLE_NAME, graph, config['tables'])
    return dfs(CONSTANT.MAIN_TABLE_NAME, config, tables, graph)
