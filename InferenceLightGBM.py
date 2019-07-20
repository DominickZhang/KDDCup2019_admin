import pandas as pd
import numpy as np
import scipy
import json
from util import timeit

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

#
class InferenceLightGBM(object):

    def __init__(self ,model_json,feature_names, cat_feature_map):

        self.model_json = model_json
        #print(self.model_json)
        self.feature_names = self.model_json['feature_names']
        self.categories = cat_feature_map


    def predict(self ,X):

        try:
            columns = list(X.columns)
        except :
            print('{} should be a pandas.DataFrame'.format(X))

        if self.model_json['feature_names'] == columns:
            y = self._predict(X)
            return y
        else:
            raise Exception("columns should be {}".format(self.feature_names) ,)

    def _sigmoid(self ,z):

        return 1.0 /( 1 + np.exp(-z))

    def _predict(self ,X):
        feat_names = self.feature_names
        results = pd.Series(index=X.index)
        trees = self.model_json['tree_info']
        for idx in X.index:
            X_sample = X.loc[idx:idx ,:]
            leaf_values = 0.0
            for tree in trees:
                tree_structure = tree['tree_structure']
                leaf_value = self._walkthrough_tree(tree_structure ,X_sample)
                leaf_values += leaf_value
            results[idx] = self._sigmoid(leaf_values)
        return results
    #
    # def _walkthrough_tree(self ,tree_structure ,X_sample):
    #
    #     if 'leaf_index' in tree_structure.keys():
    #
    #         return tree_structure['leaf_value']
    #     else:
    #
    #         split_feature = X_sample.iloc[0 ,tree_structure['split_feature']]
    #         decision_type = tree_structure['decision_type']
    #         threshold = tree_structure['threshold']
    #
    #         if decision_type == '==':
    #             feat_name = self.feature_names[tree_structure['split_feature']]
    #             categories = self.categories[feat_name]
    #             category = categories[str(split_feature)]
    #             category = str(category)
    #             threshold = threshold.split('||')
    #             if category in threshold:
    #                 tree_structure = tree_structure['left_child']
    #             else:
    #                 tree_structure = tree_structure['right_child']
    #             return self._walkthrough_tree(tree_structure ,X_sample)
    #         # 数值特征
    #         elif decision_type == '<=':
    #             if split_feature <= threshold:
    #                 tree_structure = tree_structure['left_child']
    #             else:
    #                 tree_structure = tree_structure['right_child']
    #
    #             return self._walkthrough_tree(tree_structure ,X_sample)
    #         else:
    #             print(tree_structure)
    #             print('decision_type: {} is not == or <='.format(decision_type))
    #             return None

    def _walkthrough_tree(self ,tree_structure ,X_sample,tree_index):

        if 'leaf_index' in tree_structure.keys():
            return [tree_index + '_leaf_index_'+str(tree_structure['leaf_index'])]
        else:

            #print(tree_structure['split_feature'])


            split_feature = '%f'%(X_sample[tree_structure['split_feature']])

            decision_type = tree_structure['decision_type']
            threshold = tree_structure['threshold']
            split_index = tree_index + '_split_index_'+str(tree_structure['split_index'])
            if decision_type == '==':
                feat_name = self.feature_names[tree_structure['split_feature']]
                categories = self.categories[feat_name]
                category = categories[str(split_feature)]
                category = str(category)
                threshold = threshold.split('||')
                print(category, threshold)
                if category in threshold:

                    tree_structure = tree_structure['left_child']

                else:
                    tree_structure = tree_structure['right_child']

                tmp = self._walkthrough_tree(tree_structure, X_sample,tree_index)


                return tmp #[split_index] + tmp

            elif decision_type == '<=':
                split_feature = float(split_feature)
                if split_feature <= threshold:
                    tree_structure = tree_structure['left_child']
                else:
                    tree_structure = tree_structure['right_child']

                tmp = self._walkthrough_tree(tree_structure, X_sample,tree_index)

                return tmp #[split_index] + tmp
            else:

                print('decision_type: {} is not == or <='.format(decision_type))
                return [split_index]

    def get_feaure(self,X_sample):
        trees = self.model_json['tree_info']

        features=[]
        for tree in trees:
            tree_structure = tree['tree_structure']
            features.extend(self._walkthrough_tree(tree_structure, X_sample,"tree_index_"+str(tree["tree_index"])))
        return features


    def get_node_id_feature_sparse(self,X):


        pool = ThreadPool(40)
        #results = map(self.get_feaure, np.array(X.values))
        results = pool.map(self.get_feaure, np.array(X.values))

        results = list(results)
        #print(results)
        #results = np.array(results)
        #print(results)
        results = pd.DataFrame(results)

        print(results.columns)
        print("-------------")
        results = pd.SparseDataFrame(pd.get_dummies(results)).astype("float")



        print(results)

        # columns = results.columns
        # results = scipy.sparse.csr_matrix(results)
        print(results.columns)
        return results
