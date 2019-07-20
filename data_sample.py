
import pandas as pd
import CONSTANT

def data_sample(X,y, random_state_seed, p_n_ratio=1,ratio=1):
    #x_tmp = X[CONSTANT.MAIN_TABLE_NAME]
    x_tmp = X
    X.index = y.index
    true_num = y.sum()
    num = y.shape[0]


    if float(p_n_ratio*true_num/(num-true_num))>=1:
        #return X,y
        y_new=y
    elif num-true_num>true_num:
        print("p_n_ratio",p_n_ratio)
        y_ = y[y==0].sample(frac=float(p_n_ratio*true_num/(num-true_num)), random_state=random_state_seed)
        y_new = pd.concat([y[y==1],y_])
        #if y_new.shape[0]>1000000:

    elif (num-true_num)<true_num:
        print("p_n_ratio",p_n_ratio)
        y_ = y[y==1].sample(frac=float((num-true_num)*p_n_ratio/true_num), random_state=random_state_seed)
        y_new = pd.concat([y[y==0], y_])
        #if y_new.shape[0]>1000000:
    else:
        y_new = y


    # if y_new.shape[0]>50000:
    #     ratio = float(50000)/y_new.shape[0]
    #     print("ratio", ratio)

    print("ratio", ratio)
    y_new = y_new.sample(frac=ratio, random_state=random_state_seed)
    y_new.sort_index(axis=0,inplace=True)

    X = x_tmp.iloc[y_new.index]

    return X,y_new





