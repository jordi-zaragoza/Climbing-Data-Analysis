import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# All transfroms in one function
def all_transform(df,cat_enc,num_enc_1,num_enc_2):
    '''
    This function does all the transformation together
    inputs:
    df -> dataframe that we want to transform
    cat_enc -> encoder
    num_enc_1 -> transformer 1
    num_enc_2 -> transformer 2
    
    output:
    it returns the df transformed
    '''
    numerical, categorical= cat_num_split(df)    
    categorical_enc = cat_transform_hot(categorical,cat_enc,'country')  
    X_tot = pd.concat([numerical,categorical_enc], axis=1) 
    X_tot_trans = num_transform(X_tot, num_enc_1)
    X_tot_trans2 = num_transform(X_tot_trans, num_enc_2)  
    return X_tot_trans2

# X,y split
def xy_split(df):
    X = df[['country','sex','height','weight','age','years_cl']]
    y = df[['grades_max']]
    return X,y

# X,y split
def tr_ts_split(X,y):  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)
    X_train.reset_index(inplace=True,drop=True)
    X_test.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_test.reset_index(inplace=True,drop=True)
    return X_train, X_test, y_train, y_test

# Categorical / numerical split
def cat_num_split(df):
    numerical= df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])
    return numerical, categorical

# Categorical get encoder
def cat_get_enc(df_train,column_to_encode):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_train[[column_to_encode]])
    return enc

# Categorical use encoder
def cat_transform_hot(df,enc,column_to_encode):
    onehotlabels = enc.transform(df[[column_to_encode]]).toarray()
    df_enc = pd.DataFrame(onehotlabels,columns = enc.categories_[0])
    return df_enc

# Numerical transformations
def num_get_tr_minmax(df_train):
    return preprocessing.MinMaxScaler().fit(df_train)

def num_get_tr_power(df_train):
    return preprocessing.PowerTransformer().fit(df_train)

def num_get_tr_scale(df_train):
    return preprocessing.StandardScaler().fit(df_train)

# Numerical transformations
def num_transform(df, enc):
    num_test = enc.transform(df)
    return pd.DataFrame(num_test,columns = df.columns)
