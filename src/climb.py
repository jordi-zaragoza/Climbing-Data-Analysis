import pandas as pd 
import numpy as np
import seaborn as sns

def climb_prediction(X,reg):
    '''
    Documentation:
    This function is used to make a prediction 
    
    Input values: 
    X -> features to be tested
    reg -> model

    Output:
    value predicted 
    '''
    y_pred = reg.predict(X)
    return np.round(y_pred[:,0])

def climb_translation(grade, conversion_table):        
    '''
    Documentation:
    This function translates the num grade (int) into the fra grade
    
    Input values: 
    grade -> num grade
    conversion_table -> conversion table

    Output:
    fra grade
    '''
    return conversion_table.loc[grade,'fra_routes']

def yearly_evolution(country,sex,height,weight,age,years_cl,period = 10):
    '''
    Documentation:
    This function returns a df with the evolution of the climber depending on his/her features
    
    Input values: 
    country,sex,height,weight,age,years_cl -> features
    period -> period of time to be tested

    Output:
    fra grade
    '''
    size = int(period)
    X_user = pd.DataFrame();
    for i in range(0,size):
        X_user_row = {'country':country,'sex':sex,'height':height,'weight':weight,'age':(age+i),'years_cl':(years_cl+i)}
        new_df = pd.DataFrame(X_user_row,index = [i])
        X_user = pd.concat([X_user,new_df])
    return X_user

def yearly_evolution_trans(df,conversion,years_cl):
    X_user = df.copy()
    X_user['grade_fra'] = X_user['grade_spec'].apply(lambda x: conversion.loc[round(x),'fra_routes'])
    X_user['years_future'] = X_user['years_cl']-years_cl
    return X_user      
    
def climb_transformation(df,country_enc,numerical_transform,numerical_columns):
    '''
    Documentation:
    This function does all the transformations of the climber_df in one
    
    Input values: 
    df -> climber_df
    country_enc -> encoder
    numerical_transform -> transformer1
    numerical_columns -> transformer2
    period -> period of time to be tested

    Output:
    climber_df transformed
    '''
    # apply transformations:
    # first split
    numerical = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(include=[object])
    # Categorical transformation
    column_to_encode = 'country'
    onehotlabels = country_enc.transform(categorical[[column_to_encode]]).toarray()
    categorical_jordi_enc = pd.DataFrame(onehotlabels,columns = country_enc.categories_)
    # Numerical transformation
    # For the train set
    num = numerical_transform.transform(numerical)
    numerical_jordi_trans = pd.DataFrame(num,columns = numerical_columns)
    numerical_jordi_trans = numerical_jordi_trans.drop(columns=['started'])
    
    # all together
    return pd.concat([categorical_jordi_enc,numerical_jordi_trans], axis=1) 