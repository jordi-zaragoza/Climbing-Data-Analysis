from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# -------------- MAIN CLEAN --------------------------------------------------
def clean_climber(df,conversion):
    '''
    Documentation:
    This function does the clinning of the climber df
    
    Input values: 
    df -> Climber table
    conversion -> conversion table

    Output:
    the climber df cleaned
    '''
    print("Before cleaning the table ")
    print(df.shape)
    climber_df = df.copy()
    
    # We want to put all the climbers at their max level on the last climb entry on the database:
    climber_df.age = climber_df.age + (climber_df.year_last - 2017) # 2017 is this year
    climber_df.years_cl = climber_df.years_cl + (climber_df.year_last - 2017)
    
    
    # Clean outliers for men and women sepparately
    # climber_split = climber_df[['height','weight','grades_max','sex','started']]
    climber_split_men = climber_df[climber_df['sex']==0]
    climber_split_women = climber_df[climber_df['sex']==1]

    climber_split_women_clean = climber_split_women[climber_split_women.weight < 70]
    climber_split_women_clean = climber_split_women_clean[climber_split_women_clean.height < 177]
    climber_split_women_clean = climber_split_women_clean[climber_split_women_clean.height > 149]
    climber_split_women_clean = climber_split_women_clean[climber_split_women_clean['years_cl'] < 25] # less than 25
    climber_split_women_clean = climber_split_women_clean[climber_split_women_clean['age'] < 43] # less than 43
    
    climber_split_men_clean = climber_split_men[climber_split_men.weight > 45]
    climber_split_men_clean = climber_split_men_clean[climber_split_men_clean.height > 162]
    climber_split_men_clean = climber_split_men_clean[climber_split_men_clean.height < 194]
    
    climber_all = pd.concat([climber_split_men_clean,climber_split_women_clean],axis=0)
    
    # Filters for both:
    climber_all = climber_all[climber_all['grades_max'] > 28] # 5a - min grade
    climber_all = climber_all[climber_all['grades_max'] < 79] # 9c - max grade
    
    climber_all = climber_all[climber_all['grades_count'] > 2] # more than 4 routes climbed
    
    climber_all = climber_all[climber_all['years_cl'] > 2] # more than 2 years
    climber_all = climber_all[climber_all['age'] > 14] # more than 12 years of age
    climber_all = climber_all[climber_all['age'] < 50] # more than 12 years of age
#     climber_all = climber_all[climber_all['year_last'] > 2016] # Because we 'are' in 2017 an want to analyze the climbers from last year
    
    # We drop some columns
    climber_all = climber_all.drop(columns = ['date_first','date_last','grades_count','grades_first','grades_last','year_first'])
    
    climber_all.reset_index(drop = True, inplace = True )
    
    # Get the conversion grades
    climber_all['max_fra'] = climber_all['grades_max'].apply(lambda x: conversion.loc[round(x),'fra_routes'])
    
    print("\n\nAfter cleaning the table ")
    print(climber_all.shape)
    return climber_all

# ------------ KNN Method for NAs -----------------------------

def KNN_method(df, k_neighbors, col_eval, cols_train):
    '''
    Documentation:
    This function replaces NA of a column using the KNN method with the other columns asigned
    
    Input values: 
    df -> Pandas dataframe
    k_neighbors -> Number K of neighbors we will use fir the KNN method
    col_eval -> The column name of the df with NAs we need to fill
    cols_train -> List of column names of the df that we will use for the KNN Method (with no NA)
    
    Output:
    The df with the NAs of column col_eval replaced.
    '''
    categoricals = df.select_dtypes(exclude = np.number)
    numericals = df.select_dtypes(np.number)
    
    # Values without NA
    rows_without_na = numericals[numericals[col_eval].notna()].index
    numericals_with_value = numericals[numericals[col_eval].notna()]
    numericals_with_value.shape
    
    # We need to do the X-y split.
    X_value = numericals_with_value[cols_train]
    y_value = numericals_with_value[col_eval]
    
    # We scale it
    scaler = MinMaxScaler()
    scaler.fit(X_value)
    X_normalized_value_np = scaler.transform(X_value)
    X_normalized_value_df = pd.DataFrame(X_normalized_value_np, columns=X_value.columns)
    X_normalized_value_df
    
    # We apply Knn classifier
    knn = KNeighborsRegressor(n_neighbors=k_neighbors) 
    knn.fit(X_normalized_value_df,y_value)
    pred = knn.predict(X_normalized_value_df)       
    print("The R2 of the model is: {:.2f}".format(round(r2_score(y_value, pred),2)))
    
    # We need to provide the indexes we stored earlier
    predicted_value = pd.DataFrame(pred,columns=['Predicted_value'], index = rows_without_na)
    numericals2 = pd.concat([numericals,predicted_value],axis=1)    
    print("1. Old mean: ", np.round(numericals2[col_eval].mean(),2), " vs new mean: ", np.round(numericals2['Predicted_value'].mean(),2))
    
    # Values with NA
    data_with_na = numericals2[numericals2[col_eval].isna()]
    rows_with_na = numericals2[numericals2[col_eval].isna()].index
    numericals_with_na = data_with_na.select_dtypes(np.number)
    numericals_with_na.head()
    
    # We need to be careful to select only the columns present in the dataframe that we've used to train the model.
    # In particular, the "Predicted_value" column was not there
    X_value_na = numericals_with_na[cols_train]
    
    # We apply the transformer:
    X_normalized_value_na_np = scaler.transform(X_value_na)
    X_normalized_value_na_df = pd.DataFrame(X_normalized_value_na_np, columns=X_value_na.columns)
    X_normalized_value_na_df
    
    # Now we're ready to make predictions with our model for the rows with NA's
    pred_na = knn.predict(X_normalized_value_na_df)
    
    # We create a new column to hold the final values of the value (initially filled with 0's)
    numericals2['Final_value'] = 0

    # Now let's replace the values of the new column with:
    # The old values of value when they exist
    # The predictions when value had NA's
    col_final_value = list(numericals2.columns).index('Final_value')
    col_value = list(numericals2.columns).index(col_eval)
    numericals2.iloc[rows_with_na, col_final_value] = pred_na
    numericals2.iloc[rows_without_na, col_final_value] = numericals2.iloc[rows_without_na, col_value]
    print("2. Old mean: ", np.round(numericals2[col_eval].mean(),2), " vs new mean: ", np.round(numericals2['Final_value'].mean(),2))    
    
    # Now let's replace the value with the new FINAL_value values and drop the additional columns
    numericals2[col_eval] = numericals2['Final_value']
    numericals2.drop(['Predicted_value','Final_value'],axis=1,inplace=True)
    
    # COncat all back
    climber_df = pd.concat([numericals2,categoricals], axis=1)
    return climber_df


def reduce_levels(df, col, num_groups, label_names = None):
    '''
    Documentation:
    This function replace the number of possible values inside a dataframe column, to have tthe same amount of
    rows for each group.
    
    Input values: 
    df -> Pandas dataframe
    col -> The column of the dataframe to do the transformation
    num_groups -> Number of desired groups for the dataframe column 'col'
    label_names (optional)-> Labels of each group. If it's not provided, automated group labels will be generated.
    
    Output:
    The colum values with the new groups.
    '''
    df2 = df.copy()
    if (df2[col].nunique() < num_groups):
        return df2[col]
    else:
        labels = []
        if (label_names == None):
            for i in range(num_groups):
                labels.append('Group_'+ str(i+1))
            df2[col] = pd.qcut(df2[col], num_groups, labels=labels)
            return df2[col]
        else:
            df2[col] = pd.qcut(df2[col], num_groups, labels=label_names)
            return df2[col]
    
#reduce_levels(data, 'IC3', 6)
#reduce_levels(data, 'IC3', 4, label_names = ['Low','Mid','High','Very High'])
#reduce_levels(data, 'IC3', 4)
    

    
    