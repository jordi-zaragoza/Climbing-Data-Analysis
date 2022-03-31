import pandas as pd 
import numpy as np
from datetime import datetime

# -------------- CONVERSION TABLE ----------------------------------------------------------
def conversion_table():
    pd.set_option('display.max_rows', 10)
    conversion_df = pd.read_csv('../../databases/grade.csv')
    conversion_df = conversion_df[['id','fra_routes']]
    conversion_df.reset_index(inplace = True)
    return conversion_df

# We see that we have a problme here: the id is not consecutive so we cannot calculate metrics like mean..
# We will transform the "id" into the "index" for each grade_id on the ascent table with this function
def Correct_grade_id(df):
    def evaluate(x):
        if x < 48:
            x-1
        elif x < 61:
            x-2
        elif x < 74:
            x-3
        else:
            x-4
        return x
    df['grade_id'] = df['grade_id'].apply(lambda x: evaluate(x))
    return df

# -------------- USER TABLE ----------------------------------------------------------
def user_table():
    # Get the table
    user_df = pd.read_csv('../../databases/user.csv',low_memory=False)
    user_df = user_df.rename(columns={'id':'user_id'}) 
    return user_df

def clean_user(df):
    grad = df.copy()
    print("Before cleaning the table ")
    print(grad.shape)
    # 1. We have to clean first the NA numbers
    # If we fill the NA's of "" with the mode..
    grad['country'] = grad['country'].fillna(grad['country'].mode()[0])
 
    
    grad_clean = grad.drop(columns= ['first_name','last_name','competitions','occupation','sponsor1','sponsor2','sponsor3','best_area','worst_area','guide_area','interests','presentation','deactivated','anonymous','city'])
    
    # 2. Put the date in a number
    def Birth_fun(x):
        if ~pd.isna(x) & isinstance(x, str):
            x = float(x[0:4])
        else:
            x = float("NAN")  
        return x
    grad_clean['birth'] = grad_clean['birth'].apply(Birth_fun)

    # 3. Clean countries with not so many people climbing
    # group the countries in "others"
    countries = np.array(grad_clean['country'].value_counts().head(25).index)
    grad_clean['country'] = grad_clean['country'].apply(lambda x: "other" if (x not in countries) else x)
    grad_clean

    # 4. More Cleaning
    # Remove outliners of started
    grad_clean2 = grad_clean[grad_clean.started < 2017]
    grad_clean2 = grad_clean2[grad_clean2.started > 1987]
    # Remove outliners of heigh below 120 or above 210
    grad_clean2 = grad_clean2[grad_clean2.height > 135]
    grad_clean2 = grad_clean2[grad_clean2.height < 205]
    # Remove outliners of weight
    grad_clean2 = grad_clean2[grad_clean2.weight > 25]
    grad_clean2 = grad_clean2[grad_clean2.weight < 95]
    grad_clean2.reset_index(inplace = True, drop = True)
    
    # Replace columns
    grad_clean2['age'] = 2017 - grad_clean2['birth'] # The database is from 2017
    grad_clean2['years_cl'] = 2017 - grad_clean2['started']
    grad_clean2 = grad_clean2.drop(columns = ['birth','started'])
    
    grad_clean2 = grad_clean2[grad_clean2.age < 70]
    grad_clean2 = grad_clean2[grad_clean2.age > 11]    
    
    print("\n\nAfter cleaning the table ")
    print(grad_clean2.shape)

    return grad_clean2.set_index('user_id')

# -------------- ASCENT TABLE ----------------------------------------------------------
def ascent_table():
    ascent_df = pd.read_csv('../../databases/ascent.csv',low_memory=False)
    return ascent_df

def split_notes(x):
    first_ascent = 0
    second_go = 0 
    soft = 0
    hard = 0 
    traditional = 0
    one_hang = 0
    
    x_str = x.split(', ')
    if "Second Go" in x_str:
        second_go = 1
    if "Soft" in x_str:
        soft = 1        
    if "Hard Go" in x_str:
        hard = 1
    if "Traditional" in x_str:
        traditional = 1
    if "First Ascent" in x_str:
        first_ascent = 1
    if "One Hang " in x_str:
        one_hang = 1
        
    return first_ascent, second_go, soft, hard, traditional, one_hang


def clean_ascent(df):
    ascent_clean = df.copy()
    # We have to clean first the NA numbers
    print("Before cleaning the table ")
    print(ascent_clean.shape)
    # Climb type 0 is sportsclimbing
    ascent_clean = ascent_clean[ascent_clean['climb_type'] == 0]
    ascent_clean = ascent_clean.drop(columns=['climb_type','raw_notes','description','sector','crag','exclude_from_ranking','climb_try','repeat'])
    ascent_clean = ascent_clean.drop(columns = ['id', 'total_score', 'project_ascent_date','last_year', 'yellow_id', 'chipped'])
    
    # all to lowercase
    ascent_df3 = ascent_clean.copy()
    ascent_df3['name'] = ascent_df3.name.apply(lambda x: str(x).lower())
    ascent_df3['country'] = ascent_df3.country.apply(lambda x: str(x).lower())
    ascent_df3['comment'] = ascent_df3.comment.apply(lambda x: str(x).lower())
    
    # Grade below 9c
    print(ascent_df3.shape)
    ascent_df4 = ascent_df3.copy()
    ascent_df4 = ascent_df4[ascent_df4['grade_id'] < 79] # below 9c
    ascent_df4 = ascent_df4[ascent_df4['grade_id'] > 28] # # above 5a
    ascent_df4.comment = ascent_df4.comment.fillna('-')
    ascent_df4.dropna(subset=['name'],inplace = True)
    ascent_df4.reset_index(drop = True, inplace = True)
    print(ascent_df4.shape)

    # Filtering:
    ascent_df5 = ascent_df4.copy()
    ascent_df5 = ascent_df5["?" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["??" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["don't know name" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["???" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["¿?" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["unknown" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["no name" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["????" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["?????" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["??????" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["? no name" != ascent_df5["name"]]
    ascent_df5 = ascent_df5["senza nome" != ascent_df5["name"]]
    print(ascent_df5.shape)    
    # Be sure to clean this well..

    # filter no crag no sector
    ascent_df5 = ascent_df5[ascent_df5.crag_id != 0]
    ascent_df5 = ascent_df5[ascent_df5.sector_id != 0]

    ascent_df5.reset_index(drop = True, inplace = True)
        
    # We fill all the NA comments with -
    ascent_clean = ascent_df5.copy()
    ascent_clean.notes = ascent_clean.notes.fillna('-')
    ascent_clean.name = ascent_clean.name.fillna('-')
    ascent_clean.comment = ascent_clean.comment.fillna('-')
    ascent_clean.country = ascent_clean.country.fillna('-')
    ascent_clean.dropna(inplace = True)
    ascent_clean.reset_index(drop = True, inplace = True)
    
    # Split the notes column
    notes = ascent_clean.notes.apply(split_notes)
    notes_df = pd.DataFrame(notes.to_list(), columns = ['first_ascent', 'second_go', 'soft', 'hard', 'traditional', 'one_hang'])
    ascent_clean = ascent_clean.drop(columns = ['notes'])
    ascent_clean = pd.concat([ascent_clean,notes_df], axis = 1)
    
    # If a grade is considered hard then +1 if it is considered soft then -1 to the grade_id
    def Easy_hard(row):
        x = row.grade_id
        if (row.hard):
            return x+1 
        elif (row.soft):
            return x-1 
        else:
            return x
    ascent_clean.grade_id = ascent_clean.apply(Easy_hard, axis = 1)
    
    ascent_clean = ascent_clean[ascent_clean['traditional'] == 0]
    ascent_clean = ascent_clean.drop(columns = ['soft','hard','traditional','one_hang'])
    ascent_clean.dropna(inplace = True)
    ascent_clean.reset_index(drop = True, inplace = True)
    
    # Convert date into date format
    ascent_clean['rec_date'] = ascent_clean.rec_date.apply(lambda x:  datetime.utcfromtimestamp(int(x)))
    ascent_clean['date'] = ascent_clean.date.apply(lambda x:  datetime.utcfromtimestamp(int(x)))
    
    print("\n\nAfter cleaning the table ")
    print(ascent_clean.shape)
    return ascent_clean

# -------------- GRADES TABLE --------------------------------------------------

def grades_table(df):
    climber_grades = df.copy()
        
    climber_grades = climber_grades[['user_id','grade_id','date','year']]

    # We have to make the grade correction from the grades table because the grades should be consecutive.. 1,2,3..
    climber_grades_correct = Correct_grade_id(climber_grades)

    climber_grades_correct = climber_grades_correct.rename(columns={'grade_id':'grades'}) 
    climber_grades_met = climber_grades_correct.groupby(['user_id'])

    # Get the rest values from the table
    aggfunc_ ={'grades':['count','last','first','mean','max'],'date':['last','first'], 'year':['last','first']}
    climber_grades_met = climber_grades_correct.pivot_table(index= ['user_id'], values = ['grades','date','year'], aggfunc = aggfunc_ )
    climber_grades_met.columns = ['_'.join(col) for col in climber_grades_met.columns.values] # jewell of the crown :)

    climber_grades_tot = climber_grades_met
    # Highest 5 grades
#     largest5 = (pd.DataFrame(climber_grades_met["grades"]
#         .nlargest(5).reset_index(level=1, drop=True)))

#     largest5['index'] = largest5.groupby('user_id').cumcount()  # temporary index

#     largest5 = (largest5.set_index("index", append=True)['grades']
#         .unstack()
#         .add_prefix('grades_max_'))

#     largest5['5_mean'] = largest5.mean(axis=1)
#     largest5['grades_max'] = largest5.max(axis=1)
#     largest5 = largest5[['5_mean','grades_max']]
    
    # Concatenate all
#     climber_grades_tot = pd.concat([climber_grades_met,largest5], axis = 1)       
#     climber_grades_tot['5_mean'] = climber_grades_tot['5_mean'].apply(lambda x: round(x,2))

    climber_grades_tot.dropna(inplace = True)
    climber_grades_met.reset_index(inplace = True)
    climber_grades_met = climber_grades_met.set_index('user_id')
    
    return climber_grades_tot

# -------------- COUNTRY GRADES TABLE ----------------------------

def country_table(df, conversion):
    country_data = df.copy()
    aggfunc_ = {'grades_mean':"mean",'grades_max':"max",'height':'mean','weight':'mean','age':'mean'}
    country_data = country_data.pivot_table(index= ['country'], values = ['grades_max','grades_mean','height','weight','age'], aggfunc = aggfunc_)
    country_data.reset_index(inplace=True)
    country_data
    country_data['age'] = country_data.age.apply(lambda x: np.round(x))
    country_data['height'] = country_data.height.apply(lambda x: np.round(x))
    country_data['weight'] = country_data.weight.apply(lambda x: np.round(x))
    country_data['grades_max'] = country_data.grades_max.apply(lambda x: np.round(x))
    country_data['grades_mean'] = country_data.grades_mean.apply(lambda x: np.round(x))
    country_data['max_fra'] = country_data.grades_max.apply(lambda x: conversion.loc[round(x),'fra_routes'])
    country_data['mean_fra'] = country_data.grades_mean.apply(lambda x: conversion.loc[round(x),'fra_routes'])
    return country_data




def Similar_array(arr, similarity = 0.9, show = False):
    '''
    This functions replaces the string values that are similar by the first to be checked in a np.array
    
    input:
    arr -> np.array with the strings we want to compare
    similarity -> float of the similarity threshold
    
    output:
    The np.array with the similar values replaced
    '''
    
    from difflib import SequenceMatcher
    
    def similar(a, b):
        return SequenceMatcher(None, a, b, autojunk=True).ratio()
    
    str_arr = arr.copy()
    for name1 in str_arr:
        if not (any(char.isdigit() for char in name1)): # Only non numbered names
            for name2 in str_arr:
                if not (any(char.isdigit() for char in name2)): # Only non numbered names
                    sim = similar(name1,name2)
                    if (sim > similarity) & (sim != 1):
                        if (show == True):
                            print("Replacing all the values with",name2,"by", name1,". Similarity: ",sim)
                        str_arr[str_arr==name2] = name1      
    return str_arr   