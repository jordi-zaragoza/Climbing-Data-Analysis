import seaborn as sns
import matplotlib.pyplot as plt

def results_drawer(y_pred_train,y_train,y_pred_test,y_test):
    '''
    This function plots the y_pred against the y_true for train and test sets
    '''
    pl = y_train.copy()
    pl['pred'] = y_pred_train
    pl2 = y_test.copy()
    pl2['pred'] = y_pred_test
    sns.set(rc={'figure.figsize':(15,9)})
    fig, ax = plt.subplots(2,2)
    sns.scatterplot(data=pl, x = 'pred', y = 'grades_max', ax=ax[0,0])
    sns.scatterplot(data=pl2, x = 'pred', y = 'grades_max', ax=ax[0,1])
    sns.histplot(data=pl.pred-pl.grades_max, ax=ax[1,0])
    sns.histplot(data=pl2.pred-pl2.grades_max, ax=ax[1,1])
    
def plot_evolution(df_plot):
    '''
    This function plots the evolution of the grade inserted in the df_plot
    '''
    df = df_plot.copy()
    df = df.sort_values('years_cl',ascending=False)
    sns.set(rc={'figure.figsize':(8,6)})
    sns.lineplot(x='years_future', y='grade_fra' ,data = df)