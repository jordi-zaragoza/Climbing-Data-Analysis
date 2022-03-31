from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import norm
import statistics

def Years_comparation(climber_clean):
    '''
    This function compares the mean grade between men and women that had been climbing X years using Aspin-Welch  t-test method 
    inputs:
    climber_clean -> climbers dataframe
    output:
    A dataframe of the comparison
    '''
    Z_list = []
    y_list = []
    Zc_list = []
    Accept_list = []
    alfa = 0.05

    for i in range(1,23):
        climber_clean2 = climber_clean[climber_clean.years_cl > i]
        climber_men = climber_clean2[climber_clean2.sex == 0]
        climber_women = climber_clean2[climber_clean2.sex == 1]
        Z,df = Compare_means(climber_men.grades_mean,climber_women.grades_mean,show = False)
        Zc = round(norm.ppf(1-alfa/2),2)
        Z_list.append(Z)
        y_list.append(i)
        Zc_list.append(Zc)
        Accept_list.append(Z < Zc)

    years_comp = pd.DataFrame({"Z":Z_list,"Zc":Zc_list,"Accept":Accept_list,"years":y_list})    
    return years_comp


def Compare_means(x1,x2,alfa = 0.05, show = True):
    '''
    This function returns the result of a t student calculation for 2 means
    
    Inputs:
    x1 -> panda series
    x2 -> another panda series
    alfa -> significance level  
    show -> if true it plots the results
    
    Output:
    t -> is the t value of the comparison
    values_df -> the df with the data analyzed
    '''
    def Get_params(x):
        mean = round(np.mean(x),2)
        sdev = round(np.std(x),2)
        num = round(x.shape[0],2)
        return mean,sdev,num

    def Values_df(v_men,v_women):
        met = {'metrics':['mean','sdev','num'],'men':v_men,'women':v_women}
        return pd.DataFrame(met)

    def t_calculation(mean1,mean2,sdv1,sdv2,n1,n2):
        t = (mean1-mean2)/np.sqrt((sdv1**2/n1)+(sdv2**2/n2))
        return t

    values_df = Values_df(Get_params(x1),Get_params(x2))

    t = np.round(t_calculation(values_df.men[0],values_df.women[0],
                  values_df.men[1],values_df.women[1],
                  values_df.men[2],values_df.women[2]),2)
    
    Zc = round(norm.ppf(1-alfa/2),2)

    if (show == True):
        display(values_df)
            
        if (t > Zc):
            print("\nt: ",t," > Zc:",Zc)
            print("We reject the Null Hypotesis Ho -> There are differences between them")
        elif (t < Zc):
            print("\nt: ",t," < Zc:",Zc)
            print("We accept the Null Hypotesis Ho -> There are not differences between them")

        x_axis = np.arange(20, 90, 0.01)
        plt.plot(x_axis, norm.pdf(x_axis, values_df.men[0], values_df.men[1]))
        plt.plot(x_axis, norm.pdf(x_axis, values_df.women[0], values_df.women[1]))
        plt.xlabel('Grades')
        plt.legend('mw')
        plt.title('Mean Comparative')
        plt.show()

    return t,values_df


def MetricsResults (y_train, y_pred_train,y_test,y_pred_test):
    '''
    This python file returns a dF with all the metrics from the train test true and predicted values       
    '''
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def Metrics_df(R_train,R_test):
        met = {'metrics':['R2','MSE','RMSE','MAE','MAPE'],'Train':R_train,'Test':R_test}
        return pd.DataFrame(met)
    
    def Metrics(y_true, y_pred):
        R2 = round(r2_score(y_true, y_pred),2)
        MSE = round(mean_squared_error(y_true, y_pred, squared=True),2)
        RMSE = round(mean_squared_error(y_true, y_pred, squared=False),2)
        MAE = round(mean_absolute_error(y_true, y_pred),2)
        MAPE = round(mean_absolute_percentage_error(y_true, y_pred),2)
        return [R2,MSE,RMSE,MAE,MAPE]

    return Metrics_df(Metrics(y_train, y_pred_train),Metrics(y_test, y_pred_test))


def r2_train_test(X_normalized_train_age,X_normalized_test_age,y_train_age, y_test_age,k_max = 3,weights = 'uniform'):
    '''
    This function is used to check the R2 relation with the train and test set for the KNN model    
    '''
    
    
    r2_train = []
    r2_test = []
    n_neighbors = list(range(1,k_max))
    for k in n_neighbors:
        knn = KNeighborsRegressor(n_neighbors=k, weights = weights) 
        knn.fit(X_normalized_train_age,y_train_age)

        pred = knn.predict(X_normalized_train_age) 
        r2_train.append(r2_score(y_train_age, pred))

        pred = knn.predict(X_normalized_test_age) 
        r2_test.append(r2_score(y_test_age, pred))

    plt.scatter(n_neighbors,r2_train)
    plt.scatter(n_neighbors,r2_test)
    plt.xlabel("K number")
    plt.ylabel("R2")
    plt.legend(['train','test'])
    plt.show()