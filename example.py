from rolling_grid_search import rolling_grid_search_ML
import pandas as pd
import numpy as np
from pandas import *
from numpy import *
from sklearn import svm
from sklearn.model_selection import TimeSeriesSplit,ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm


data = pd.read_csv("all_data.csv",skiprows=list(range(2449,2615)))
data.set_index("DATE", inplace = True)
var_lst = data.columns.tolist()
for var in var_lst:
    data[var] = data[var].astype(float)
var_lst = data.columns.tolist()
num_lags = 1
for var in var_lst:
    data[var] = data[var].astype(float)
    for lag in range(1,num_lags + 1):
        col_name = "L"+str(lag)+"."+str(var)
        data[col_name] = data[var].shift(-lag)
data.dropna(axis=0, how='any', inplace = True)

def rmse(actual,pred):
    import numpy as np
    len_lst = len(actual)
    e_2 = []
    for i in range(0,len_lst):
        e_2.append((actual[i]-pred[i])**2)
    return (np.array(e_2).mean())**(0.5)
def crit_min(score_lst):
    min_val = score_lst[0]
    min_index = 0
    counter = 0
    for score in score_lst:
        if (score < min_val):
            min_index = counter
            min_val = score
        counter += 1
    return (min_index,min_val)

knnreg = KNeighborsRegressor()
param_grid = {"n_neighbors": [1,2,3,4,5],"p":[1,2,3,4,5]}
#model, X, y, param_grid, cv, scoring, crit, window_size
r = rolling_grid_search_ML(model = knnreg, y = DataFrame(data["US_EU"]), X = data[["L1.US_EU","L1.US_UK"]],
                       group_size = 365, param_grid=param_grid, scoring = rmse, 
                       crit = crit_min, window_size = 7, size_hyper_sel = 30)

params_lst = r.params
actual_lst = r.actual
pred_lst = r.pred

import matplotlib.pyplot as plt

params_df = DataFrame.from_dict(data = params_lst)
params_fig = params_df.plot(kind='line', title="Change in Hyperparameters", grid=False)
params_fig.set_xlabel("Group Number")
params_fig.set_ylabel("Hyperparameters")
plt.show()

pred_df = pd.DataFrame(data=[pred_lst,actual_lst])
pred_df = pred_df.transpose()
pred_df.rename(index=str, columns={0: "Prediction", 1: "Actual"}, inplace = True)
pred_fig = pred_df.plot(kind='line', title="Actual and Predicted Values", grid=False)
pred_fig.set_xlabel("Time")
pred_fig.set_ylabel("Exchange Rates")
plt.show()
