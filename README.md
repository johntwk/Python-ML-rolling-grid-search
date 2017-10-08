# Machine Learning: Grid Search Hyperparameter Tuning and Rolling Forecast

## Description <a name="Description"></a>
This Python function uses machine learning modelling object from scikit-learn to implement a design of grid search hyperparameter selection that respects temporal ordering of time-series, and forecast time-series using the sliding (rolling)-window strategy.

## Table of Contents
1. [Description](#Description)
2. [Installation](#Installation)
3. [Methodology](#Methodology)
4. [Usage](#Usage)
5. [Future Development](#FutureDevelopment)
6. [Files](#Files)
7. [Reference](#Reference)
8. [License](#License)

## Installation<a name="Installation"></a>
You can place the file __rolling_grid_search.py__ in the same directory of your Python file and call the function __rolling_grid_search_ML__ by

```Python
from rolling_grid_search import rolling_grid_search_ML
```


## Methodology<a name="Methodology"></a>
1. Divide the time-series into a group of size **group_size**. Since the number of data points may not be divisible by **group_size**, the remaining part at the end of the time-series form another group.

   
2. Within each group, select the first **size_hyper_sel** observations for tuning hyperparameters. 
   In case the number of remaining observations from the last group is smaller than or equal to _size_hyper_sel_, 
   the hyperparameters for this group will be the same as the previous group.
   Otherwise, the function will tune hyperparameters for this group in the same way as other groups. 


3. Run one-step ahead forecast with window size equal to **window_size** using hyperparameters obtained from 2.


4. Compute the score using **scoring** function.


## Usage<a name="Usage"></a>
### 1. Inputs

#### (1) Parameters of rolling_grid_search_ML

| Parameter      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model          | A scikit learn machine learning model object e.g. KNeighborsRegressor().                                                                                                                                                                                                                                                                                                                                                                                           |
| X              | Pandas dataframe of features.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| y              | Pandas dataframe of labels.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| param_grid     | A dictionary of hyper-parameters for tuning Keys of the dictionary are names of the hyperparameters in the object model;Values of the dictionary are lists of values for tuning. e.g. param_grid = {"n_neighbors": [1,2,3,4,5,6],#,"p":[1,2,3,4,5,6,7,8,9,10]}                                                                                                                                                                                                     |
| scoring        | A used-defined function to compute scores for grid search and forecasting. Input of the function (actual,pred):      (i) actual: a list of float actual values e.g. [1,2,3,4]    (ii) pred  : a list of float predicted values e.g. [1.1,0,2.5,3.7] Return of the function: the float score (e.g. 0.005)                                                                                                                                                           |
| crit           | A user-defined function to determine which set of hyperparameters is optimal. Input of the function (score_lst):     (i) score_lst: a list of float scores e.g. [0.05, 0.01, 0.007, 0.3]   (ii) Return of the function: a 2-tuple with the first element being the                                                index of the most desirable score and                                                the second element being the score the 2 elements are float |
| window_size    | An integer specifying the window size                                                                                                                                                                                                                                                                                                                                                                                                                              |
| size_hyper_sel | An integer specifying the size of samples for#,hyperparameter optimization                                                                                                                                                                                                                                                                                                                                                                                         |
                              
#### (2) User-defined Functions

##### **scoring** Function
To choose which set of hyperparameters to use, users need to construct a user-defined function and pass it to the function **rolling_grid_search_ML** by parameter **scoring**. The **scoring** function must have a list of actual values (**actual_lst**) and a list of predicted values (**pred_lst**) as inputs and return the score as a numerical value. For example, we pass the function **rmse** as **scoring**. 

```Python
def rmse(actual,pred):
    import numpy as np
    len_lst = len(actual)
    e_2 = []
    for i in range(0,len_lst):
        e_2.append((actual[i]-pred[i])**2)
    return (np.array(e_2).mean())**(0.5)
```


#### **crit** Function
To identify which set of hyperparameters is the best, users need to pass another user-defined function to **crit** of function **rolling_grid_search_ML**. The **crit** function must have a list of numerical values scores (**score_lst**) as inputs and return a 2-tuple. The 2-tuple has the first element as the index (starting from 0) of the optimal score in the list and the second element contains the optimal score.

```Python
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
```

### 2. Outputs
Information for tracking the implementation of the function and the score.

### 3. Returns
The function returns a named tuple with the following information

| Name   | Description                                     |
|--------|-------------------------------------------------|
| score  | the score of this forecast                      |
| params | a list of hyperparameters used in this forecast |
| actual | a list of actual exchange rates                 |
| pred   | a list of predcited values                      |

### Example (example.py)
Suppose that users have a data set called **all_data.csv** in the same directory as this Python file.

```Python
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
```

Then, users can produce a graph of parameters and a graph of actual and predicted values.
```Python
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
```

<div id="bg">
  <img src="Figure_1_.png" alt="">
</div>  

<div id="bg">
  <img src="Figure_2_.png" alt="">
</div>  

## Future Development<a name="FutureDevelopment"></a>
1. Support multi-step ahead forecast.
2. Support expanding forecast.

## Files<a name="Files"></a>
1. rolling_grid_search.py
2. example.py

## Reference<a name="Reference"></a>
1. Rizvi, S. A. A., Roberts, S. J., Osborne, M. A., & Nyikosa, F. (2017). A Novel Approach to Forecasting Financial Volatility with Gaussian Process Envelopes. arXiv preprint arXiv:1705.00891.

## License <a name="License"></a>

MIT License

Copyright (c) 2017 John Tsang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
