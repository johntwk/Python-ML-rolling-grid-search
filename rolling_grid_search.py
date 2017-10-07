def rolling_grid_search_ML (model, X, y, group_size, param_grid, scoring, crit, window_size, size_hyper_sel):
    # Output Header
    # Import necessary libraries
    from sklearn.model_selection import TimeSeriesSplit,ParameterGrid
    import pandas as pd
    import numpy as np    
    
    # Output Header and Information
    len_X = len(X)
    print "Rolling Grid Search for Machine Learning Models"
    print "By John Tsang"
    print "-------------------------------------------------\n"
    print "Model             :\n",model
    print "Length of Data Set:",len(X)
    print "Group size        :",group_size
    print "Labels (y)        :",y.columns.tolist()
    print "Features (X)      :",X.columns.tolist()
    print "Window Size       :",window_size
    print "Hyperparameter    :",size_hyper_sel
    print "Selection Size"
    
    # Group data 
    group_X_lst = []
    group_y_lst = []
    
    num_of_groups = len_X / group_size
    for group_num in range(1,num_of_groups+1):
        start = (group_num - 1) * group_size
        end   = (group_num  ) * group_size
        group_X_lst.append(X.iloc[start:end].copy())
        group_y_lst.append(y.iloc[start:end].copy())
    if (len_X % group_size != 0):
        start = num_of_groups * group_size
        end = len_X
        group_X_lst.append(X.iloc[start:end].copy())
        group_y_lst.append(y.iloc[start:end].copy())
        num_of_groups += 1
    print "Number of groups  :",num_of_groups,"\n"
    
    # Hyperparameter Selection for each group
    len_hyp_sel = size_hyper_sel
    def grid_rolling_hyperparam_sel(model,X,y,param_grid,window_size,scoring,crit):
        len_X = len(X)
        # Convert X and y to data structure that can be inputs for scikit learn models
        X = X.values
        y = y.values

        # Generate cv for one-step-ahead forecast
        tscv_rolling = TimeSeriesSplit(n_splits=len_X-window_size, max_train_size = window_size)
        
        # Initialize lists to store params and scores
        param_lst = []
        score_lst = []
        
        # Generate all combinations of hyperparameters for Grid Search
        param_grid = ParameterGrid(param_grid)
        
        # Grid Search using the first []
        for params in param_grid:
            actual_lst = []
            pred_lst   = []
            for train_index, test_index in tscv_rolling.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model = model.set_params(**params)
                model.fit(X_train,y_train)
                actual_lst.append(y_test)
                y_pred = model.predict(X_test)
                pred_lst.append(y_pred)
            score_lst.append(scoring(actual_lst, pred_lst))
            param_lst.append(params)
        # Find optimal param for this X,y pair
        rt_tuple = crit(score_lst)
        # Store
        rt = dict()
        rt["best_params"] = param_lst[rt_tuple[0]]
        rt["optimal_score"] = rt_tuple[1]
        return rt
    
    # Tune hyperparameters
    print "Hyperparameter Tuning...\n"
    counter = 0
    best_params_lst = []
    for group_X, group_y in zip(group_X_lst,group_y_lst):
        # Select samples for tuning
        group_X = group_X[0:len_hyp_sel].copy()
        group_y = group_y[0:len_hyp_sel].copy()
        # Hyperparameter tuning implementation
        rt = grid_rolling_hyperparam_sel(model = model, X = group_X, y = group_y,
                                         param_grid = param_grid, window_size = window_size,
                                         scoring = scoring, crit = crit)
        print "[",counter,"] Length:",len(group_X),rt
        best_params_lst.append(rt["best_params"])
        counter += 1

    # Rolling forecast
    print "\n"
    print "Rolling Forecast...\n"
    pred_lst = []
    actual_lst = []
    counter = 0
    for group_X, group_y, best_params in zip(group_X_lst,group_y_lst,best_params_lst):
        print "Working on group [",counter,"]"
        # Rolling Forecast using best_params
        model = model.set_params(**best_params)
        # Convert X and y to data structure that can be inputs for scikit learn models
        group_X = group_X.values
        group_y = group_y.values
        # Generate cv for one-step-ahead forecast
        len_X = len(group_X)
        tscv_rolling = TimeSeriesSplit(n_splits=len_X-window_size, max_train_size = window_size)
        for train_index, test_index in tscv_rolling.split(group_X):
            # Divide samples for training and testing
            X_train, X_test = group_X[train_index], group_X[test_index]
            y_train, y_test = group_y[train_index], group_y[test_index]
            # Fit the model with training data
            model.fit(X_train,y_train)
            # Store the actual value
            actual_lst.append(y_test[0][0])
            # Store predicted values
            y_pred = model.predict(X_test)
            pred_lst.append(y_pred[0][0])
        counter += 1
    # Compute scores for this forecast
    score = scoring(actual_lst, pred_lst)
    print "\nScore:",score
    # Pack results into a named tuple as return of the function
    # score:  the score of this forecast
    # params: list of hyperparameters used in this forecast
    # actual: list of actual exchange rates
    # pred  : list of predcited values
    import collections
    RollingGrid = collections.namedtuple('RollingGrid', ['score', 'params', 'actual', 'pred'])
    p = RollingGrid(score=score, params = best_params_lst, actual = actual_lst, pred = pred_lst)
    return p

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