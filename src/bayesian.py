import pandas as pd
import numpy as np
from sklearn import ensemble

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import pipeline
from sklearn import preprocessing
from functools import partial
from skopt import space
from skopt import gp_minimize

def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params)) #dictionary of parameters
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x,y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest= x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)

        accuracies.append(fold_acc)
    
    return -1.0*np.mean(accuracies)  #-1 * mean because we have to minimize (for accuracy score) but for log loss we don't have to 




if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    X  = df.drop('price_range', axis=1).values
    y  = df.price_range.values
    #when you use scikit-optimize we have to create param space
    param_space = [
        space.Integer(3,15, name='max_depth'),
        space.Integer(100,600, name='n_estimators'),
        space.Categorical(["gini", "entropy"], name='criterion'),
        space.Real(0.01, 1,prior='uniform', name='max_features'),
    ]
    #should be in order as of param space
    param_names = [
        "max_depth",
        'n_estimators',
        'criterion',
        'max_features'

    ]
    optimization_function = partial(
        optimize,
        param_names=param_names,
        x=X,
        y=y
    
    )

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls = 15,
        n_random_starts=10,
        verbose=10
    )
    
    print("best params")
    print("****************************")
    print(
        dict(zip(
            param_names, result.x
        ))
    )
    print("****************************")




