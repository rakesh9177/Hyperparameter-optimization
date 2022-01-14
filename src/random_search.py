import pandas as pd
import numpy as np
from sklearn import ensemble

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import pipeline
from sklearn import preprocessing






if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    X  = df.drop('price_range', axis=1).values
    y  = df.price_range.values


    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)

    classifer = pipeline.Pipeline(
        [
            ("scaling",scl),
            ("pca", pca),
            ("rf", rf)
        ]
    )

    #parameter grid is dictionary which contains the paramteres which are supposed to be optimized
    #remember: "pca__" value of key and "n_components" is parameter , so we have to use below notation for pipeline
    param_grid = {
        "pca__n_components":np.arange(5,10),                  
        "rf__n_estimators": np.arange(100,1500,100),
        "rf__max_depth" : np.arange(1,20,1),
        "rf__criterion" :["gini","entropy"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifer,
        param_distributions=param_grid,
        scoring="accuracy",
        n_iter=10,
        verbose=10,
        n_jobs=-1,
        cv=5
    )

    model.fit(X,y)

    print("best score: ",model.best_score_)
    print("best parameters: ",model.best_estimator_.get_params())


#Note:  scoring function can be changed based on the problem statement. make_score from sklearn can be used to build custom scoring model
