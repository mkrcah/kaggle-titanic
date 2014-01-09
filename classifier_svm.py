from __future__ import print_function
from __future__ import division

import warnings
import pandas as pd
import math
import numpy as np
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn import preprocessing
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report



#------------------------------------------------------------
# Utils
#------------------------------------------------------------

def entropy(labels):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for p in probs:
        ent -= p * math.log(p, n_classes)

    return ent


#------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------

def preprocess_data(df):

    # sex: replace male/female with 0/1
    df.Sex = df.Sex.apply(lambda x: 0 if x == 'male' else 1)

    # age: replace n/a with median age
    df.Age.fillna(df.Age.median(), inplace=True)

    df.Fare.fillna(df.Fare.median(), inplace=True)

    # embarked: replace with numerical values
    df['Embarked-num'] = df.Embarked.apply(lambda x: 0 if x == 'C' else (1 if x == 'Q' else 2))

    return df


CV_SET_SIZE = 0.7

inputData = preprocess_data(pd.read_csv('input-data/train.csv'))

X = inputData[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked-num']]
y = inputData['Survived']


#------------------------------------------------------------
# Exploration: Information gain of each attribute
#------------------------------------------------------------

forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)

importances = zip(X.columns.tolist(), forest.feature_importances_)
importances = pd.DataFrame(importances, columns=['Attribute', 'Importance'])
importances = importances.sort('Importance', ascending=False)
print('Attribute importances')
print(importances)


#------------------------------------------------------------
# Train SVM classifier
#------------------------------------------------------------

def trainSVM():
    # split to train and cv set
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=CV_SET_SIZE)

    print()
    print('Classifiers')

    param_grid = [
        {'C': [1, 10, 100], 'kernel': ['linear']},
    #    {'C': [1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
    ]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(C=1), param_grid, cv=5, scoring=score, n_jobs=4, verbose=5)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_cv, clf.predict(X_cv)
        print(classification_report(y_true, y_pred))
        print()


# -------------------------------------------------------
# Predicting on test data


inputData = preprocess_data(pd.read_csv('input-data/test.csv'))
X_test = inputData[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked-num']]

# SVM kernel=linear, C=100, 7 covariates
print('Train SVM on complete train data')
clf = svm.SVC(C=100, kernel='linear')
clf.fit(X, y)

print('Predict on test data')
y_test = clf.predict(X_test)
output = pd.DataFrame({'PassengerId': inputData.PassengerId, 'Survived': y_test})

print('Output predictions')
print(output.head())

output.to_csv('output/prediction_05_svm_linear_c100_covariates_7.csv', index=False)


