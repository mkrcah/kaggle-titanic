from __future__ import print_function
from __future__ import division

import warnings
import pandas as pd
import math
import numpy as np
from sklearn import tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import preprocessing
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from pandas.tools.rplot import *
import sys



import pylab as pl
from sklearn.externals import joblib

# excellent plotting in pandas: http://nbviewer.ipython.org/urls/gist.github.com/fonnesbeck/5850463/raw/a29d9ffb863bfab09ff6c1fc853e1d5bf69fe3e4/3.+Plotting+and+Visualization.ipynb

# https://github.com/RahulShivkumar/Titanic-Kaggle/blob/master/titanic.py
# http://www.kaggle.com/c/titanic-gettingStarted/forums/t/6699/sharing-experiences-about-data-munging-and-classification-steps-with-python
# http://www.philippeadjiman.com/blog/2013/09/12/a-data-science-exploration-from-the-titanic-in-r/
# http://tfbarker.wordpress.com/2013/12/22/datamining/
# http://www.kaggle.com/c/titanic-gettingStarted/forums/t/5760/basic-machine-learning-course-at-edx-form-caltech
# http://triangleinequality.wordpress.com/2013/09/05/a-complete-guide-to-getting-0-79903-in-kaggles-titanic-competition-with-python/
# http://www.kaggle.com/c/titanic-gettingStarted/forums/t/5540/a-first-timer-s-journey-to-a-0-78947
# http://www.kaggle.com/c/titanic-gettingStarted/forums/t/4069/top-20-kagglers-i-am-wondering-what-they-are-doing
# SVM anova http://scikit-learn.org/stable/auto_examples/svm/plot_svm_anova.html#example-svm-plot-svm-anova-py
# TODO: bayesian network, GradientBoostingClassifier
# TODO: area under curve
# TODO: difference between Random Forrest and Boosted Trees



class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators):
        self.estimators = estimators


    def fit(self, X, y):
        """ Fit all classifiers """
        for clf in self.estimators.values():
            clf.fit(X, y)


    def predict(self, X):
        """ Ensemble prediction of classifiers """

        predictions = pd.DataFrame(
            columns=self.estimators.keys(),
            index=range(0, X.shape[0]))

        for name, clf in self.estimators.items():

            predictions[name] = clf.predict_proba(X)[:, 0]

        votes_sum = predictions.sum(axis=1) / len(self.estimators)

        return votes_sum.apply(lambda x: 0 if x >= 0.5 else 1)


def get_proxies_all_data():

    train = pd.read_csv('input-data/train.csv').drop('Survived', axis=1)
    test = pd.read_csv('input-data/test.csv')
    all = train.append(test)

    # aggreg_mean_age

    salutation = all.Name.apply(lambda x: x.split(',')[1].split(' ')[1])
    replace = {
        'Mlle.': 'Miss.',
        'Major.': 'Dr.',
        'Capt.': 'Dr.',
        'Sir.': 'Mr.',
        'the': 'Mr.',
        'Don.': 'Mr.',
        'Jonkheer.': 'Mr.',
        'Mme.': 'Miss.',
        'Lady.': 'Miss.',
        'Ms.': 'Miss.',
        'Dona.': 'Mrs.'
    }
    salutation = salutation.apply(lambda x: replace[x] if x in replace else x)
    all['Salutation'] = salutation
    #group_columns = ['Pclass', 'Sex', 'Salutation']
    group_columns = ['Salutation']
    aggreg_mean_age = pd.Series(all.groupby(group_columns).mean().Age, name='MeanAge')

    # number of people for one ticket
    ticket_size = all.groupby('Ticket').size()
    ticket_size.name = 'TicketSize'

    return aggreg_mean_age, ticket_size


def extract_features(df):

    X = df[['Sex']]

    #----------------------------------------------------------------
    # Feature engineering
    #----------------------------------------------------------------

    aggreg_mean_age, ticket_size = get_proxies_all_data()

    # Sex: 0/1
    X['Sex'] = df.Sex.apply(lambda x: 0 if x == 'male' else 1)

    # Salutation: dummies (Mr, Mrs,...) from name
    salutation = df.Name.apply(lambda x: x.split(',')[1].split(' ')[1])
    replace = {
        'Mlle.': 'Miss.',
        'Major.': 'Dr.',
        'Capt.': 'Dr.',
        'Sir.': 'Mr.',
        'the': 'Mr.',
        'Don.': 'Mr.',
        'Jonkheer.': 'Mr.',
        'Mme.': 'Miss.',
        'Lady.': 'Miss.',
        'Ms.': 'Miss.',
        'Dona.': 'Mrs.'
    }
    salutation = salutation.apply(lambda x: replace[x] if x in replace else x)
    df['Salutation'] = salutation
    X = X.join(pd.get_dummies(salutation, prefix='Salut'))

    # Age: n/a fill with median, then normalized
    X['Age'] = df.Age
    #group_columns = ['Pclass', 'Sex', 'Salutation']
    group_columns = ['Salutation']
    mean_age = df.join(aggreg_mean_age, on=group_columns).MeanAge
    X.Age[X.Age.isnull()] = mean_age
    X['Age'] = preprocessing.scale(X.Age)


    # SibSp: identity
    X['SibSp'] = df.SibSp

    # Parch: identity
    X['Parch'] = df.Parch

    # Embarked: dummies
    X = X.join(pd.get_dummies(df.Embarked, prefix='Emb'))

    # Fare: normalize, add indicator if zero fare
    X['Fare'] = df.Fare.fillna(12.415462)
    # one nan in test data for male in 3rd class, replaced with mean fare for male in 3rd class
    X['Fare'] = preprocessing.scale(np.power(X.Fare, 0.1))
    X['ZeroFare'] = df.Fare.apply(lambda x: 0 if x == 0 else 1)

    # Kid: 0/1 if age < 2
    X['Kid'] = (df.Age < 2).apply(lambda x: 0 if x else 1)

    # Alone: 0/1 if no siblings, parents, children
    X['Alone'] = (df.SibSp + df.Parch).clip_upper(1)

    # Pclass: to dummies
    X = X.join(pd.get_dummies(df.Pclass, prefix='Pclass'))

    # Floor: 1/0 if cabin is defined
    X['Cabin'] = df.Cabin.isnull().map(lambda x: 1 if x else 0)
    floor = df.Cabin.fillna('U').str.replace('[0-9 ]', '').apply(lambda x: x[-1])
    X = X.join(pd.get_dummies(floor, prefix='Floor'))

    if 'Floor_T' not in X.columns:
        X['Floor_T'] = 0    # absent in test data

    # Ticket: No. of people on one ticket
    X['TicketSize'] = df.join(ticket_size, on='Ticket').TicketSize.clip_upper(1)

    #----------------------------------------------------------------
    # Unused features
    #----------------------------------------------------------------

    #print(X.head(5))
    #X['Age'] = preprocessing.scale(df.Age.fillna(df.Age.median()))
    #df['Embarked-num'] = df.Embarked.apply(lambda x: 0 if x == 'C' else (1 if x == 'Q' else 2))
    #df['PriceGroup'] = df.Fare.apply(lambda x: 0 if x <= 10 else (1 if x <= 20 else 2))
    #df['PriceGroup'] = df.Fare.apply(lambda x: 0 if x <= 5 else 1 if x <= 15 else 2 if x <= 100 else 3)
    #df['PriceGroup'] = df.Fare.apply(lambda x: 0 if x <= 5 else 1 if x <= 15 else 2 if x <= 100 else 3)
    #X = inputData[['Pclass']]
    #X = inputData[['Sex', 'Age', 'Fare']]
    # extract floor from cabin - this seems too complicated

    return X



def letsgo():

    pd.set_printoptions(max_rows=200, max_columns=100)
    kFold = 4
    max_score_sofar = 0.836181


    #----------------------------------------------------------------
    # Training model
    #----------------------------------------------------------------

    print('Loading and preprocessing data')
    trainData = pd.read_csv('input-data/train.csv')
    X = extract_features(trainData)
    y = trainData['Survived']

    #print(X.head(10))

    C_grid = [0.1, 0.2, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1, 1.2, 1.4, 1.5, 2, 3, 4, 10]
    params = [
        #{
        #    'name': 'Linear SVM',
        #    'clf': svm.SVC(kernel='linear', probability=True),
        #    'param_grid': [{'C': C_grid}]
        #},
        #{
        #    'name': 'Radial SVM',
        #    'clf': svm.SVC(kernel='rbf', probability=True),
        #    'param_grid': [{'C': C_grid, 'gamma': [0.1, 0.01, 0.001, 0.0001]}]
        #},
        #{
        #    'name': 'Logistic regression',
        #    'clf': linear_model.LogisticRegression(),
        #    'param_grid': [{'C': C_grid}]
        #},
        {
            'name': 'Decision tree',
            'clf': DecisionTreeClassifier(),
            'param_grid': [{'max_depth': range(3, 20)}]
        },
        #{
        #    'name': 'Random forrest',
        #    'clf': RandomForestClassifier(),
        #    'param_grid': [{'n_estimators': [10, 50, 100, 200, 300], 'max_depth': [5, 10, 50, 100]}]
        #},
    ]

    print("--------------------------------------")
    clf = linear_model.LogisticRegression(C=0.4)
    scores = cross_validation.cross_val_score(clf, X, y, cv=kFold, n_jobs=4, verbose=0)
    print("Best classifier: %0.6f (+/- %0.3f), difference %0.6f" % (scores.mean(), scores.std() * 2, scores.mean() - max_score_sofar))


    print("--------------------------------------")
    best_estimators = {}
    for p in params:
        clf = GridSearchCV(p['clf'], p['param_grid'], cv=kFold, scoring='accuracy', n_jobs=4, verbose=0)
        clf.fit(X, y)
        print("%20s   %+0.6f   %0.6f %30s " % (p['name'], clf.best_score_ - max_score_sofar, clf.best_score_, clf.best_params_, ))
        best_estimators[p['name']] = clf.best_estimator_

    #if len(best_estimators) >= 1:
    #print("--------------------------------------")
    #ensemble = EnsembleClassifier(best_estimators)
    #scores = cross_validation.cross_val_score(ensemble, X, y, cv=kFold, verbose=0)
    #print("Ensemble of best estimators:: %0.6f (+/- %0.3f), improvement by %0.6f" % (scores.mean(), scores.std() * 2, scores.mean() - max_score_sofar))


    #clf_to_submission = ensemble
    clf_to_submission = linear_model.LogisticRegression(C=0.4)


    #----------------------------------------------------------------
    # Predict test data
    #----------------------------------------------------------------

    # train on whole train data
    clf_to_submission.fit(X, y)

    testData = pd.read_csv('input-data/test.csv')
    X_test = extract_features(testData)
    X_test = X_test[X.columns.tolist()]

    print('Predicting test data')
    y = clf_to_submission.predict(X_test)
    output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': y})

    print('Output predictions')
    print(output.head())

    filename = 'output/prediction_13.csv'
    output.to_csv(filename, index=False)
    print('Predictions saved to ')


#exploration()
letsgo()

#main()


#train_simple_svm()
#plot_clf()




