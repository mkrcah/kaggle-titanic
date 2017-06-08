
from sklearn import tree, linear_model
from sklearn import ensemble
from sklearn import cross_validation
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('input-data/train.csv')

# -------------------------------------------------------
# Preprocessing

# male/female to 0/1
#df.Sex = df.Sex.apply(lambda x: 0 if x=='male' else 1 )

# n/a age replace with average age
df.Age.fillna(df.Age.median(), inplace=True)


# -------------------------------------------------------
# Model selection

x = df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y = df.Survived

scores = pd.DataFrame(columns=['depth', 'train', 'cv'])

MAX_DEPTH = 10
EXPERIMENTS_PER_DEPTH = 10
CV_SET_SIZE = 0.7

for max_depth in range(1,MAX_DEPTH+1):

    #print "Max-depth ", max_depth
    for i in range(1,EXPERIMENTS_PER_DEPTH+1):

        # split to train and cv set
        train_x, cv_x, train_y, cv_y = cross_validation.train_test_split(x,y, test_size=CV_SET_SIZE)

        # train
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(x,y)

        # save result scores
        scores = scores.append(pd.DataFrame([dict(
            depth=max_depth,
            train=clf.score(train_x,train_y),
            cv=clf.score(cv_x,cv_y)
            )]), ignore_index=True)

        print scores
        print scores.groupby("depth").mean().sort("cv", ascending=False).cv.head()

        BEST_MAX_DEPTH = 4
        print "--------------------------"
        print "Best depth ", BEST_MAX_DEPTH
        print "--------------------------"

        scores.boxplot(column=['train','cv'], by=['depth'])
        plt.show()

exit()

# -------------------------------------------------------
# Model training

clf = tree.DecisionTreeClassifier(max_depth=BEST_MAX_DEPTH)
clf.fit(x,y)  # train on full training data



# -------------------------------------------------------
# Predicting on test data

td = pd.read_csv('input-data/test.csv')
td.Sex = td.Sex.apply(lambda x: 0 if x=='male' else 1 )
td.Age.fillna(td.Age.median(), inplace=True)
td.Fare.fillna(td.Fare.median(), inplace=True)

x = td[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y = clf.predict(x)

output = pd.DataFrame({'PassengerId' : td.PassengerId, 'Survived' : y})
print output.head()

output.to_csv('output/prediction_04_dt_crossvalidation.csv', index=False)

