import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
import csv as csv
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import string
#Ticket, Name, Cabin, PassenderId

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if type(big_string) != str:
           return np.nan
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan

def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Mr', 'Master']:
        return 1
    elif title in ['Countess', 'Mme']:
        return 2
    elif title in ['Mlle', 'Ms', 'Mrs', 'Miss']:
        return 3
    elif title =='Dr':
        if x['Sex']=='Male':
            return 1
        else:
            return 2
    else:
        return title

train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)       

train_df['Title']=train_df['Name'].map(lambda x: substrings_in_string(x, title_list))
train_df['Title']=train_df.apply(replace_titles, axis=1)

test_df['Title']=test_df['Name'].map(lambda x: substrings_in_string(x, title_list))
test_df['Title']=test_df.apply(replace_titles, axis=1)


#cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
#train_df['Deck']=train_df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

#test_df['Deck']=test_df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train_df[(train_df['Gender'] == i) & (train_df['Pclass'] == j+1)]['Age'].dropna().median()
 
train_df['AgeFill'] = train_df['Age']
for i in range(0, 2):
    for j in range(0, 3):
        train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender == i) & (train_df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

#train_df['AgeIsNull'] = pd.isnull(train_df.Age).astype(int)

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']

train_df['Age*Class'] = train_df.AgeFill * train_df.Pclass

median_fare = np.zeros(3)
if len(train_df.Fare[train_df.Fare.isnull()]) > 0:
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = train_df[ train_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        train_df.loc[ (train_df.Fare.isnull()) & (train_df.Pclass == f+1 ), 'Fare'] = median_fare[f]


train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
train_df = train_df.drop(['Age'], axis=1)
train_df = train_df.dropna()
target = train_df['Survived'].values
train_df = train_df.drop(['Survived'], axis=1)

if len(test_df.Fare[test_df.Fare.isnull()]) > 0:
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]


test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(test_df.Embarked[test_df.Embarked.isnull()]) > 0:
    test_df.Embarked[test_df.Embarked.isnull()] = test_df.Embarked.dropna().mode().values
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

test_df['AgeFill'] = test_df['Age']
for i in range(0, 2):
    for j in range(0, 3):
        test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

#test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass

ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
test_df = test_df.drop(['Age'], axis=1)
print train_df.head()
print test_df.head()
print 'test_df'
print test_df.keys()
print 'train_df'
print train_df.keys()

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


clf = RandomForestClassifier()
param_grid = {"n_estimators": [10, 20, 40, 50, 100, 200],
              "max_depth": [3, None],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
grid_search.fit(train_data, target)

print 'Training...'
print grid_search.best_params_

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data, target )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
