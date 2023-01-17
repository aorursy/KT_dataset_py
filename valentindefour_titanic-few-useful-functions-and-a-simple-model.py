import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn import feature_selection

from sklearn import model_selection



from sklearn.metrics import classification_report, accuracy_score



import matplotlib.pyplot as plt

import seaborn as sb



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def prepare_data():

    

    train = pd.read_csv('/kaggle/input/titanic/train.csv')

    test = pd.read_csv('/kaggle/input/titanic/test.csv')

    

    def format_df(df):

        

        df = df.join(pd.get_dummies(df['Sex'], prefix='is', drop_first = True))



        df = df.join(pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first = True))



        df['title_norm'] = df['Name'].map(lambda x: 'Mr.' if 'Mr.' in x

                                          else 'Mrs.' if 'Mrs.' in x

                                          else 'Miss' if 'Miss' in x

                                          else 'Master' if 'Master' in x

                                          else 'None')



        df = df.join(pd.get_dummies(df['title_norm'], prefix = 'is', drop_first = True))



        df['Age'].fillna(df['Age'].mean(), inplace = True)



        df['Fare_log'] = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)



        df['size_fam'] = df['SibSp'] + df['Parch']



        df['is_alone'] = df['size_fam'].map(lambda x: 1 if x>0 else 0)



        df['is_child'] = df['Age'].map(lambda x: 1 if x<8 else 0)



        df.drop(['Cabin','Ticket','Name','Sex','Embarked','title_norm','Fare'], axis = 1, inplace = True)



        df.set_index('PassengerId', inplace = True, drop = True)

        

        return df

    

    train = format_df(train)

    test = format_df(test)

    

    features = test.columns

    target = train['Survived']

    

    return train, test, features, target
def compare_classifiers():

    

    classifiers = [ensemble.GradientBoostingClassifier(),ensemble.RandomForestClassifier(),naive_bayes.GaussianNB(),neighbors.KNeighborsClassifier(),

                   svm.SVC(probability=True),tree.DecisionTreeClassifier(),XGBClassifier(),linear_model.LogisticRegression(max_iter = 500)]

    

    classifiers_comparison = pd.DataFrame()



    cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .4, train_size = .6, random_state = 42 )



    for classifier in classifiers:

        

        classifier.fit(X_train, Y_train)

        classifiers_comparison = classifiers_comparison.append({'classifier':classifier.__class__.__name__,

                                    'score_train': classifier.score(X_train, Y_train),

                                    'fit_time': model_selection.cross_validate(classifier, train[features], target, cv  = cv_split)['fit_time'].mean(),

                                    'score_time': model_selection.cross_validate(classifier, train[features], target, cv  = cv_split)['score_time'].mean(),

                                    'score_test': model_selection.cross_validate(classifier, train[features], target, cv  = cv_split)['test_score'].mean()},

                                   ignore_index = True)

    

    return classifiers, classifiers_comparison.sort_values(by='score_test', ascending = False)
def features_viz(clf):

    

    plt.figure(figsize=(20,7))

    

    clf.fit(X_train, Y_train)

    print(clf.__class__.__name__)

    print(clf.get_params())

    df = pd.DataFrame({'features':features, 'importance':clf.feature_importances_})

    df.sort_values(by = 'importance', ascending = False, inplace = True)



    sb.barplot(df.features, df.importance)
def test_viz_params(clf, params):

    

    n=1

    plt.figure(figsize=(20,20))

    

    for parameter in params:

        

        test_results = []

                    

        model = model_selection.GridSearchCV(estimator = clf,

                                             param_grid = {parameter : params[parameter]},

                                             cv = 3,

                                             n_jobs = -1,

                                             verbose = False,

                                             scoring = 'accuracy')

        

        model.fit(X_train, Y_train)

        

        print(model.best_params_)



        test_results = model.cv_results_['mean_test_score'].tolist()

                

        plt.subplot(int(len(params)/2)+1,2,n)

        sb.lineplot(params[parameter], test_results)

        plt.ylabel("Accuracy")

        plt.xlabel(parameter)

        

        n+=1
train, test, features, target = prepare_data()



X_train, X_test, Y_train, Y_test = train_test_split(train[features], target, random_state=42)
train.head()
classifiers_list, classifiers_dataframe = compare_classifiers()



classifiers_dataframe
features_viz(classifiers_list[0])
features_viz(classifiers_list[5])
params ={'max_depth' : np.linspace(1, 32, 32, endpoint=True),

             'max_features' : [x for x in range(len(features))],

             'min_samples_leaf' : np.linspace(0.1, 2, 20, endpoint=True),

             'min_samples_split' : np.linspace(0.1, 2, 20, endpoint=True),

            'n_estimators' : [100, 200, 300, 400, 500, 700, 1000, 2000]}



test_viz_params(ensemble.RandomForestClassifier(), params)