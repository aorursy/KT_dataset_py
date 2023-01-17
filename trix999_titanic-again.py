#this is a library very useful (speeddml.com) helping to make faster analysis

from speedml import Speedml



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





################################################## Custom functions ###################



def plot_histogram(df, row,col,n_bins):

    g = sns.FacetGrid(df, col=col)

    g.map(plt.hist, row, bins=n_bins)



    

def box_plot(data):

    sns.boxplot(data=data)

    plt.xlabel("Attribute Index")

    plt.ylabel(("Quartile Ranges - Normalized "))

    

    

def split_and_train(X,y, test_size, classifier):

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                        test_size = test_size, random_state = 0)

    y_train = y_train.astype(int)

    y_test = y_test.astype(int)

    classifier.fit(X_train, y_train)

    y_test_pred = classifier.predict(X_test)

    return y_test_pred, y_test, X_test, X_train, y_train    



def roc_graph(y_true, y_pred):

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)

    lw = 2

    plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    plt.figure()



def evaluate_classifier(y_true, y_pred, target_names):

    from sklearn.metrics import confusion_matrix    

    # Making the Confusion Matrix 

    print('Confusion Matrix')

    cm = confusion_matrix(y_true, y_pred)

    import seaborn as sn

    sn.heatmap(cm, annot=True)

    plt.figure()

    print(cm)

    print('\n')

    # Report

    print('Report')

    from sklearn.metrics import classification_report

    print(classification_report(y_true, y_pred, target_names = target_names))

    roc_graph(y_true, y_pred)

    

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    from sklearn.model_selection import learning_curve

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt     


# using speedml too

sml = Speedml('../input/train.csv', 

              '../input/test.csv', 

              target = 'Survived',

              uid = 'PassengerId')



#let define a starting point to track how the eda evolves

starting_point = sml.eda()



# which nulls

sml.train.isnull().sum()

sml.test.isnull().sum()

# visualize numerical

sml.plot.correlate()
# let's dig through all categorical features vs target 



sml.train[['Pclass', 'Survived']].groupby(['Pclass'], 

             as_index=False).mean().sort_values(by='Survived', ascending=False)



sml.train[['Sex', 'Survived']].groupby(['Sex'], 

             as_index=False).mean().sort_values(by='Survived', ascending=False)



sml.train[['Embarked', 'Survived']].groupby(['Embarked'], 

             as_index=False).mean().sort_values(by='Survived', ascending=False)
# let's dig through all discrete numerical features vs target 



sml.train[['Parch', 'Survived']].groupby(['Parch'], 

             as_index=False).mean().sort_values(by='Survived', ascending=False)



sml.train[['SibSp', 'Survived']].groupby(['SibSp'], 

             as_index=False).mean().sort_values(by='Survived', ascending=False)
# let's dig visually through all continuos numerical features vs target 



plot_histogram(sml.train, 'Age','Survived',20)



plot_histogram(sml.train, 'Fare','Survived',20)
# categorize sex

sml.feature.mapping('Sex', {'male': 1, 'female': 0})    



# impute numerical with median and categorical with most common value

sml.feature.impute()
sml.plot.continuous('Fare')



#remove outliers

sml.feature.outliers('Fare',upper=98)



sml.plot.continuous('Fare')



sml.eda()
sml.plot.continuous('SibSp')



#remove outliers

sml.feature.outliers('SibSp',upper=99)



sml.plot.continuous('SibSp')



sml.eda()
sml.feature.density('Age')

sml.feature.density('Fare')



sml.feature.drop(['Cabin','Ticket','Name'])



#freq_port = sml.train.Embarked.dropna().mode()[0]





sml.feature.labels(['Embarked'])



sml.eda()
def is_alone(df):

    if (df['Parch'] + df['SibSp'] > 0):

        return 0

    else:

        return 1

        

sml.train['Is_Alone'] = sml.train.apply(is_alone, axis=1)

sml.test['Is_Alone'] = sml.test.apply(is_alone, axis=1)



sml.feature.drop(['Parch', 'SibSp'])



sml.eda()
sml.plot.correlate()



sml.plot.distribute()
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)



sml.model.data()





y_test_pred, y_test, X_test, X_train, y_train = split_and_train(X = sml.train_X,y = sml.train_y, 

                                                                test_size = 0.30, classifier = classifier)



evaluate_classifier(y_test, y_test_pred, target_names = ['Not Survived', 'Survived'])
#coeff_df = pd.DataFrame(sml.train.columns.delete(7))

#coeff_df.columns = ['Feature']

#coeff_df["Correlation"] = pd.Series(classifier.coef_[0])

#coeff_df.sort_values(by='Correlation', ascending=False)


select_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

fixed_params = {'learning_rate': 0.1, 'subsample': 0.8, 

                'colsample_bytree': 0.8, 'seed':0, 

                'objective': 'binary:logistic'}



sml.xgb.hyper(select_params, fixed_params)
select_params = {'learning_rate': [0.3, 0.1, 0.01], 'subsample': [0.7,0.8,0.9]}

fixed_params = {'max_depth': 7, 'min_child_weight': 5, 

                'colsample_bytree': 0.8, 'seed':0, 

                'objective': 'binary:logistic'}



sml.xgb.hyper(select_params, fixed_params)


tuned_params = {'learning_rate': 0.1, 'subsample': 0.8, 

                'max_depth': 7, 'min_child_weight': 5,

                'seed':0, 'colsample_bytree': 0.8, 

                'objective': 'binary:logistic'}

sml.xgb.cv(tuned_params)



sml.xgb.cv_results.tail(5)

tuned_params['n_estimators'] = sml.xgb.cv_results.shape[0] - 1

sml.xgb.params(tuned_params)



sml.xgb.classifier()

sml.model.evaluate()

sml.plot.model_ranks()
sml.xgb.fit()

sml.xgb.predict()



sml.xgb.feature_selection()



sml.xgb.sample_accuracy()
from xgboost import XGBClassifier

classifier = XGBClassifier(learning_rate = 0.1, subsample = 0.8, 

                max_depth=7, min_child_weight=5,

                seed=0, colsample_bytree= 0.8, 

                objective='binary:logistic', n_estimators = 50)



y_test_pred, y_test, X_test, X_train, y_train = split_and_train(sml.train_X, sml.train_y, 

                                                                0.25, classifier)





evaluate_classifier(y_test, y_test_pred, target_names = ['Not Survived', 'Survived'])

plot_learning_curve(classifier, "Training", sml.train_X, sml.train_y.astype(int), (0.7, 1.01), cv=10)
#let's submit

data_test = pd.read_csv('../input/test.csv')

ids = data_test.iloc[:,0]



predictions = classifier.predict(sml.test_X)



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions2.csv', index = False)

output.head()


