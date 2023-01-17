import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score



import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

df.head()
df['profitable'] = df.revenue > df.budget

df['profitable'] = df['profitable'].astype(int)



regression_target = 'revenue'

classification_target = 'profitable'



df['profitable'].value_counts()
df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(how="any")

df.shape
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']

outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]

plotting_variables = ['budget', 'popularity', regression_target]



axes = pd.plotting.scatter_matrix(df[plotting_variables], alpha=0.15, \

       color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))

plt.show()

print(df[outcomes_and_continuous_covariates].skew())
for covariate in ['budget', 'popularity', 'runtime', 'vote_count', 'revenue']:

    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))

print(df[outcomes_and_continuous_covariates].skew())
df.to_csv("movies_clean.csv")
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('movies_clean.csv')
# Define all covariates and outcomes from `df`.

regression_target = 'revenue'

classification_target = 'profitable'

all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']



regression_outcome = df[regression_target]

classification_outcome = df[classification_target]

covariates = df[all_covariates]



# Instantiate all regression models and classifiers.

linear_regression = LinearRegression()

logistic_regression = LogisticRegression()

forest_regression = RandomForestRegressor(max_depth=4, random_state=0)

forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):

    predictions = estimator.fit(X, y).predict(X)

    return r2_score(y, predictions)

    

def accuracy(estimator, X, y):

    predictions = estimator.fit(X, y).predict(X)

    return accuracy_score(y, predictions)
from sklearn.model_selection import cross_val_score

linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)

forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)



# Plot Results

plt.axes().set_aspect('equal', 'box')

plt.scatter(linear_regression_scores, forest_regression_scores)

plt.plot((0, 1), (0, 1), 'k-')



plt.xlim(0, 1)

plt.ylim(0, 1)

plt.xlabel("Linear Regression Score")

plt.ylabel("Forest Regression Score")



# Show the plot.

plt.show()
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)

forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)



# Plot Results

plt.axes().set_aspect('equal', 'box')

plt.scatter(logistic_regression_scores, forest_classification_scores)

plt.plot((0, 1), (0, 1), 'k-')



plt.xlim(0, 1)

plt.ylim(0, 1)

plt.xlabel("Linear Classification Score")

plt.ylabel("Forest Classification Score")



# Show the plot.

plt.show()
positive_revenue_df = df[df["revenue"] > 0]



regression_outcome = positive_revenue_df[regression_target]

classification_outcome = positive_revenue_df[classification_target]

covariates = positive_revenue_df[all_covariates]



linear_regression = LinearRegression()

logistic_regression = LogisticRegression()

forest_regression = RandomForestRegressor(max_depth=4, random_state=0)

forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)

linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)

forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)

logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)

forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)



np.mean(forest_regression_scores)
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)

forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)



plt.axes().set_aspect('equal', 'box')

plt.scatter(logistic_regression_scores, forest_classification_scores)

plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.xlabel("Linear Regression Score")

plt.ylabel("Forest Regression Score")



plt.show();



forest_classifier.fit(positive_revenue_df[all_covariates], positive_revenue_df[classification_target])

sorted(list(zip(all_covariates, forest_classifier.feature_importances_)), key=lambda tup: tup[1])
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)

forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome,cv=10, scoring=accuracy)



# Plot Results

plt.axes().set_aspect('equal', 'box')

plt.scatter(logistic_regression_scores, forest_classification_scores)

plt.plot((0, 1), (0, 1), 'k-')



plt.xlim(0, 1)

plt.ylim(0, 1)

plt.xlabel("Linear Classification Score")

plt.ylabel("Forest Classification Score")



# Show the plot.

plt.show()

# Print the importance of each covariate in the random forest classification.

forest_classifier.fit(positive_revenue_df[all_covariates], classification_outcome)

for row in zip(all_covariates, forest_classifier.feature_importances_,):

        print(row)