import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.preprocessing import Imputer



cols_to_use = ['Distance', 'Landsize', 'BuildingArea']



def get_some_data():

    data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

    y = data.Price

    X = data[cols_to_use]

    my_imputer = Imputer()

    imputed_x = my_imputer.fit_transform(X)

    return imputed_x, y



X, y = get_some_data()

my_model = GradientBoostingRegressor()

my_model.fit(X,y)

my_plots = plot_partial_dependence(my_model, features=[0,1,2], X=X,

                                  feature_names=cols_to_use,

                                  grid_resolution = 100)
def get_some_data():

    cols_to_use = ['Distance','Landsize', 'BuildingArea','YearBuilt']

    data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

    y=data.Price

    X = data[cols_to_use]

    my_imputer = Imputer()

    imputed_X = my_imputer.fit_transform(X)

    return imputed_X, y
from sklearn.ensemble.partial_dependence import partial_dependence,plot_partial_dependence



# get data

X,y = get_some_data()



my_model = GradientBoostingRegressor()



my_model.fit(X,y)



my_plots = plot_partial_dependence(my_model, features=[0,3],

                                  X=X,

                                  feature_names= ['Distance','Landsize', 'BuildingArea','YearBuilt'],

                                  grid_resolution=50)



plots_data = partial_dependence(my_model, [0],

                               X=X, 

                               grid_resolution=10)

print(plots_data)

#it has price value in 1st array and  Distance value in second array
titanic_data = pd.read_csv('../input/titanic/train.csv')

titanic_y = titanic_data.Survived

clf = GradientBoostingClassifier()

titanic_X_colns = ['PassengerId', 'Age', 'Fare']

titanic_X = titanic_data[titanic_X_colns]

my_imputer = Imputer()

imputed_titanic_X = my_imputer.fit_transform(titanic_X)



clf.fit(imputed_titanic_X, titanic_y)

titanic_plots = plot_partial_dependence(clf, features=[1,2], 

                                        X=imputed_titanic_X,feature_names=titanic_X_colns, grid_resolution=20)