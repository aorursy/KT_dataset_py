from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

classificationModels = [
    {
        'name': "Gradient Boosting Classifier",
        'estimator': GradientBoostingClassifier(),
        'params': [
            {},
            {
                'n_estimators': [70, 100, 150, 240], 'min_samples_split': [2, 3, 4],
                'max_depth': [2, 3, 4], 'max_features': [None, 'auto']
            }
        ]
    },
    {
        'name': "Random Forest Classifier",
        'estimator':  RandomForestClassifier(),
        'params': [
            {},
            {
                'n_estimators': [70, 100, 120, 160], 'max_depth': [None, 5, 15, 30],
                'max_features': [None, 'auto']
            },
        ]
    },
    {
        'name': "Ada Boost Classifier",
        'estimator': AdaBoostClassifier(),
        'params': [
            {},
            {
                'n_estimators': [20, 50, 80, 100]
            }
        ]
    }
]

regressionModels = [
    {
        'name': "Gradient Boosting Regressor",
        'estimator': GradientBoostingRegressor(),
        'params': [
            {},
            {
                'n_estimators': [70, 100, 150, 240], 'min_samples_split': [2, 3, 4],
                'max_depth': [2, 3, 4], 'max_features': [None, 'auto'], 'loss':['ls', 'huber']
            }
        ]
    },
    {
        'name': "Random Forest Regressor",
        'estimator':  RandomForestRegressor(),
        'params': [
            {},
            {
                'n_estimators': [10, 70, 100, 120, 160], 'max_depth': [None, 5, 15, 30],
                'max_features': [None, 'auto'],
            },
        ]
    },
    {
        'name': "Ada Boost Regressor",
        'estimator': AdaBoostRegressor(),
        'params': [
            {},
            {
                'n_estimators': [20, 50, 80, 100]
            }
        ]
    }
]

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class __DataFrameSelectorByIdx(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_ids, acceptable_missing_threshold=0.35):
        self.attribute_ids = attribute_ids
        self.acceptable_missing_threshold = acceptable_missing_threshold

    def fit(self, X, y=None):
        # Exclude columns with number of missing values exceeding threshold
        allRows = len(X)
        missing = list(X.isna().sum())
        self.attribute_ids = list(filter(
            lambda idx: missing[idx]/allRows <= self.acceptable_missing_threshold, self.attribute_ids))

        return self

    def transform(self, X):
        return X.iloc[:, self.attribute_ids].values


def NumericalPipeline(numeric_indexes):
    return Pipeline([
        ('selector', __DataFrameSelectorByIdx(numeric_indexes)),
        ('imputer', SimpleImputer(strategy="median")),
        ('min_max_scaler', MinMaxScaler()),
        #     ('std_scaler', StandardScaler())
    ])


def CategoricalPipeline(categorical_indexes):
    return Pipeline([
        ('selector', __DataFrameSelectorByIdx(categorical_indexes)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder', OneHotEncoder(sparse=False))
    ])

import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion


class Transformer:
    def __init__(self):
        self.inputVariables = 0
        self.numeric_ids = []
        self.categorical_ids = []
        self.ready = False

    def fit_transform(self, X):
        self.inputVariables = len(X.columns)

        for idx, dtype in enumerate(X.dtypes):
            if pd.api.types.is_numeric_dtype(dtype):
                self.numeric_ids.append(idx)
            else:
                self.categorical_ids.append(idx)

        transformers = []

        if len(self.numeric_ids) > 0:
            transformers.append(
                ("numerical_pipeline", NumericalPipeline(self.numeric_ids)))

        if len(self.categorical_ids) > 0:
            transformers.append(
                ("categorical_pipeline", CategoricalPipeline(self.categorical_ids)))

        self.pipeline = FeatureUnion(transformer_list=transformers)

        self.ready = True
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        if(not self.ready):
            print("Cannot transform. You must call 'fit_transform' first")
            return None

        if (len(X.columns) != self.inputVariables):
            print("Column number must be equal in training and predicting process")
            return None

        return self.pipeline.transform(X)

    def getIds(self):
        return {
            'all': self.inputVariables,
            'categorical': self.categorical_ids,
            'numerical': self.numeric_ids
        }

from sklearn.model_selection import GridSearchCV


def generateModel(X, y, isClassification, verbose=True):
    if isClassification:
        models = classificationModels
        scoring = 'accuracy'
    else:
        models = regressionModels
        scoring = 'neg_root_mean_squared_error'

    finalModel = models[0]

    for model in models:
        if verbose:
            print("Testing: ", model['name'], end='')

        model['grid_search_result'] = GridSearchCV(model['estimator'], model['params'],
                                                   cv=5, scoring=scoring, n_jobs=-1)
        model['grid_search_result'].fit(X, y)

        if verbose:
            print(" {0:.4f}".format(model['grid_search_result'].best_score_))

        if model['grid_search_result'].best_score_ > finalModel['grid_search_result'].best_score_:
            finalModel = model

    if verbose:
        print("Chosen model: ", finalModel['name'],
              "{0:.4f}".format(finalModel['grid_search_result'].best_score_))

    return {
        'estimator': finalModel['grid_search_result'].best_estimator_,
        'name': finalModel['name'],
        'params': finalModel['grid_search_result'].best_params_,
    }

import pandas as pd
import joblib


class Machine:
    # Machine for machine learning purposes

    def __init__(self, schema=None):
        self.modelParams = None
        self.modelName = "There is no model yet"
        self.model = None
        self.transformer = None
        self.isClassifier = False
        self.isTrained = False
        if isinstance(schema, str):
            with open(schema, 'rb') as file:
                prev = joblib.load(file)
                self.modelParams = prev.modelParams
                self.modelName = prev.modelName
                self.model = prev.model
                self.transformer = prev.transformer
                self.isClassifier = prev.isClassifier
                self.isTrained = prev.isTrained

    def learn(self, dataset_file, header_in_csv=False, verbose=True):
        dataset = pd.read_csv(dataset_file, header=(
            0 if header_in_csv else None))

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        self.isClassifier = isinstance(y[0], str)

        self.transformer = Transformer()
        X_prep = self.transformer.fit_transform(X)

        modelData = generateModel(X_prep, y, self.isClassifier, verbose)

        self.modelName = modelData['name']
        self.model = modelData['estimator']
        self.modelParams = modelData['params']
        self.isTrained = True

    def predict(self, features_file, output_file="output.csv", header_in_csv=False):
        if not self.isTrained:
            print("Run learn function first")
            return

        X_pred = pd.read_csv(features_file, header=(
            0 if header_in_csv else None))

        X_pred_prepared = self.transformer.transform(X_pred)

        y_pred = self.model.predict(X_pred_prepared)

        y_pread_dataframe = pd.DataFrame(data=y_pred, columns=["results"])

        y_pread_dataframe.to_csv(output_file)
        print("Results saved to ", output_file)

    def learnAndPredict(self, train_set_file, prediction_features_file,
                        output_file="output.csv", headers_in_csvs=False,
                        verbose=True):
        self.learn(train_set_file, headers_in_csvs, verbose)
        self.predict(prediction_features_file, output_file, headers_in_csvs)

    # def predictOne(self):
    #     pass

    def saveMachine(self, output_file_name="machine.pkl"):
        with open(output_file_name, 'wb') as file:
            joblib.dump(self, file)

    def showParams(self):
        print(self.modelName)
        print(self.modelParams)
        print(self.model.feature_importances_)

# Create automl machine instance
machine = Machine()

# Train machine learning model
machine.learn('/kaggle/input/iris-uploaded/iris.csv')

# Predict the outcomes
machine.predict('/kaggle/input/iris-uploaded/iris-pred.csv', 'output.csv')

# Show parameters of the model
machine.showParams()

# Save Machine with trained model to "machine.pkl"
machine.saveMachine('machine.pkl')

# Create new machine based on the schema
machine2 = Machine('machine.pkl')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
# df = pd.read_csv('/kaggle/input/automodele/iris_test_data.csv')
df = pd.read_csv('/kaggle/input/iris-uploaded/iris.csv')
print(df)
