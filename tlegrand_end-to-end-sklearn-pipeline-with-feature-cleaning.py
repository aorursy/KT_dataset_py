# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from kaggle.competitions import nflrush

env = nflrush.make_env()
raw = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv")
raw.columns
raw['is_run'] = raw.NflId == raw.NflIdRusher

raw_single = raw[raw.is_run==True]

raw_single.fillna(-999,inplace=True)
yards = raw_single.Yards

X = raw_single.drop(["Yards", "is_run"], axis=1)
yards.describe()
import seaborn as sns

sns.distplot(yards)
y = yards
categorical_features = ["Team", "Quarter", "Down", "PossessionTeam",

                       "FieldPosition", "OffenseFormation", "PlayDirection",

                       "PlayerCollegeName", "HomeTeamAbbr", "VisitorTeamAbbr",

                       "Stadium", "Location", "StadiumType", "Turf",

                       "GameWeather", "WindDirection"]

for c in categorical_features:

    print(f'{c}: {len(X[c].unique())}')
X['Stadium'].apply(lambda x: x.lower()).value_counts()
stadium_mapping = {

    "oakland alameda-county coliseum": "oakland-alameda county coliseum",

    "mercedes-benz dome": "mercedes-benz superdome",

    "twickenham": "twickenham stadium",

    "los angeles memorial coliesum": "los angeles memorial coliseum",

    "m & t bank stadium": "m&t bank stadium",

    "centurylink": "centurylink field",

    "paul brown stdium": "paul brown stadium",

    "firstenergystadium": "firstenergy stadium",

    "centuryfield": "centurylink field",

    "first energy stadium": "firstenergy stadium",

    "firstenergy": "firstenergy stadium",

    "m&t stadium": "m&t bank stadium",

    "broncos stadium at mile high": "sports authority field at mile high",

    "nrg": "nrg stadium",

    "metlife": "metlife stadium",

    "mercedes-benz stadium": "mercedes-benz superdome"

}
X['Location'].apply(lambda x: x.lower().replace(".", "")).value_counts()
location_mapping = {

    "chicago. il": "chicago, il",

    "jacksonville, florida": "jacksonville, fl",

    "london": "london, england",

    "los angeles, calif": "los angeles, ca",

    "jacksonville florida": "jacksonville, fl",

    "cleveland": "cleveland, oh",

    "miami gardens, fla": "miami gardens, fl",

    "baltimore, maryland": "baltimore, md",

    "kansas city,  mo": "kansas city, mo",

    "new orleans": "new orleans, la",

    "cleveland ohio": "cleveland, oh",

    "e rutherford, nj": "east rutherford, nj",

    "seattle": "seattle, wa",

    "cleveland,ohio": "cleveland, oh",

    "houston, texas": "houston, tx",

    "cleveland, ohio": "cleveland, oh",

    "charlotte, north carolina": "charlotte, nc",

    "detroit": "detroit, mi",

    "pittsburgh": "pittsburgh, pa",

    "cincinnati, ohio": "cincinnati, oh",

    "miami gardens, fla": "miami gardens, fl",

    "arlington, texas": "arlington, tx",

    "orchard park ny": "orchard park, ny",

    "indianapolis, ind": "indianapolis, in",

    "chicago il": "chicago, il",

    "mexico city": "mexico city, mexico"

}
X['StadiumType'].apply(lambda x: str(x).lower().replace(".", "")).value_counts()
stadium_type_mapping = {

    "outside": "outdoor",

    "outdor": "outdoor",

    "ourdoor": "outdoor",

    "outddors": "outdoor",

    "oudoor": "outdoor",

    "outdoors": "outdoor",

    "indoors": "indoor",

    "retractable roof": "dome",

    "retr. roof-closed": "dome, closed",

    "retr. roof - closed": "dome, closed",

    "domed, closed": "dome, closed",

    "closed dome": "dome, closed",

    "domed": "dome, closed",

    "indoor, roof closed": "dome, closed",

    "retr. roof closed": "dome, closed",

    "retr. roof-open": "dome, open",

    "bowl": "nan",

    "heinz field": "nan",

    "open": "dome, open",

    "dome": "dome, closed",

    "outdoor retr roof-open": "dome, open",

    "retr. roof - open": "dome, open",

    "indoor, open roof": "dome, open",

    "cloudy": "nan"

}
X['Turf'].apply(lambda x: str(x).lower()).value_counts()
turf_mapping = {

    "naturall grass": "natural grass",

    "natural": "natural grass",

    "artifical": "artificial",

    "fieldturf 360": "field turf",

    "fieldturf360": "field turf",

    "fieldturf": "field turf"

}
X['GameWeather'].apply(lambda x: str(x).lower()).value_counts()
game_weather_mapping = {

    "sunny, windy": "sunny",

    'cloudy, light snow accumulating 1-3"': "cloudy",

    "rain chance 40%": "cloudy",

    "showers": "rainy",

    "cloudy, chance of rain": "cloudy",

    "t: 51; h: 55; w: nw 10 mph": "nan",

    "cloudy with periods of rain, thunder possible. winds shifting to wnw, 10-20 mph.": "cloudy",

    "sunny and clear": "sunny",

    "sun & clouds": "sunny",

    "coudy": "cloudy",

    "sunny and cold": "sunny",

    "sunny skies": "sunny",

    "cloudy, 50% change of rain": "cloudy",

    "clear and cool": "clear",

    "partly clear": "clear",

    "partly cloudy": "cloudy",

    "rain likely, temps in low 40s.": "rainy",

    "cloudy and cold": "cloudy",

    "partly clouidy": "cloudy",

    "cloudy, fog started developing in 2nd quarter": "foggy",

    "sunny, highs to upper 80s": "sunny",

    "mostly sunny skies": "sunny",

    "scattered showers": "rainy",

    "cloudy, rain": "rainy",

    "clear and warm": "clear",

    "cold": "nan",

    "30% chance of rain": "rainy",

    "mostly coudy": "cloudy",

    "sunny and warm": "sunny",

    "rain shower": "rainy",

    "cloudy and cool": "cloudy",

    "clear and cold": "clear",

    "heavy lake effect snow": "snowy",

    "snow": "snowy",

    "clear and sunny": "sunny",

    "light rain": "rainy",

    "clear skies": "clear",

    "n/a indoor": "indoor",

    "indoors": "indoor",

    "partly sunny": "sunny",

    "mostly sunny": "sunny",

    "n/a (indoors)": "indoor",

    "controlled climate": "nan",

    "rain": "rainy",

    "mostly cloudy": "cloudy",

    "partly cloudy": "cloudy",

    "party cloudy": "cloudy"

}
X['WindDirection'].apply(lambda x: str(x).lower().replace("-", "")).value_counts()
wind_direction_mapping = {

    "from ese": "wnw",

    "east north east": "ene",

    "13": "nan",

    "south southwest": "ssw",

    "from sse": "nnw",

    "south southeast": "sse",

    "from wsw": "ene",

    "north/northwest": "nnw",

    "from nne": "ssw",

    "from ssw": "nne",

    "west northwest": "wnw",

    "east southeast": "ese",

    "north east": "ne",

    "1": "nan",

    "8": "nan",

    "westsouthwest": "wsw",

    "from w": "e",

    "southeast": "se",

    "from s": "n",

    "from sw": "ne",

    "southwest": "sw",

    "northwest": "nw",

    "northeast": "ne",

    "east": "e",

    "from nnw": "sse",

    "south": "s",

    "north": "n",

    "west": "w"

}
X["WindSpeed"].value_counts()
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion, Pipeline

import re

                                              

class CleaningTransformer( BaseEstimator, TransformerMixin ):

    def _cast_to_int(self, x):

        try:

            if "-" in x:

                x = np.mean(x.split("-"))

            return int(x)

        except:

            return 0

        

    def fit( self, X, y = None ):

        return self 

    

    def transform( self, X, y = None ):

        for c in X.columns.values:

            if c == "Stadium":

                X.loc[:, c] = X[c].apply(lambda x: x.lower()).apply(lambda x: stadium_mapping.get(x, x))

            if c == "Location":

                X.loc[:, c] = X[c].apply(lambda x: x.lower().replace(".", "")).apply(lambda x: location_mapping.get(x, x))

            if c == "StadiumType":

                X.loc[:, c] = X[c].apply(lambda x: str(x).lower()).apply(lambda x: stadium_type_mapping.get(x, x))

            if c == "Turf":

                X.loc[:, c] = X[c].apply(lambda x: str(x).lower()).apply(lambda x: turf_mapping.get(x, x))

            if c == "GameWeather":

                X.loc[:, c] = X[c].apply(lambda x: str(x).lower()).apply(lambda x: game_weather_mapping.get(x, x))

            if c == "WindDirection":

                X.loc[:, c] = X[c].apply(lambda x: str(x).lower().replace("-", "")).apply(lambda x: wind_direction_mapping.get(x, x))

            if c == "WindSpeed":

                X.loc[:, c] = X[c].apply(lambda x: self._cast_to_int(str(x).lower().replace("mpg", ""))).fillna(0)

        return X



class DateTransformer( BaseEstimator, TransformerMixin ):

    def fit( self, X, y = None ):

        return self 

    

    def transform( self, X, y = None ):

        for c in X.columns.values:

            X.loc[:, c] = pd.to_datetime(X[c])

            if c == "PlayerBirthDate":

                X[f'{c}_year'] =X[c].dt.year

            else:

                X[f'{c}_hour'] = X[c].dt.hour

                X[f'{c}_minute'] = X[c].dt.minute

                X[f'{c}_second'] = X[c].dt.second

            X = X.drop([c], axis=1)

        return X

    

class HeightTransformer( BaseEstimator, TransformerMixin ):

    def fit( self, X, y = None ):

        return self 

    

    def transform( self, X, y = None ):

        X.loc[:, "height"] = X.iloc[:,0].apply(lambda x: float(x.replace("-", ".")))

        return X.drop(["PlayerHeight"], axis=1)



class PersonnelTransformer( BaseEstimator, TransformerMixin ): 

    def __init__(self, personnels):

        self.personnels = personnels

        

    def fit( self, X, y = None ):

        return self

    

    def _find_match(self, pattern, string):

        m = re.match(f"\d {pattern}", string)

        if m is not None:

            return m.group()[0]

        return 0

    

    def transform( self, X, y = None ):

        for c in ["OffensePersonnel", "DefensePersonnel"]:

            for p in self.personnels:

                X[f"{c}_{p}"] = X[c].apply(lambda x: self._find_match(p, x))

            X = X.drop([c], axis=1)

        return X
class CastTransformer( BaseEstimator, TransformerMixin ):

    def __init__(self, target):

        self.target = target

        

    def fit( self, X, y = None ):

        return self

    

    def transform( self, X, y = None ):

        if self.target == "str":

            X = pd.DataFrame(X).astype(str)

        if self.target == "int":

            X = pd.DataFrame(X).astype(int)

        return X
def yards_to_cdf(yards):

    return np.array([1 if i > yards + 99 else 0 for i in range(200)])
ignored_features = ["GameId", "PlayId", "NflId", "NflIdRusher", "DisplayName"]

cleaning_features = ["Stadium", "Location", "StadiumType", "Turf", "WindDirection", "GameWeather", "WindSpeed"]

categorical_features = ["Team", "Quarter", "Down", "PossessionTeam",

                       "FieldPosition", "OffenseFormation", "PlayDirection",

                       "PlayerCollegeName", "HomeTeamAbbr", "VisitorTeamAbbr",

                       "Stadium", "Location", "StadiumType", "Turf",

                       "GameWeather", "WindDirection"]

numeric_features = ["X", "Y", "S", "A", "Dis", "Orientation", "Dir", "JerseyNumber", "Season",

                    "YardLine", "Distance", "HomeScoreBeforePlay", "VisitorScoreBeforePlay",

                   "DefendersInTheBox", "PlayerWeight", "Week", "Temperature",

                   "Humidity", "WindSpeed"]

date_features = ["GameClock", "TimeHandoff", "TimeSnap", "PlayerBirthDate"]

personnel_features = ["OffensePersonnel", "DefensePersonnel"]

height_feature = ["PlayerHeight"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.drop(ignored_features, axis=1), y, test_size=0.3)
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

personnels = ["RB", "TE", "WR", "OL", "DL", "LB", "DB", "QB"]

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('str_cast', CastTransformer("str")),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

date_transformer = Pipeline(steps=[('date', DateTransformer())])

height_transformer = Pipeline(steps=[('height', HeightTransformer())])

personnel_transformer = Pipeline(steps=[('personnel', PersonnelTransformer(personnels))])
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(

    transformers=[

        ('dat', date_transformer, date_features),

        ('per', personnel_transformer, personnel_features),

        ('hei', height_transformer, height_feature),

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])
from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import VarianceThreshold

models = [

    RandomForestRegressor(random_state=42,n_jobs=-1, criterion='mae'),

    ElasticNet(),

    LGBMRegressor(random_state=42,n_jobs=-1, learning_rate=0.005, importance_type = 'gain', metric='mae')

]

model_scores = []

for model in models:

    pipe = Pipeline(steps=[

                      ('cleaner', CleaningTransformer()),

                      ('preprocessor', preprocessor),

                      ('selector', VarianceThreshold()),

                      ('regressor', model)], verbose=True)

    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")

    model_scores.append(np.mean(scores))

    print(f"{model} mean: {np.mean(scores)} std: {np.std(scores)}")
grid_params = [

    {'bootstrap': [True, False],

     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

     'max_features': ['auto', 'sqrt'],

     'min_samples_leaf': [1, 2, 4],

     'min_samples_split': [2, 5, 10],

     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

    ,{"max_iter": [1, 5, 10],

      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],

      "l1_ratio": np.arange(0.0, 1.0, 0.1)}

    ,{'learning_rate': [0.005],

    'n_estimators': [40],

    'num_leaves': [6,8,12,16],

    'boosting_type' : ['gbdt'],

    'objective' : ['binary'],

    'random_state' : [501], 

    'colsample_bytree' : [0.65, 0.66],

    'subsample' : [0.7,0.75],

    'reg_alpha' : [1,1.2],

    'reg_lambda' : [1,1.2,1.4]}

]
from sklearn.model_selection import RandomizedSearchCV

best_score_index = np.argmax(model_scores)

best_model = models[best_score_index]

param_distributions = {f'search__regressor__{k}': v for k,v in grid_params[best_score_index].items()}

final_pipe = Pipeline(steps=[

                      ('cleaner', CleaningTransformer()),

                      ('preprocessor', preprocessor),

                      ('selector', VarianceThreshold()),

                      ('regressor', best_model)], verbose=True)

# search = RandomizedSearchCV(final_pipe, param_distributions=grid_params[best_score_index])

final_pipe.fit(X_train, y_train)

y_pred = pd.Series(final_pipe.predict(X_train))
import warnings

warnings.filterwarnings("ignore")
def generate_prediction(model, df, test_df):

    cols = df.columns

    # Pipelines require columns to be in the same order

    yards_predicted = pd.Series(model.predict(test_df[cols]))

    return np.vstack(yards_predicted.apply(lambda x: yards_to_cdf(x)))
for (test_df, sample_prediction_df) in env.iter_test():

    predictions = generate_prediction(final_pipe, X_train, test_df)

    env.predict(pd.DataFrame(data=predictions[:,1:], columns=sample_prediction_df.columns))
env.write_submission_file()