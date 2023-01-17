#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
%matplotlib inline

import os
print(os.listdir("../input"))
#Loading dataset
nba_df = pd.read_csv("../input/playoff_shots.csv")
nba_df.head()
#define correlation of statistics
corr = nba_df.corr()
#create heatmap
plt.subplots(figsize=(15,10))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#In order to better understand the factors around made shots, we need to better visualize the Data

nba_shot_halfcourt_df = nba_df.query('LOC_Y<400')
#Filter shot data within the halfcourt, anything over 400 is an outlier

sns.lmplot('LOC_X', # Horizontal coordinate of shot
           'LOC_Y', # Vertical coordinate of shot
           col="TEAM_NAME", col_wrap= 4, #Display plot by team
           data=nba_shot_halfcourt_df, # Data source
           fit_reg=False, # Don't fix a regression line
           hue='EVENT_TYPE', legend=True,
           scatter_kws={"s": 12}) # S marker size
#Drop non-numerical data fields that statistically irrelevant or covered in another column
nba_df.drop(['GRID_TYPE', 'PLAYER_NAME', 'TEAM_NAME', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 'HTM', 'VTM'], inplace=True, axis=1)
nba_df.head()
#check for missing values
print(pd.isnull(nba_df).sum())
#Quantify Shot_Type
shot_type_mapping = {'3PT Field Goal': 3, '2PT Field Goal': 2}
nba_df['SHOT_TYPE'] = nba_df['SHOT_TYPE'].map(shot_type_mapping)
nba_df['SHOT_TYPE'].head(5)
#Quantify Shot Zone Range
shot_zone_range_mapping = {'24+ ft.': 24, 'Less Than 8 ft.': 7, '16-24 ft.': 16, '8-16 ft.': 8, 'Back Court Shot': 50}
nba_df['SHOT_ZONE_RANGE'] = nba_df['SHOT_ZONE_RANGE'].map(shot_zone_range_mapping)
nba_df['SHOT_ZONE_RANGE'].head(5)
#Quantify Shot Zone Area
shot_zone_area_mapping = {'Back Court(BC)': 0, 'Left Side(L)': 1, 'Left Side Center(LC)': 2, 'Center(C)': 3, 'Right Side Center(RC)': 4, 'Right Side(R)': 5}
nba_df['SHOT_ZONE_AREA'] = nba_df['SHOT_ZONE_AREA'].map(shot_zone_area_mapping)
nba_df['SHOT_ZONE_AREA'].head(5)
#Quantify Shot Zone Basic
shot_zone_basic_mapping = {'Backcourt': 0, 'Left Corner 3': 1,'Right Corner 3': 2, 'Above the Break 3': 3, 'Mid-Range': 4, 'In The Paint (Non-RA)': 5, 'Restricted Area': 6}
nba_df['SHOT_ZONE_BASIC'] = nba_df['SHOT_ZONE_BASIC'].map(shot_zone_basic_mapping)
nba_df['SHOT_ZONE_BASIC'].head(5)
#Create dummy variable for shotype
shot_dummy = pd.get_dummies(nba_df['ACTION_TYPE'])
nba_df = pd.concat([nba_df,shot_dummy], axis = 1)
nba_df.drop(['ACTION_TYPE'], inplace=True, axis=1)
nba_df.head()
#Split data to predict if the shot was made or missed
X = nba_df.drop('EVENT_TYPE', axis = 1)
y = nba_df['EVENT_TYPE']

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
#Predict through Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=350)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
# Check results
print(classification_report(y_test, pred_rfc))