'''
Classifier creation for top 10. Similar, code for top 5 & top 3.
'''
# Importing the classifier
from sklearn.ensemble import RandomForestClassifier

# For downsampling
from sklearn.utils import resample,shuffle

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Reading data 
# Set parameter nrows = anyNumber to extract that many records/rows from the original uncompressed data
# for work efficiency
# Parser Engine aka engine parameter is set to python due to some error it solves, but its optional 
# and the default value is just fine as long as it works   
data = pd.read_csv("../input/aggregate/agg_match_stats_0.csv", nrows = 500000, engine = 'python')  

# Extracting needed data
value_list = list(range(1,11)) # Set range as per requirement
df1 = data[data.team_placement.isin(value_list)].copy()
df2 = data[~data.team_placement.isin(value_list)].copy()
df1['team_placement'] = 10
df2['team_placement'] = -10
frames = [df1,df2]
df = pd.concat(frames)
df.drop(['date','match_id','match_mode','game_size','player_survive_time','player_assists','player_kills','player_dbno','player_name','team_id'], axis = 1, inplace = True)

# Checking for missing values
# print( df.isnull().values.any() )

# Checking for imbalances in cases for each class
# print( df.team_placement.value_counts() )

# Resampling imbalanced data
df_majority = df[df.team_placement==-10]
df_minority = df[df.team_placement!=-10]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=135167, # to match minority class
                                 random_state=42)  # reproducible results

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Randomly marking 70% rows for training
df_downsampled['is_train'] = np.random.uniform(0, 1, len(df_downsampled)) <= .70

# Setting team_placement as categorical
change = {"team_placement":     {-1: "!top 10", 1: "top 10"}}
df_downsampled.replace(change, inplace=True)
df_downsampled["team_placement"] = df_downsampled["team_placement"].astype('category')

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df_downsampled[df_downsampled['is_train']==True], df_downsampled[df_downsampled['is_train']==False]

train = shuffle(train)
test = shuffle(test)

features = train[train.columns[0:4]]
y = train['team_placement']
x_test = test[test.columns[0:4]]
y_test = test['team_placement']

# Training the model
clf = RandomForestClassifier(n_jobs=-1, oob_score = True, n_estimators = 100, random_state=42)
clf.fit(features, y)

# Accuracy Scores
print ('Internal Accuracy Score', clf.oob_score_)
print ('RF accuracy: TRAINING', clf.score(features,y))
print ('RF accuracy: TESTING', clf.score(x_test,y_test))

# Confusion Matrix
preds = clf.predict(x_test)
print( pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted']) )

# Feature Importance
feature_imp = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(features.columns.values, clf.feature_importances_):
    feature_imp[feature] = importance
print(feature_imp)