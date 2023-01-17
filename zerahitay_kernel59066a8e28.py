import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/kaggle/input/titanic/'
train = pd.read_csv(path + 'train.csv').set_index('PassengerId')
test = pd.read_csv(path + 'test.csv').set_index('PassengerId')
merged = pd.concat([train, test], axis=0)
print(merged.head())
import matplotlib.pyplot as plt
for df in [merged]: 
    df['Gender'] = [0 if x == 'female'  else 1 for x in df['Sex']]
    df['Attend'] = (2*df['SibSp']**1 + 1.5*df['Parch']**1 + 3*df['Gender']**1+3/28*np.abs(df['Age']-28) + df['Pclass']**1)
labels = train['Survived'].unique()
colors = {1: 'b', 0: 'r'}
handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in labels]

explore_train = merged[merged["Survived"].isnull()==False]
fig, axes = plt.subplots(ncols=int(len(explore_train[['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Attend']].columns)), figsize=(20,6))

for col, ax in zip(explore_train[['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Attend']], axes):
    if col == 'Survived': continue
    df2 = explore_train.groupby(['Survived', col])[col].count().unstack('Survived').fillna(0)
    df2[[0,1]].plot.bar(ax=ax, title=col, stacked=True, color=[colors[i] for i in explore_train['Survived']])
fig.delaxes(axes[0])
plt.legend(handles, labels, title="Survived")
plt.tight_layout()    
plt.show()
def cabin(df):
    cabin_only = df.copy()
    cabin_only["Cabin_Data"] = cabin_only["Cabin"].isnull().apply(lambda x: not x)
    cabin_only["Deck"] = cabin_only["Cabin"].str.slice(0,1)
    cabin_only["Room"] = cabin_only["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
    cabin_only['Embarked'].fillna(value = 'N', inplace = True)
    cabin_only["Survived"] = cabin_only["Survived"].fillna(-1)
    cabin_only["Deck"] = cabin_only["Deck"].fillna("N").astype(str)
    cabin_only["Age"] = cabin_only["Age"].fillna(-1)
    cabin_only["Fare"] = cabin_only["Fare"].fillna(-1)
    cabin_only["Room"] = cabin_only["Room"].fillna(-1).astype(int)
    cabin_only = cabin_only.drop(columns=['Name','Ticket','Cabin','Sex','Attend',"Cabin", "Cabin_Data"])
    return cabin_only

merged_cabin = cabin(merged)
from sklearn.preprocessing import OneHotEncoder

# creating instance of one-hot-encoder
merged_cabin = merged_cabin[:].reset_index()
encoder = OneHotEncoder(sparse=False)
enc_df  = pd.DataFrame(encoder.fit_transform(merged_cabin[['Deck','Embarked']]))
enc_df.columns = encoder.get_feature_names(['Deck','Embarked'])
merged_cabin.drop(['Deck','Embarked'] ,axis=1, inplace=True)
OH_merged = pd.concat([merged_cabin, enc_df], axis=1)
OH_merged.head(10)
# Returning back the missing values
for column in ['Survived','Room','Embarked_N','Embarked_C','Embarked_Q','Embarked_S',
               'Deck_A','Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G','Deck_N','Deck_T']:
    OH_merged[column] = pd.to_numeric(OH_merged[column], downcast='integer')
    
OH_merged = OH_merged.replace(-1, np.NaN)

OH_merged.info()
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

X = OH_merged.drop(['Survived'], axis=1)
#imputer = KNNImputer(n_neighbors=2, weights="distance")
imputer = SimpleImputer(missing_values = np.nan, strategy ='median')
X = imputer.fit_transform(X)

df3 = pd.DataFrame(data=X, columns=OH_merged.drop(['Survived'], axis=1).columns)
All_merged = pd.concat([OH_merged['Survived'], df3], axis=1)

# As a merged dataset of Train and Test are being processed altogether, making sure both were correctly processed
All_merged[885:895]
#Normalize data set to 0-to-1 range
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(0, 1))

def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(All_merged[col])),columns=[col])
    return df

#Scale both the training inputs and outputs
merged_scaled = scaleColumns(All_merged, ['Pclass','Age','Fare','SibSp','Parch','Room'])
merged_scaled[885:895]
processed_train, processed_test = merged_scaled[:891], merged_scaled[891:]
y_train = pd.DataFrame(processed_train.pop('Survived'))
y_test = pd.DataFrame(processed_test.pop('Survived'))
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
for i in [4,6,8]:
    X = processed_train
    encoded_Y = y_train
    # baseline model
    def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(21, input_dim=21, activation='relu'))
        model.add(Dense(meta[0], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    # evaluate model with standardized dataset
    meta = [25, 70, i, 40]

    estimator = KerasClassifier(build_fn=create_baseline, epochs=meta[1], batch_size=meta[2], verbose=0)

    kfold = StratifiedKFold(n_splits=meta[3], shuffle=True)
    results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
model = Sequential()
model.add(Dense(21, input_dim=21, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, encoded_Y, batch_size=4, epochs=70)

predictions = model.predict(processed_test)
submission = pd.read_csv(path + 'gender_submission.csv')
submission['Survived'] = predictions.round().astype('int')
submission[:]
submission.to_csv('submission.csv', index=False)
