# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
df.head()
df.columns
df.hotel.value_counts()
df.isnull().any()
df.isnull().sum()
nan_replacements = {"children:": 0.0,"country": "Unknown", "agent": 0, "company": 0}
df_cln = df.fillna(nan_replacements)


df_cln["meal"].replace("Undefined", "SC", inplace=True)


zero_guests = list(df_cln.loc[df_cln["adults"]
                   + df_cln["children"]
                   + df_cln["babies"]==0].index)
df_cln.drop(df_cln.index[zero_guests], inplace=True)
df_cln.shape
df = df_cln
sns.heatmap(df.corr())
sns.countplot(df.arrival_date_year)
arrival = df.groupby(['arrival_date_year','arrival_date_month'])[['hotel']].count()
arrival
average_children = round(df["children"].mean())
df.groupby(['arrival_date_year', 'arrival_date_month']).size().plot.bar(figsize=(15,5))
sns.countplot(df.hotel)
rate = df['is_canceled'].value_counts().tolist()
bars =  ['Not Cancel','Cancel']
y_pos = np.arange(len(bars))
plt.bar(y_pos,height=rate , width=0.3 ,color= ['red','blue'])
plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.title("Cancel Rate", fontdict=None, size = 'large')
plt.show()
plt.title('Cancellation')
plt.ylabel('Cancel_Sum')

df.groupby(['hotel','arrival_date_year'])['is_canceled'].sum().plot.bar(figsize=(10,5))
sns.countplot(df.stays_in_weekend_nights)
sns.countplot(df.stays_in_week_nights)
top_countries = df["country"].value_counts().nlargest(10).astype(int)
df.groupby(['arrival_date_month','arrival_date_year'])['children', 'babies'].sum().plot.bar(figsize=(15,5))
plt.figure(figsize=(25,10))
sns.barplot(x=top_countries.index, y=top_countries, data=df)
cancel_corr = df.corr()["is_canceled"]
cancel_corr.abs().sort_values(ascending=False)[1:]
df.groupby("is_canceled")["reservation_status"].value_counts()
#When implementing data engineering, we need to distinguish and classify the nature of features.
num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
                "stays_in_weekend_nights","stays_in_week_nights","adults","children",
                "babies","is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled","agent","company",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]

cat_features = ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

features = num_features + cat_features
X = df.drop(["is_canceled"], axis=1)[features]
y = df["is_canceled"]


num_transformer = SimpleImputer(strategy="constant")

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                               ("cat", cat_transformer, cat_features)])
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

base_models = [("DT_model", DecisionTreeClassifier(random_state=42)),
               ("RF_model", RandomForestClassifier(random_state=42,n_jobs=-1)),
               ("LR_model", LogisticRegression(random_state=42,n_jobs=-1)),
               ("XGB_model", XGBClassifier(random_state=42, n_jobs=-1))]
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score

kfolds = 4
split = KFold(n_splits=kfolds, shuffle=True, random_state=42)

for name, model in base_models:
    
    model_steps = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
    
    
    cv_results = cross_val_score(model_steps, 
                                 X, y, 
                                 cv=split,
                                 scoring="accuracy",
                                 n_jobs=-1)
   
    min_score = round(min(cv_results), 4)
    max_score = round(max(cv_results), 4)
    mean_score = round(np.mean(cv_results), 4)
    std_dev = round(np.std(cv_results), 4)
    print(f"{name} cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")

model_pipe.fit(X,y)


onehot_columns = list(model_pipe.named_steps['preprocessor'].
                      named_transformers_['cat'].
                      named_steps['onehot'].
                      get_feature_names(input_features=cat_features))


feat_imp_list = num_features + onehot_columns


feat_imp_df = eli5.formatters.as_dataframe.explain_weights_df(
    model_pipe.named_steps['model'],
    feature_names=feat_imp_list)
feat_imp_df.head(10)