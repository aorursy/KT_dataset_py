import pandas as pd

import featuretools as ft
df_train = pd.read_csv('../input/X_train.csv', encoding='cp949', dtype={'goods_id': 'category'})

df_test = pd.read_csv('../input/X_test.csv', encoding='cp949', dtype={'goods_id': 'category'})

# Assign a unique ID for each transaction 

df = pd.concat([df_train, df_test]).reset_index(drop=True).reset_index().rename(columns={'index': 'trans_id'})

df.head()
# A dataframe to create a feature matrix for each customer 

cu = pd.DataFrame({'cust_id': df.cust_id.unique()})



# Specify a dictionary with all the entities

entities = {

    "cust": (cu, "cust_id"),

    "trans": (df, "trans_id")

}



# Specify how the entities are related

relationships = [

    ("cust", "cust_id", "trans", "cust_id")

]
derived_features, _ = ft.dfs(entities=entities, relationships=relationships, target_entity="cust")
derived_features.info()
# One-hot encoding for categorical features

derived_features.drop(['MODE(trans.goods_id)'], axis=1, inplace=True)

derived_features = pd.get_dummies(derived_features).reset_index()



# Fill NA

derived_features.fillna(0, inplace=True)



# Split Data

X_train = pd.DataFrame({'cust_id': df_train.cust_id.unique()})

X_train = pd.merge(X_train, derived_features, how='left')



X_test = pd.DataFrame({'cust_id': df_test.cust_id.unique()})

X_test = pd.merge(X_test, derived_features, how='left')



# Remove unnecessary features

IDtest = X_test.cust_id;

X_train.drop(['cust_id'], axis=1, inplace=True)

X_test.drop(['cust_id'], axis=1, inplace=True)

y_train = pd.read_csv('../input/y_train.csv').gender
# Learn XGB

from xgboost import XGBClassifier

import sys, warnings

if not sys.warnoptions: warnings.simplefilter("ignore")



model = XGBClassifier(random_state=0, n_jobs=-1)

model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:,1]

fname = 'submission.csv'

submissions = pd.concat([IDtest, pd.Series(pred, name="gender")] ,axis=1)

submissions.to_csv(fname, index=False)

print("'{}' is ready to submit." .format(fname))