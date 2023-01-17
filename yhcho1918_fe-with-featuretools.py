import pandas as pd
import featuretools as ft
df_train = pd.read_csv('../input/X_train.csv', encoding='cp949', dtype={'goodcd': 'category'})
df_test = pd.read_csv('../input/X_test.csv', encoding='cp949', dtype={'goodcd': 'category'})
# Assign a unique ID for each transaction 
df = pd.concat([df_train, df_test]).reset_index(drop=True).reset_index().rename(columns={'index': 'transid'})
df.head()
# A dataframe to create a feature matrix for each customer 
cu = pd.DataFrame({'custid': df.custid.unique()})

# Specify a dictionary with all the entities
entities = {
    "cust": (cu, "custid"),
    "trans": (df, "transid")
}

# Specify how the entities are related
relationships = [
    ("cust", "custid", "trans", "custid")
]
derived_features, _ = ft.dfs(entities=entities, relationships=relationships, target_entity="cust")
derived_features.info()
# One-hot encoding for categorical features
derived_features.drop(['MODE(trans.goodcd)'], axis=1, inplace=True)
derived_features = pd.get_dummies(derived_features).reset_index()

# Fill NA
derived_features.fillna(0, inplace=True)

# Split Data
X_train = pd.DataFrame({'custid': df_train.custid.unique()})
X_train = pd.merge(X_train, derived_features, how='left')

X_test = pd.DataFrame({'custid': df_test.custid.unique()})
X_test = pd.merge(X_test, derived_features, how='left')

# Remove unnecessary features
IDtest = X_test.custid;
X_train.drop(['custid'], axis=1, inplace=True)
X_test.drop(['custid'], axis=1, inplace=True)
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