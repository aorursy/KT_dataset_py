import pandas as pd

df = pd.read_csv('/kaggle/input/melbourne-housing-snapshot/melb_data.csv')

df.head()
s = (df.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
features = df[['Rooms','Type','Method','Bathroom','Car','Regionname']]

features.head()
features.Method.value_counts()
mapping = {'h':1,

           'u':2,

           't':3

          }

features['type_RP'] = features.Type.map(mapping) 

features.type_RP.value_counts()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



df1 = features[['Type']]

df1['Type_LE'] = le.fit_transform(features['Type'])

df1.value_counts()
df2 = pd.get_dummies(features['Type'])

df2.value_counts()
from sklearn.preprocessing import LabelBinarizer

lb_style = LabelBinarizer()

lb_results = lb_style.fit_transform(features["Type"])

pd.DataFrame(lb_results, columns=lb_style.classes_).value_counts()