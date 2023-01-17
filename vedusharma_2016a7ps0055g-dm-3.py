import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from IPython.display import HTML

import base64 



pd.set_option('display.float_format', lambda x: '%.3f' % x)
test_df = pd.read_csv("../input/dm-assignment-3/Test_data.csv")

# test_df
test_df=test_df.drop(columns="Unnamed: 1809")
test_df.info()
test_df.isnull().values.any()

test=test_df.drop(columns="FileName")

data_mal = pd.read_csv("../input/dm-assignment-3/train_malware.csv")

data_mal['Harmful'] = 1

data_mal
data_mal.info()

data_mal.isnull().values.any()

data_benign = pd.read_csv('../input/dm-assignment-3/train_benign.csv')

data_benign['Harmful'] = 0

data_benign
data_benign.info()

data_benign.isnull().values.any()

data_benign["FileName"] = data_benign["FileName"].apply(lambda x: x + 1999)

concat_mb = pd.concat([data_mal,data_benign])

concat_mb
concat_mb.info()

concat_mb['Harmful'].unique()

plt.hist(concat_mb.Harmful)

concat_mb.drop(columns = 'FileName',axis =1 ,inplace = True)

concat_mb.duplicated().sum()

concat_df=concat_mb.drop_duplicates()

concat_df.info()
X=concat_df.drop('Harmful',axis=1)

y=concat_df['Harmful']
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf = RandomForestClassifier(n_estimators=1000, max_depth = 15, random_state = 30, n_jobs = -1)

rf.fit(X_train, y_train)
rf.score(X_test,y_test)

features = test.columns

prediction = rf.predict(test[features])
output = pd.DataFrame(test_df['FileName'])

output['Class'] = prediction.astype(int)

output
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

  csv = df.to_csv(index=False)

  b64 = base64.b64encode(csv.encode())

  payload = b64.decode()

  html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

  html = html.format(payload=payload,title=title,filename=filename)

  return HTML(html)

create_download_link(output)