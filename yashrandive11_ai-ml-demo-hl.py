import pandas as pd
data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
type(data)
data
data.describe(include = 'all')