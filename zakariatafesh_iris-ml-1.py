import pandas as pd
iris_file = '../input/iris-data/iris.csv'
data_iris = pd.read_csv(iris_file)
data_iris = data_iris.dropna(axis=0)

data_iris.columns
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]
df = pd.DataFrame()
data_iris.columns
y = data_iris.variety
y
X = [data_iris.columns[:-1]]
X