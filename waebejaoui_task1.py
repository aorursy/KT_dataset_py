import pandas as pd

data = pd.read_csv("../input/invoice/invoice-amounts.csv")

data.shape
data.head()
data.columns.values
datatype = pd.read_csv("../input/datatype/Test 1 - FedEx Datatype Dictionary.csv")

datatype.shape