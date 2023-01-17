import numpy as np
import pandas as pd
import pandas_profiling as pdp
import os
import warnings
warnings.filterwarnings("ignore")
print(os.listdir("../input"))
from IPython.display import HTML
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
profile = pdp.ProfileReport(train)
profile.to_file(outputfile="train.html")
HTML(filename='train.html')
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
profile = pdp.ProfileReport(test)
profile.to_file(outputfile="test.html")
HTML(filename='test.html')
transaction = pd.read_csv("../input/historical_transactions.csv", parse_dates=["purchase_date"])
profile = pdp.ProfileReport(transaction)
profile.to_file(outputfile="transaction.html")
HTML(filename='transaction.html')
new_transaction = pd.read_csv("../input/new_merchant_transactions.csv", parse_dates=["purchase_date"])
profile = pdp.ProfileReport(new_transaction)
profile.to_file(outputfile="new_transaction.html")
HTML(filename='new_transaction.html')
merchant = pd.read_csv("../input/merchants.csv")
profile = pdp.ProfileReport(merchant)
profile.to_file(outputfile="merchant.html")
HTML(filename='merchant.html')
# data_dict = pd.read_excel("../input/Data_Dictionary.xlsx")
# data_dict
