import pandas as pd


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)

print(train.head(n=5))
print(test.head(n=5))

train.QuoteConversion_Flag.values
train.drop(['QuoteNumber'], axis=1)

pd.to_datetime(train['Original_Quote_Date'])
pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train['Date'].apply(lambda x: int(str(x)[:4]))
