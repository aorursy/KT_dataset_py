import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

train = pd.read_csv("../input/train_without_noise.csv", parse_dates=['timestamp'])
print ("lines:", str(train.shape[0]) + ", " "cols:", train.shape[1])
train.tail(2).transpose()
MISS = {}
for i in range(train.shape[1]):
    MISS[train.columns[i]] = sum(pd.isnull(train[train.columns[i]]))/len(train)
MISS = {k: v for k, v in MISS.items() if v > 0}
MISS = sorted(MISS.items(), key=lambda x: x[1], reverse=True )
hist_x = []
hist_y = []
for i in MISS:
    hist_x.append(i[0])
    hist_y.append(i[1])
df = pd.DataFrame({"x":hist_x, "y":hist_y})
df.columns = ['name', '% NULL']
ind = np.arange(df.shape[0])
width = 0.6
fig, lx = plt.subplots(figsize=(12,12))
rects = lx.barh(ind, df['% NULL'], color='orange')
lx.set_yticks(ind)
lx.set_yticklabels(df['name'], rotation='horizontal')
lx.set_xlabel("Пропущено значений")
plt.show()
train["month"] = train["timestamp"].dt.month
train["year"] = train["timestamp"].dt.year
train["day"] = train["timestamp"].dt.day
train["days_passed"] = (train["timestamp"] - min(train["timestamp"])).apply(lambda x: x.days)

train.drop("timestamp", axis=1, inplace = True)
train['price_doc'].hist(bins=100)
train['price_doc'][(train['price_doc'] >1000000) & (train['price_doc'] <20000000)].hist(bins=100)
x = train.drop(["price_doc", "id"], axis=1)
y = np.log1p(train["price_doc"])

for col in x.columns:
    if x[col].dtype == 'object':
        lab = LabelEncoder()
        lab.fit(list(x[col].values)) 
        x[col] = lab.transform(list(x[col].values))
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)

xgb_params = {'max_depth': 5,
              'eta': 0.1,             
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'subsample': 0.8
}
mx_train = xgb.DMatrix(x_train, y_train)
mx_val = xgb.DMatrix(x_val, y_val)
full_train = xgb.DMatrix(x, y)

xgb_cv = xgb.cv(xgb_params, mx_train, num_boost_round=1000, early_stopping_rounds=5,
    verbose_eval=5, show_stdv=False)
xgb_cv[['train-rmse-mean', 'test-rmse-mean']].plot()
best_round = len(xgb_cv)
part_model = xgb.train(xgb_params, mx_train, num_boost_round= best_round)

preds = part_model.predict(mx_val)
validation =  {x : y for x in list(y_val) for y in  preds}

RMSE = 0
for x,y in validation.items():
    RMSE += (x - y)**2
RMSE = RMSE/len(validation)
RMSE
full_model = xgb.train(xgb_params, full_train, num_boost_round= best_round)

fig, ax = plt.subplots(1, 1, figsize=(12, 15))
xgb.plot_importance(full_model, max_num_features=60, height=0.5, ax=ax)
imp = full_model.get_fscore()
imp = sorted(imp.items(), key=lambda x: x[1], reverse=True )
imp = [x[0] for x in imp[:19]]
imp.append("price_doc")

fig, ax = plt.subplots(1, 1, figsize=(16, 16))
corr = train[imp].corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)