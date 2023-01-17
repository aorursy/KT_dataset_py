# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test = pd.read_csv("/kaggle/input/santander-product-recommendation/test_ver2.csv.zip",encoding="latin1", compression="zip")
test.head(2)
sample = pd.read_csv("/kaggle/input/santander-product-recommendation/sample_submission.csv.zip",encoding="latin1", compression="zip")
sample.head(100)
df = pd.read_csv("/kaggle/input/santander-product-recommendation/train_ver2.csv.zip",encoding="latin1", compression="zip",nrows=10000)
for ft in df.columns:
    print(ft," : ",df[ft].unique()," : ",len(df[ft].unique()))
def processData(df,getDummies=True):
    data = df.copy()
    data.drop(columns=["ult_fec_cli_1t","canal_entrada"],inplace=True)
    data["conyuemp"] = data["conyuemp"].fillna('N')
    data["renta"] = data["renta"].fillna(data["renta"].mean())
    #Filling nan vlaues 
    def is_float(string):
      try:
        return float(string) or float(string)==0  
      except:  # String is not a number
        return False

    data["ind_empleado"] = data["ind_empleado"].fillna('N')

    data["age"] = data["age"].replace(' ', '', regex=True)
    data["age"] = data["age"].replace('.', '')

    data["age"] = data["age"].replace('NA',np.nan)
    data["age"] = data["age"].astype(float)

    data["ind_nuevo"] = data["ind_nuevo"].fillna(1)


    data["antiguedad"] = data["antiguedad"].replace(' ', '', regex=True)
    data["antiguedad"] = data.loc[:,"antiguedad"].replace("NA",1)
    data["antiguedad"] = data["antiguedad"].astype(int)
    data.loc[data.antiguedad<0,"antiguedad"] = 1    
    data["indfall"] = data["indfall"].fillna('S')
    data["tipodom"] = data["tipodom"].fillna(0)
    data["ind_actividad_cliente"] = data["ind_actividad_cliente"].fillna(0)
    data.drop(columns=["indrel","fecha_alta","nomprov"],inplace=True)
    data = data.dropna()    
    if(getDummies):
        data = pd.get_dummies(data, columns=['indresi','indext','conyuemp','indfall','sexo',
                                         'pais_residencia','ind_empleado','tiprel_1mes',
                                        "segmento"],drop_first=True)
    return data
def dataBin(df, getDummies=True):
    binned = df.copy()
    binned["renta"] = pd.qcut(binned["renta"], 3, labels=["low", "mid", "high"])
    binned["age"] = pd.cut(binned["age"], [0, 40, 80,200], labels=["low","mid","high"])
    binned["antiguedad"] = pd.cut(binned["antiguedad"], [0, 50, 150,250], labels=["low","mid","high"])
    if(getDummies):
        binned = pd.get_dummies(binned, columns=["renta","age","antiguedad","cod_prov"],drop_first=True)
    return binned

import seaborn as sns
import matplotlib.pyplot as plt
data = processData(df)
sns.set_style('whitegrid')
#Binning on renta, age, antiguedad
data['age'].plot(kind='hist')
plt.show()
data['renta'].plot(kind='hist')
plt.show()

data['antiguedad'].plot(kind='hist')
plt.show()

#age can be binned: 0-40, 40-80, 80+

data['antiguedad'].unique()
binned= dataBin(data)
ItemNames = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1',]
itemcols = ItemNames
itemcols.append("ncodpers") 

#binned
UserItemMatrix = df[itemcols]
UserItemMatrix = UserItemMatrix.groupby("ncodpers").sum() 
UserItemMatrix
ratings_dict = {
    "item": [],
    "user": [],
    "rating": [],
}
UserItemMatrix 

indexes = UserItemMatrix.index
items = UserItemMatrix.columns
for i in range(len(UserItemMatrix)):
    for j in range(len(items)):
        rating = UserItemMatrix.iloc[i,j]
        
        if(rating>0):
            ratings_dict[ "item" ].append(items[j])
            ratings_dict[ "user" ].append(indexes[i])
            ratings_dict["rating"].append(rating)
        else:
            ratings_dict[ "item" ].append(items[j])
            ratings_dict[ "user" ].append(indexes[i])
            ratings_dict["rating"].append(0)#UserItemMatrixCrop.iloc[:,j].mean())
from surprise import Reader, Dataset

max_rating = UserItemMatrix.max().values.max()
dfRatings = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(0, max_rating))

data = Dataset.load_from_df(dfRatings[["user", "item", "rating"]], reader)

from surprise import SVD
from surprise.model_selection import GridSearchCV

param_grid = {
    "n_epochs": [1, 30],
    "lr_all": [0.002, 0.003],
    "reg_all": [0.4]
}
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3,refit=True)
gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
from surprise.model_selection import KFold
from surprise import accuracy
from surprise import KNNBasic

kf = KFold(n_splits=5)
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = SVD()

for trainset, testset in kf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    accuracy.mae(predictions, verbose=True)
    accuracy.rmse(predictions, verbose=True)
testPersons = test["ncodpers"]
submission = {"ncodpers":[],"added_products":[]}
for personid in testPersons.values:
    preds = ""
    for itemName in ItemNames:
        pred= gs.predict(personid,itemName)
        prob = pred[3]/max_rating
        if(prob>0.05):
            preds+= " " + itemName if preds != "" else itemName
    submission["ncodpers"].append(personid)
    submission["added_products"].append(preds)        
Submissiondf = pd.DataFrame(data=submission)  
Submissiondf.set_index("ncodpers")
Submissiondf.to_csv("sub7MData5e-2.csv",index=False)
featuresNotUserInfo = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1',"fecha_dato"]
data = processData(df.drop(columns=featuresNotUserInfo),getDummies = False)
data = dataBin(data,getDummies= False)
userInfo = data.groupby("ncodpers").last()
userInfo = userInfo.reset_index()

userInfo.rename(columns={"ncodpers":"user_id"},inplace=True)
userInfo = userInfo.to_dict()
!pip install turicreate
import turicreate as tc

SF_userInfo = tc.SFrame(userInfo)
#!pip install turicreate
turiDict = {}
turiDict["item_id"] = ratings_dict["item"]
turiDict["user_id"] = ratings_dict["user"]
turiDict["rating"] = ratings_dict["rating"]
actions  = tc.SFrame(turiDict)
training_data, validation_data = tc.recommender.util.random_split_by_user(actions)
model = tc.recommender.create(training_data,target='rating',user_data=SF_userInfo)


model
model.save("recommendations.model")

ItemNames = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1',]
model.get_similar_items(ItemNames, k=24)
testPersons = test["ncodpers"]
#Recommendation skorları
recommendations = model.recommend(testPersons.values)
submission = {"ncodpers":[],"added_products":[]}

#0.9 thresholddakilerin üstü alınır
users = recommendations[recommendations["score"]>0.9]

#Test dosyasındak her user için, öneri skorlarında o user id'ye denk gelen itemleri ekle
for id in testPersons.values:
    submission["ncodpers"].append(id)
    itemsOfUser = users[users["user_id"]==id]["item_id"]
    
    #userların alacağı tüm itemleri boşluk ile ayırıp string haline getir
    itemString = ""
    for item in itemsOfUser:
        itemString += " "+item
        
    print(id," : ",itemString)    
    submission["added_products"].append(itemString)  
Submissiondf = pd.DataFrame(data=submission)  
Submissiondf.head(5)
Submissiondf.set_index("ncodpers")
Submissiondf.to_csv("turicreate.csv",index=False)