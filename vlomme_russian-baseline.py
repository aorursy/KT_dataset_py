from sklearn import ensemble, model_selection, datasets, metrics, tree, linear_model, preprocessing 

import xgboost as xgb

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

test = test[test.Date > "2020-04-14"]

all_data = pd.concat([train, test],ignore_index=True).sort_values(by=['Country_Region','Province_State','Date'])

all_data['ConfirmedCases'] = all_data['ConfirmedCases'].fillna(-1)

all_data['Fatalities'] = all_data['Fatalities'].fillna(-0)

countries = pd.read_csv("../input/countries/countries.csv")

countries.drop(['iso_alpha3','iso_numeric','official_name','name','iso_alpha2'], axis='columns',inplace=True)

#Карантин

data_quarantine = pd.read_csv("../input/countries/quarantine_dates.csv")

data_quarantine = data_quarantine.groupby("Country").max().loc[:,"Start date"]

data_quarantine.loc["Russia"] = "2020-03-30"

countries = countries.set_index("ccse_name", drop=True)

countries["Quarantine"] = data_quarantine

#countries = countries.rename(columns={"ccse_name": "countries"})

countries_mean = countries.mean()

all_data = all_data.merge(countries, how ="left" , left_on="Country_Region", right_on='ccse_name')

all_data['fertility_rate'] = all_data['fertility_rate'].fillna(countries_mean['fertility_rate'])

all_data['median_age'] = all_data['median_age'].fillna(countries_mean['median_age'])

all_data['migrants'] = all_data['migrants'].fillna(countries_mean['migrants'])

all_data['urban_pop_rate'] = all_data['urban_pop_rate'].fillna(countries_mean['urban_pop_rate'])

all_data['density'] = all_data['density'].fillna(countries_mean['density'])

all_data['land_area'] = all_data['land_area'].fillna(countries_mean['land_area'])

all_data['population'] = all_data['population'].fillna(countries_mean['population'])

all_data['world_share'] = all_data['world_share'].fillna(countries_mean['world_share'])

all_data['Quarantine'] = all_data['Quarantine'].fillna("2020-04-01")



all_data.drop(['Id'], axis='columns',inplace=True)

all_data['Province_State'] = all_data['Province_State'].fillna("zzz")
data2 = all_data



#Удалим дни без больных

data2 = data2[data2.ConfirmedCases != 0]

data2.loc[data2.ConfirmedCases == -1,"ConfirmedCases"] = 0

# Считаем дни от первого больного

data2["Date"] = pd.to_datetime(data2.Date)

#data2 = data2[(data2["Date"]<pd.to_datetime("2020-05-1")) & ((data2["Date"]>pd.to_datetime("2020-04-1")) | (data2.ConfirmedCases !=-1))]

data4 = data2[["Country_Region","Date"]].groupby("Country_Region").min()

data4.columns = ["Date_min"]

data2 = data2.merge(data4, how = 'left', left_on='Country_Region', right_on='Country_Region')

data2["days"] = (data2.Date - data2.Date_min).dt.days

data2["days_mart"] = (data2.Date - pd.to_datetime("2020-03-1")).dt.days

data2["days_after_Quarantine"] = (data2.Date - pd.to_datetime(data2.Quarantine)).dt.days

data2.drop(['Date_min'], axis='columns',inplace=True)

data2.Date = data2["Date"].apply(lambda x: pd.Series(x.strftime("%m-%d")))

data2 = data2.rename(columns={"ConfirmedCases": "confirmed","Fatalities":"deaths","Country_Region":"countries"})

data2[data2.countries == "Russia"].iloc[0:70]
data2[data2.countries == "Russia"]
data2.confirmed = np.log10(data2.confirmed+1)

data2.deaths = np.log10(data2.deaths+1)

old_con = data2["confirmed"].iloc[:-1]

old_con2 = data2["deaths"].iloc[:-1]

data2 = data2.iloc[1:]

data2["pred_conf"] = old_con.values

data2["pred_deaths"] = old_con2.values

data2 = data2.iloc[1:]

data2["delta_conf1"] = old_con.values[1:] - old_con.values[:-1]

data2["delta_deaths1"] = old_con2.values[1:] - old_con2.values[:-1]

data2 = data2.iloc[1:]

data2["delta_conf2"] = old_con.values[1:-1] - old_con.values[:-2]

data2["delta_deaths2"] = old_con2.values[1:-1] - old_con2.values[:-2]

data2 = data2.iloc[1:]

data2["delta_conf3"] = old_con.values[1:-2] - old_con.values[:-3]

data2["delta_deaths3"] = old_con2.values[1:-2] - old_con2.values[:-3]

data2 = data2.iloc[1:]

data2["delta_conf4"] = old_con.values[1:-3] - old_con.values[:-4]

data2["delta_deaths4"] = old_con2.values[1:-3] - old_con2.values[:-4]

data2.confirmed = data2.confirmed - data2["pred_conf"]

data2.deaths = data2.deaths - data2["pred_deaths"]

data2[data2.countries == "Russia"].iloc[0:70]
data2[data2.countries == "Russia"].iloc[70:80,10:25]
#Выберите день начала приватных данных начиная с 1 марта

days_x = 45

model_Confirmed, model_Death = {}, {}



#Удаляем ненужные фичи

#data3 = data2.drop(['Quarantine','world_share','urban_pop_rate','population', 'migrants','median_age','land_area','fertility_rate','density','delta_deaths2','delta_deaths1','pred_deaths','date','Date_min'], axis='columns')

data3 = data2.drop(['Quarantine','world_share', 'migrants','median_age','land_area','fertility_rate','density','Date'], axis='columns')

#data3 = data2.drop(['Quarantine','delta_deaths2','delta_deaths1','pred_deaths','date','Date_min'], axis='columns')

data3.population = data3.population/10**6



#Представляем штат в one_hot

one_hot = pd.get_dummies(data3['Province_State'])

data3 = data3.join(one_hot)





#Отделяем пару стран на тест

data_Korea = data3[(data3.countries == 'Korea, South') & (data3.confirmed > 0)]

data_Russia = data3[(data3.countries == 'Russia') & (data3.confirmed > 0)]



#Приватные данные

new_data = data3[data2.days_mart >= days_x]

old_data = data3[data2.days_mart < days_x]

#old_data = old_data[(old_data.countries != 'Russia')&(old_data.countries != 'Korea, South')]



#Логорифмируем предсказания

#old_data["confirmed"] = log(old_data.confirmed) 



#Убираем ответ из данных

train_labels = old_data.confirmed

train_death = old_data.deaths

train_data = old_data.drop(['Province_State', 'ForecastId','confirmed','countries','deaths'], axis='columns')

train_data
def pred_score(models, data, death = False, plot = 0):

  predictions_all = 0

  plt.rcParams['figure.figsize'] = [20, len(models)*10]

  data = data[(data.confirmed > 0)]

  death_labels = 10**(data.deaths + data.pred_deaths)- 1

  labels = 10**(data.confirmed + data.pred_conf )- 1

  data = data.drop(['Province_State','ForecastId','confirmed','countries','deaths'], axis='columns')  

  for i, model in enumerate(models):

    if model == "reg":

      predictions = models[model].predict(scaler.transform(data))

    else:

      predictions = models[model].predict(data)

    if death:

      predictions = predictions + data.pred_deaths.values

    else:

      predictions = predictions + data.pred_conf.values    

    predictions = 10**predictions - 1

    predictions_all = predictions_all + predictions

    if death:

      print("Ошибка MALE по смертям", model, np.mean(np.abs(np.log10((predictions+1)/(death_labels+1)))))

    else:

      print("Ошибка MALE по заражениям", model, np.mean(np.abs(np.log10((predictions+1)/(labels+1)))))

    if plot:

      plt.subplot(len(models), 1, i+1)

      if plot==1:

        plt.plot(predictions, label = "Предсказанное значение") 

        if death:

          plt.plot(death_labels.values, label = "Истинное значение")

          plt.gca().set(xlabel='Дни от случая первого заражения', ylabel='Смерти')

        else:

          plt.plot(labels.values, label = "Истинное значение")  

          plt.gca().set(xlabel='Дни от случая первого заражения', ylabel='Заражения')

      if plot==2:

        plt.scatter(np.arange(0,len(predictions),1), predictions, s = 1, label = "Предсказанное значение") 

        if death:

          plt.scatter(np.arange(0,len(predictions),1), death_labels.values, s = 1,  label = "Истинное значение")

          plt.gca().set(xlabel='Дни от случая первого заражения', ylabel='Смерти')

        else:

          plt.scatter(np.arange(0,len(predictions),1), labels.values, s = 1,  label = "Истинное значение")  

          plt.gca().set(xlabel='Дни от случая первого заражения', ylabel='Заражения')      

      plt.title(model)

      plt.grid(True)

      plt.legend() 

  predictions_all = predictions_all/len(models)

  if death:

    print("Ошибка MALE по смертям средняя", np.mean(np.abs(np.log10((predictions_all+1)/(death_labels+1)))))

  else:

    print("Ошибка MALE по заражениям средняя", np.mean(np.abs(np.log10((predictions_all+1)/(labels+1)))))      
treeDepth = 30

mdl = tree.DecisionTreeRegressor(max_depth=treeDepth)

param_grid = {

    'n_estimators': [100],

    'learning_rate': [0.0002],

    'loss' : ["exponential"]

                }

regrMdl = ensemble.AdaBoostRegressor(base_estimator=mdl)

model_Confirmed["Adaboost"] = model_selection.RandomizedSearchCV(estimator = regrMdl, param_distributions = param_grid, n_iter = 100, 

                                         cv = 3, verbose=0, random_state=42, n_jobs = -1).fit(train_data, train_labels)

model_Death["Adaboost"] = model_selection.RandomizedSearchCV(estimator = regrMdl, param_distributions = param_grid, n_iter = 100, 

                                         cv = 3, verbose=0, random_state=42, n_jobs = -1).fit(train_data, train_death)


model_Confirmed["RandomForest"]  = ensemble.RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42,

                                                                  n_jobs = -1).fit(train_data, train_labels)

model_Death["RandomForest"] = ensemble.RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42,

                                                            n_jobs = -1).fit(train_data, train_death)

                                                            


model_Confirmed["Xgboost"] = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=200, max_depth=20, random_state=42,

                                              n_jobs = -1).fit(train_data, train_labels)

model_Death["Xgboost"] = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=200, max_depth=20, random_state=42,

                                          n_jobs = -1).fit(train_data, train_death)
#model_Confirmed["new"] = 

#model_Death["new"] = 
#del model_Confirmed["Ridge"] 

#del model_Death["Ridge"] 
# Данные new_data - новые с дня Х, 

# data_Russia - Данные по России

# data_Korea  - Данные по Франции

data_pred = data_Korea 



# Тип графика 0 - без графика

# 1 - plot

# 2 - scatter

plot_type = 1



#Строим для заражений, или смертей

death_bool = False 

if death_bool:

  model = model_Death

else:

  model = model_Confirmed
pred_score(model, data_pred, death_bool, plot_type)
# Сколько дней прогнозируем?

days_prog = 30



Matrix_confirmed,Matrix_Death ={},{}

for model_name in model_Confirmed:

  new_data_list = new_data[new_data.days_mart == days_x]

  labels = new_data_list.countries

  #predictions_confirmed = [(10**new_data_list.pred_conf - 1).tolist()]

  #predictions_Death = [(10**new_data_list.pred_deaths - 1).tolist()]

  predictions_confirmed = []

  predictions_Death = []

  new_data_list = new_data_list.drop(['Province_State', 'ForecastId','confirmed','countries','deaths'], axis='columns')

  for _ in range(days_prog):

    #print(data.iloc[-10:-7,-5:])

    prediction_confirmed = model_Confirmed[model_name].predict(new_data_list)

    prediction_Death = model_Death[model_name].predict(new_data_list)



    prediction_confirmed = prediction_confirmed*(0.95**(new_data_list.days_after_Quarantine.values/10 - 1.4))

    prediction_Death = prediction_Death*(0.95**(new_data_list.days_after_Quarantine.values/10  - 1.4))

    

    prediction_confirmed = prediction_confirmed + new_data_list.pred_conf.values

    prediction_Death = prediction_Death + new_data_list.pred_deaths.values

    

    """bad_confirmed = prediction_confirmed<new_data_list["pred_conf"]

    prediction_confirmed[bad_confirmed] = new_data_list[bad_confirmed]["pred_conf"]+np.round(new_data_list[bad_confirmed]["delta_conf1"]*0.9)

    bad_Death = prediction_Death<new_data_list["pred_deaths"]

    prediction_Death[bad_Death] = new_data_list[bad_Death]["pred_deaths"]+np.round(new_data_list[bad_Death]["delta_deaths1"]*0.9)"""

    new_data_list["delta_conf2"] = new_data_list["delta_conf1"]

    new_data_list["delta_conf1"] = prediction_confirmed - new_data_list["pred_conf"] 

    new_data_list["pred_conf"] = prediction_confirmed



    new_data_list["delta_deaths2"] = new_data_list["delta_deaths1"]

    new_data_list["delta_deaths1"] = prediction_Death - new_data_list["pred_deaths"] 

    #data[data["delta_conf1"] < 0].loc[:,"delta_conf1"] = 0

    new_data_list["pred_deaths"] = prediction_Death  

    prediction_Death = 10**prediction_Death - 1

    prediction_confirmed = 10**prediction_confirmed - 1  

    new_data_list[["days_mart", "days"]] += 1

    

    predictions_Death.append(np.round(prediction_Death))

    predictions_confirmed.append(np.round(prediction_confirmed))

  data_list = pd.date_range('2020-04-'+str(days_x-30), periods = days_prog, freq ='d')

  data_list = data_list.strftime('%#m-%#d') 

  Matrix_confirmed[model_name] = pd.DataFrame(predictions_confirmed,columns = labels,index = data_list)

  Matrix_Death[model_name] = pd.DataFrame(predictions_Death,columns = labels,index = data_list)

Matrix_confirmed["Среднее"] = (Matrix_confirmed["Xgboost"] + Matrix_confirmed["RandomForest"] + Matrix_confirmed["Adaboost"])//3

Matrix_Death["Среднее"] = (Matrix_Death["Xgboost"] + Matrix_Death["RandomForest"] + Matrix_Death["Adaboost"])//3
def plot_new(list_countries, Matrix, Death = False,subplot_x = 2):

  dat2 = data2.copy()  

  plt.rcParams['figure.figsize'] = [20, len(list_countries)*20/subplot_x**2]

  dat2.deaths = 10**(dat2.deaths + dat2.pred_deaths)- 1

  dat2.confirmed = 10**(dat2.confirmed + dat2.pred_conf )- 1

  for i, Country in enumerate(list_countries):

    plt.subplot(len(list_countries)// subplot_x + 1, subplot_x, i+1)

    yyy = dat2[(dat2.countries == Country)&(dat2.confirmed>0)&(dat2.days_mart>=days_x - 10)]

    

    if Death:

      plt.plot(yyy.Date, yyy.deaths, label = "Истинное значение")

      plt.gca().set(xlabel='Дата', ylabel='Смерти')

    else:

      plt.plot(yyy.Date, yyy.confirmed, label = "Истинное значение")

      plt.gca().set(xlabel='Дата', ylabel='Заражения')

    for name in Matrix:

      yyy2 = Matrix[name][[Country]]

      plt.plot(yyy2, label = name)

    plt.title(Country)

    plt.xticks(np.arange(0, (10+days_prog), ((10+days_prog)//14)*subplot_x))

    #plt.yticks(np.linspace(0,10+days_prog,30//subplot_x))

    plt.grid(True)

    plt.legend()
# Создаём список стран

list_countries = countries.index.to_list()[66:70]

#list_countries = []

list_countries.extend(["Russia", "Italy"])



# Количество графиков в строке

subplot_x = 2



#Строим для заражений, или смертей

death_bool = False 

if death_bool:

  Matrix = Matrix_Death

else:

  Matrix = Matrix_confirmed
plot_new(list_countries,Matrix,death_bool,subplot_x)
ans = Matrix_confirmed["Adaboost"].stack().reset_index()

ans_Death = Matrix_Death["Adaboost"].stack().reset_index()

ans["prediction_deaths"] = ans_Death[0]

ans["Province_State"] =  new_data.groupby(["countries", "Province_State"]).max().reset_index()["Province_State"].tolist()*30

ans.columns = ['Date','Country_Region','ConfirmedCases', 'Fatalities',"Province_State"]

ans['Date'] = '2020-'+ans['Date']



train2 = train[(train.Date >= "2020-04-02")]

train2['Province_State'] = train2['Province_State'].fillna("zzz")

train2.drop(['Id'], axis='columns',inplace=True)

ans = pd.concat([train2, ans],ignore_index=True)

ans['ConfirmedCases'] = ans['ConfirmedCases']

ans['Fatalities'] = ans['Fatalities']

ans = ans.sort_values(by=['Country_Region','Province_State','Date'])

ans['ForecastId'] = np.arange(1,len(ans)+1)

ans = ans[['ForecastId','ConfirmedCases','Fatalities']]

ans.to_csv('submission.csv', index=False)