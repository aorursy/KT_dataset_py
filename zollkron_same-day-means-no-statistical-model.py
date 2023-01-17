import pandas as pd
input_dir = "../input/m5-forecasting-accuracy/"

data = pd.read_csv(input_dir + "sales_train_validation.csv")

data
data.columns
calendar_data = pd.read_csv(input_dir + "calendar.csv")

calendar_data
sale_dates = calendar_data['date']

sale_dates
#from datetime import datetime  

#from datetime import timedelta  



#sale_date = datetime(2011,1,29) #First Date in the Sales Calendar

#print(sale_date.date())
i = 0

for column in enumerate(data.columns):

    #print(i, column)

    if i > 5:

        sale_date = sale_dates[i-6]

        #print(sale_date)

        data.rename(columns={data.columns[column[0]]: sale_date}, inplace= True) 

    i += 1

print(data.columns)
def get_units_sold(data, month_day):

    columns = list(data.columns)

    #print(columns)

    col = []

    for column in columns:

        if month_day in column:

            col.append(column)

    columns = col

    #print(columns)

    units_sold = data.loc[:,columns]

    #print(units_sold)

    return units_sold
def get_mean(units_sold, index):

    #print(units_sold.loc[index,:])

    sum_units = sum(units_sold.loc[index,:])

    units = len(units_sold.loc[index,:])

    result = sum_units//units

    #print(result)

    return result
def get_list_means(units_sold):

    list_means = []

    for i in range(len(units_sold)):

        list_means.append(get_mean(units_sold, i))

    return list_means
#units_sold = get_units_sold(data,"07-04") #Example Independence Day

#get_mean(units_sold, 2186) #Example Item with index 2186

#print(get_list_means(units_sold)) #List of all Items Means by Same Day
validation_dates = calendar_data.loc[1913:1940,'date']

prediction_validation = {}

for validation_date in list(validation_dates):

    #print(validation_date)

    month_day = validation_date[5:]

    #print(month_day)

    units_sold = get_units_sold(data, month_day)

    list_means = get_list_means(units_sold)

    prediction_validation[validation_date] = list_means

print(prediction_validation)
#print(prediction_validation)
evaluation_dates = calendar_data.loc[1941:1969,'date']

prediction_evaluation = {}

for evaluation_date in list(evaluation_dates):

    #print(validation_date)

    month_day = evaluation_date[5:]

    #print(month_day)

    units_sold = get_units_sold(data, month_day)

    list_means = get_list_means(units_sold)

    prediction_evaluation[evaluation_date] = list_means

print(prediction_evaluation)
#print(prediction_evaluation)
final_columns = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17'

           ,'F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28']

columns = list(prediction_validation.keys())

ids = data['id']

#print(ids)

df = pd.DataFrame(prediction_validation, columns = columns)

i = 0

for column in enumerate(df.columns):

    df.rename(columns={df.columns[column[0]]: final_columns[i]}, inplace= True)

    i += 1

df.insert(0, 'id', ids)

print(df)
ids2 = list(ids)

for i in range(len(ids2)):

    ids2[i] = ids2[i].replace("validation","evaluation")

#print(ids2)

columns = list(prediction_evaluation.keys())

df2 = pd.DataFrame(prediction_evaluation, columns = columns)

i = 0

for column in enumerate(df2.columns):

    df2.rename(columns={df2.columns[column[0]]: final_columns[i]}, inplace= True)

    i += 1

df2.insert(0, 'id', ids2)

print(df2)
final_df = pd.concat([df, df2], ignore_index=True)

final_df
final_df.to_csv("submission.csv", index = False)