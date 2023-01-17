import pandas as pd



stat = pd.read_csv("../input/data.csv")
print(stat)

print(stat.info())
# Узнаём пропуски в каждом из столбцов и распечатаем те столбцы где есть прпуски:

#print(stat[stat['days_employed'].isnull()].count()) #- 2174 пропуск

#print(stat[stat['total_income'].isnull()].count())  # - 2174 пропуск

#print(stat[stat['dob_years'].isnull()].count()) #- пропуски отсутвуют

#print(stat[stat['education'].isnull()].count()) #- пропуски отсутвуют

#print(stat[stat['income_type'].isnull()].count()) #- пропуски отсутвуют

#print(stat[stat['children'].isnull()].count()) #- пропуски отсутвуют

#print(stat[stat['family_status'].isnull()].count()) #- пропуски отсутвуют  

#print(stat[stat['purpose'].isnull()].count())  #пропуски отсутвуют  

#print(stat[stat['debt'].isnull()].count())  #пропуски отсутвуют    

#print(stat[stat['gender'].isnull()].count())  #пропуски отсутвуют  

 
stat.isnull().sum()
# Шаг 1. Заполним все пропуски значениями Nan:

stat['days_employed'] = pd.to_numeric(stat['days_employed'], errors='coerce')

stat['total_income'] = pd.to_numeric(stat['total_income'], errors='coerce')



# Шаг 2. Заменим отрицательные значения стажа работы и количества детей на положительные,а стаж переведем в года:

stat['children']= abs(stat['children'])

stat['days_employed']= abs(stat['days_employed'])

stat['days_employed']= stat['days_employed'].apply(lambda x: x/365)



# Шаг 3. Посчетаем медиану этих столбцов:

stat_day_med = stat['days_employed'].median() #средний стаж в днях - 1808,36 день, в годах - 4 года

stat_total_med = stat['total_income'].median() #средний доход 145017,93 руб



# Шаг 4.Заполним пропуски значениями их медиан:

stat['days_employed']= stat['days_employed'].fillna(value=stat_day_med)

stat['total_income']= stat['total_income'].fillna(value=stat_total_med)



print(stat['total_income'])

# Произведём ручной поиск дубликатов с учётом регистра. Выберем категориальные столбцы где чаще всего есть разный регистр:

stat['education'] = stat['education'].str.lower()

stat['family_status'] = stat['family_status'].str.lower()

stat['income_type'] = stat['income_type'].str.lower()

stat['purpose'] = stat['purpose'].str.lower()



# Теперь найдём дубликаты в каждом столбце

#print(stat['education'].value_counts())  #дубликатов в этом значениии не найдено (5 параметров)

#print(stat['family_status'].value_counts())  #дубликатов в этом значениии не найдено (5 параметров)

#print(stat['income_type'].value_counts()) #дубликатов в этом значениии не найдено (8 параметров)



print(stat['purpose'].value_counts())  # 38 значений, которые по своему дуюлируют друг друга

#stat['education'] = stat['education'].drop_duplicates()  



# Обрабокту дубликатов буду делать по трём столбцам одной строки, чтобы избежать попадания в дубликаты одинковых 

# значений клиентов банка. Данный метод считаю более применимым к данному кейсу



#duplicatedRows = stat[stat.duplicated(['family_status', 'education'])]

#print(duplicatedRows)

# Лемматизировать будем столбец который дал нам большое количество дубликатов и создадим для него отдельный столбец:

for i in range (len(stat)):

    lemmas = m.lemmatize(stat['purpose'][i])



stat['purpose_lemma'] = stat['purpose'].apply(m.lemmatize)

print(stat['purpose_lemma'])
# Категоризацию данных построим по принципу зависимости количества детей от возможной дефолтности

# Для понимания зависимости объеденим в категорию следующие значения:

# "Дефолтниками' будут признаны в значении: children <= 5 и значении debt = 1 ("Дефолт" - просрочка)

# 'Плательщики' в значении: children <= 5  и значении unemployed = 0

def children_and_debt(row):

    children = row['children']

    debt = row['debt']

   

    if 1 < children <= 5:

        if debt == 1:

            return 'Дефолтники'

       

    if 1 < children <= 5:

        if debt == 0:

            return 'Плательщики'

        

    if children == 0:

        if debt == 1:

            return 'Дефолтники без детей'

        

    if  3 <= children <= 5:

        if debt == 1:

            return 'Дефолтники c тремя детьми'

        

# Протестируем: Первое - количество детей, второе - кредитная задолженность:     

row_values = [0, 1]



row_columns = ['children', 'debt'] #названия столбцов

row = pd.Series(data=row_values, index=row_columns)

 

print(children_and_debt(row)) #проверяем работу функции

 
# Наблюдаем как работает функция с заданными значениями, для этого создаём новый столбец:

stat['children_return'] = stat.apply(children_and_debt, axis=1)

print(stat['children_return'].value_counts())



#Плательщики    2210 - 92%

#Дефолтники      1733 - 8% 

#### Из них без детей - 1063 - 61 %
#Первую зависимость нашли в этапе 6, где через цикл её можно было проследить 

print(stat.groupby('children')['debt'].value_counts())

#Выберем метод свободных таблиц для определения зависимости.

# Сопоставим семейное положение и их ID параметром:

# женат / замужем -  0;   гражданский брак -1;    довец / вдова - 2;

# в разводе - 3;          не женат / не замужем - 4  



def family_status_and_debt(row):

    family_status = row['family_status']

    debt = row['debt']

#В браке:   

    if family_status == 0:

        if debt == 1:

            return 'Дефолтный клиент в браке'

       

    if family_status == 0:

        if debt == 0:

            return 'Платежеспособный клиент не в браке'

        

# В гражданском браке:        

    if family_status == 1:

        if debt == 1:

            return 'Дефолтный клиент в гражданском браке'

       

    if family_status == 1:

        if debt == 0:

            return 'Платежеспособный клиент в гражданском браке'

        

# Вдовец / вдова:        

    if family_status == 2:

        if debt == 1:

            return 'Дефолтный клиент вдовец / вдова'

       

    if family_status == 2:

        if debt == 0:

            return 'Платежеспособный клиент вдовец / вдова'

        

# в разводе:        

    if family_status == 3:

        if debt == 1:

            return 'Дефолтный клиент в разводе'

       

    if family_status == 3:

        if debt == 0:

            return 'Платежеспособный клиент в разводе'

        

# не женат / не замужем:        

    if family_status == 4:

        if debt == 1:

            return 'Дефолтный клиент не женат'

       

    if family_status == 4:

        if debt == 0:

            return 'Платежеспособный клиент не замужем'

        

# Протестируем: Первое - статутс достатка клиента, второе - кредитная задолженность:     

row_values = [1, 1]



row_columns = ['family_status', 'debt'] #названия столбцов

row = pd.Series(data=row_values, index=row_columns)

 

print(family_status_and_debt(row)) #проверяем работу функции

# Наблюдаем как работает функция с заданными значениями, для этого создаём новый столбец:

stat['family_status_return'] = stat.apply(family_status_and_debt, axis=1)

print(stat['family_status'].value_counts())

def total_income_and_debt(row):

    total_income = row['total_income']

    debt = row['debt']

#Низкий достаток:   

    if 1 < total_income <= 50000:

        if debt == 1:

            return 'Дефолтный клиент с низким достатком'

       

    if 1 < total_income <= 50000:

        if debt == 0:

            return 'Платежеспособный клиент с низким достатком'

        

# Средний достаток:        

    if 51000  < total_income <= 150000:

        if debt == 1:

            return 'Дефолтный клиент со средним достатком'

       

    if 51000  < total_income <= 150000:

        if debt == 0:

            return 'Платежеспособный клиент со средним достатком'

        

# Высокий достаток:        

    if 150000  < total_income <= 500000:

        if debt == 1:

            return 'Дефолтный клиент с высоким достатком'

       

    if 150000  < total_income <= 500000:

        if debt == 0:

            return 'Платежеспособный клиент с высоким достатком'

# Протестируем: Первое - статус достатка клиента, второе - кредитная задолженность:     

row_values = [300000, 1]



row_columns = ['total_income', 'debt'] #названия столбцов

row = pd.Series(data=row_values, index=row_columns)

 

print(total_income_and_debt(row)) #проверяем работу функции
stat['total_income_return'] = stat.apply(total_income_and_debt, axis=1)

print(stat['total_income_return'].value_counts()) # считаем общие значения

otn1 = stat.loc[(stat['total_income_return'] == 'Платежеспособный клиент с высоким достатком')].count().sum() / stat.loc[(stat['total_income_return'] == 'Дефолтный клиент с высоким достатком')].count().sum()

otn2 = stat.loc[(stat['total_income_return'] == 'Платежеспособный клиент со средним достатком')].count().sum() / stat.loc[(stat['total_income_return'] == 'Дефолтный клиент со средним достатком')].count().sum()

otn3 = stat.loc[(stat['total_income_return'] == 'Платежеспособный клиент с низким достатком')].count().sum() / stat.loc[(stat['total_income_return'] == 'Дефолтный клиент с низким достатком')].count().sum()

print()

print('Отношение клиентов с высоким достатком:', otn1) 

print('Отношение клиентов со средним достатком:', otn2)

print('Отношение клиентов с низким достатком:', otn3)
#Для ответа на это вопрос применим результаты лематизации

# Недвижимость - 10 упоминаний; автомобиль - 9; образование - 9; 

# жильё - 7 (что оносится к недвижимости); строительство - 3. 



def purpose_lemma(row):

    if 'жилье' in row:

        return 1

    elif 'недвижимость' in row:

        return 1

    elif 'коммерческий' in row:

        return 1

    elif 'операция' in row:

        return 1

    elif 'сделка' in row:

        return 2

    elif 'автомобиль' in row:

        return 2

    elif 'образование' in row:

        return 3

    elif 'свадьба' in row:

        return 4

    elif 'строительство' in row:

        return 5

    else:

        return 6

    

stat['purpose_lemma'] = stat['purpose'].apply(purpose)

purpose_table = stat.groupby(['purpose_lemma'], as_index = False).agg({'debt':'sum', 'purpose': 'count'})

#purpose_table.columns = ['Цели кредита', 'Кол-во должников', 'purpose_id']

#print(purpose_table)



stat_pivot = stat.pivot_table(index='purpose_lemma', columns='debt', values='purpose', aggfunc='count')

print(stat_pivot)