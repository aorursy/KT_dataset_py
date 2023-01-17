import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime as date, timedelta

import matplotlib.pyplot as plt

import os



# Notebook created using kaggle.com



df_column_names = pd.read_excel('../input/pracownicy/Pracownicy.xlsx', sheet_name = 'Kolumny', header = None)

print(df_column_names[0])
df_column_names_copy = df_column_names.copy()

print(df_column_names_copy)
df_column_names.head(len(df_column_names))



#Dzięki wyświetleniu danych, można określić ich strukturę oraz zapoznać się z formą pokazanych danych. 

#Używając nawiasów kwadratowych program dodatkowo ukazuje nam nazwę (Name) oraz typ danych (dtype).
df = pd.read_excel('../input/pracownicy/Pracownicy.xlsx', sheet_name = 'Dane', header = None)

df.head()



# Z kolei podejrzenie pierwszych 5 elementów 'dane' pokazuje nam rzeczywisty wygląd DataFrame'a.

# Z tego poziomu można już określić, że dane będą wymagały obrobienia ponieważ występują dane wybrakowane 

# 'NaN'. Są one specjalnymi danymi definiowanymi przez IEE754. Zgodnie z dokumentacją są one wartościami, 

# które nie są blisko żadnej innej a do tej grupy należy NaN, inf, -inf.
# Sprzężenie danych (kolumn do rekordów danych):



# Od teraz dl_column_names ma być równy ciągowi pierwszych elementów napotkanych w kolejnych iteracjach siebie. 

# Tworzy nam to szyk kolejnych nazw kolumn, które można łatwo dodać używając argumentu names dla metody read_excel.

df_column_names = [i[0] for i in df_column_names.values]



df = pd.read_excel('../input/pracownicy/Pracownicy.xlsx', sheet_name = 1, 

                   names = df_column_names, header = None)

df.head(2)
df.dtypes
df.describe
df.shape
# Przyporządkowanie danym odpowiednich typów 

# (kontrola nad typami oraz 'static typing' znacząco zmniejsza ilość błędów w czasie 

# kompilacji. Specjalistą Pythona nie jestem ale nie bez powodu cały świat porzucił JavaScript 

# na rzeczy chociaż TypeScript ;) ) tak więc model który mnie interesuje powinien wyglądać tak:



# Co prawda spowoduje to zmianę NaN na string nan o czym trzeba pamiętać!



df = df.astype({

    'Imie':'str',

    'Nazwisko':'str',

    'Stanowisko':'str',

    'Data Przyjecia':'datetime64',

    'Data Zwolnienia':'datetime64',

    'Typ Umowy':'str',

    'Pensja':'int64'

})

df.dtypes
# Utworzenie kolumny index przez wykorzystanie reset_index

df = df.reset_index()



df.head()
# Przypisanie kolumny index jako nasz index oraz zmiana nazwy na bardziej przystępną



df = df.set_index(['index'])

df.index.names = (['id'])

df.head()
# Przekształcenia stringa przykładowe:



df['Stanowisko'].str.upper()
# Sprawdź czy stanowisko kończy się na podany człon:



df['Stanowisko'].str.endswith('dawca')
df['Stanowisko'] = df['Stanowisko'].replace('nan', 'Dyrektor')

df.head(len(df))



# Data Zwolnienia nie wymaga zmiany. NaT będzie oznaczał, że osoba wciąż jest zatrudniona.

# Ze względu na zastosowanie statycznego typowania najłatwiej jest mi zastąpić pozycje funkcją replace
df['Typ Umowy'] = df['Typ Umowy'].replace('nan', 'Kontrakt')

df.head(len(df))
# Sortowanie po dacie przyjęcia:



df.sort_values('Data Przyjecia')
# Średnie zarobki na danym stanowisku sortowane malejąco: 

#(przykład: Antoni Kucharski był zatrudniony powtórnie, dane bez poprawki na to poniżej:)



df_mean_job_income = df[['Stanowisko', 'Pensja']].groupby(by = ['Stanowisko']).mean().sort_values(by = ['Pensja'], ascending = False)

df_mean_job_income
# Dane z poprawką na date dzisiejszą:



df_reworked = df.copy()



# Wyciągam listę id osób do usunięcia:



no_longer_working_ids = df_reworked[df_reworked['Data Zwolnienia'] < date.today()].index



df_reworked.drop(no_longer_working_ids, inplace = True)



df_reworked.head(len(df_reworked))



# Pozostały tylko osoby zatrudnione.
# Średnia zarobków obecnie pracujących ludzi:



only_working = df_reworked[['Stanowisko', 'Pensja']].groupby(

    by = ['Stanowisko']).mean().sort_values(by = ['Pensja'], ascending = False).round(2)

only_working
# Posortowane malejąco odchylenie standardowe pensji na poszczególnym stanowisku. Używam danych całościowych.



# Najpierw wyliczam odchylenie, zaokrąglam oraz sortuje je malejąco oraz zmieniam nazwę kolumny Pensja



std_deviation = df[['Stanowisko','Pensja']].groupby('Stanowisko').std().round(2).sort_values(by = ['Pensja'], ascending = False)

std_deviation.columns = ['Odchylenie standardowe pensji']

std_deviation
# Timedelta zwiększona o 8 godzin



df_timedelta = df.copy()



df_timedelta['Data Przyjecia'] = df_timedelta['Data Przyjecia'] + timedelta(hours = 8)

df_timedelta
# Wykres prezentujący liczbę pracowników w każdym miesiącu od momentu powstania ów sklepu.



# Najpierw znaleźć pierwszy miesiąc działania sklepu:



df['Data Przyjecia'].sort_values().head(1)
df['Data Zwolnienia'].sort_values().head(1)
# Ostatnia data w rejestrze:



df['Data Przyjecia'].sort_values(ascending = False).head(1)
df['Data Zwolnienia'].sort_values(ascending = False).head(1)
# Wiemy teraz, że oś x będzie przedziałem od 2016-05-09 do 2020-01-01 

# (ponieważ ostatnia data znaleziona to 2019-12-12)

# Pozwoli to doliczyć ostatnie daty przyjecia.



dates = pd.date_range(start = '2016-05-09', end = '2020-01-01', freq = 'M', closed = 'left')

dates
formatted_dates = dates.strftime('%Y-%m')



count = 0

staff_monthly = []



for date_id,date in enumerate(dates):

    for row in range(len(df)):

        if (dates[date_id-1] != dates[len(dates) - 1]):

            if (dates[date_id-1] < df['Data Przyjecia'][row] < date):

                count = count + 1

            if (dates[date_id-1] < df['Data Zwolnienia'][row] < date):

                count = count - 1

        else:

            if (df['Data Przyjecia'][row] < date):

                count = count + 1

            if (df['Data Zwolnienia'][row] < date):

                count = count - 1

    staff_monthly.append(count)

        

print(staff_monthly, len(staff_monthly), len(dates))
# Upiększam daty:



dates_stringified = dates.strftime('%Y-%m')
plt.figure(figsize=(14,8))

plt.xticks(rotation=90)

plt.yticks(staff_monthly)

plt.grid()

plt.plot(dates_stringified,staff_monthly)

plt.show()