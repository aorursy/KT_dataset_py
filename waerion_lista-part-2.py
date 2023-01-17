import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime, timedelta

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
country_code_2017_df = pd.read_csv('../input/GFDDData.csv')[['Country Name','Indicator Name','2017']]
country_code_2017_df.head()
# Dane zawierają ponad 25 tysięcy rekordów na trzech kolumnach plus numeracja rekordów.

country_code_2017_df.shape
country_code_2017_df.columns
# Dwie pierwsze kolumny są obiektami czyli najprawdopodobniej zwykłymi typami string oraz 
# rok ma wysoką dokładność dla liczb typu float 
# (czyli typu rzeczywistych, przechowuje liczby zmiennoprzecinkowe w zakresie 64 bitowym 
# znane dokładniej jako "double-precision floating-point")

country_code_2017_df.dtypes
country_code_2017_df.size
# Dane posiadają automatycznie ustawiony index 
# (prawdopdobnie i tak będzie trzeba go resetować i ustawić)

country_code_2017_df.index
# Ukazuje ile wymiarów opisuje tą ramkę

country_code_2017_df.ndim
# Inny sposób ukazanie danych z shape oraz podglądu danych

country_code_2017_df.memory_usage
# Uzyskana ramka pokazuje w przejrzysty sposób pobrane dane. Widać teraz wartości liczbowe 
# dla każdego wybranego wskaznika. Mamy bardzo dużo braków w danych.

# Pivot table:

pivot_df = country_code_2017_df.pivot_table(index = 'Country Name', columns = 'Indicator Name', values = '2017')
pivot_df.head()
# Państwa, które zmagały się z kryzysem to te wyznaczane przez 'Indicator Name':
# 'Bank cost to income ratio (%)', 'Bank non-performing loans to gross loans (%)'

# Zaś najistorniejszy wydaje się być:
# 'Banking crisis dummy (1=banking crisis, 0=none)'

country_code_2017_df['Indicator Name'].unique()
# Tak więc wykorzystam dane zero jedynkowe dla kryzysu:
# Dane zawierały bardzo dużo NaN, wymagało to metody: dropna()
# Kraje w kryzysie bankowym to Mołdawia oraz Ukraina

bank_crisis_countries = pivot_df['Banking crisis dummy (1=banking crisis, 0=none)'].where(pivot_df['Banking crisis dummy (1=banking crisis, 0=none)'] == 1).dropna()
bank_crisis_countries
# Po sprawdzeniu pierwszy 15 rekordów ustaliłem, że 10 zawiera komplet danych

pivot_df.isna().sum().sort_values(ascending = True).head(15)
# Wybrałem prawie kompletną 'Bank cost to income ratio (%)'

# Przedstawione dane w histogramie ukazują rozkład informacji dla danej kolumny.
# Oś x przedstawia procentowy koszt uzyskania przychodu przez bank
# Oś y ukazuje populację
# Na podstawie tych danych można wywnioskować, że absolutna większość Banków ma koszty 
# uzyskania przychodu na poziomie 40%-65%. Co ciekawe jest bank który przekracza 100%.
# Bank ten ponosi stratę, jest to bank Syrian Arab Republic (odczytane z kodu poniżej histogramu)
# Histogram ten pozwala ocenić w jakich krajach banki mają bardzo dobry zwrot np. z instrumentów 
# finansowych jakimi mogą być kredyty

pivot_df['Bank cost to income ratio (%)'].hist(bins=30, figsize=(20, 20), grid = False)
pivot_df['Bank cost to income ratio (%)'].sort_values(ascending = False)
# Ramka uzyskana po pivot z uzupełnionymi brakującymi danymi przez wprowadzenie wartości uśrednionych


pivot_df_mean_values = pivot_df.mean().round(4)
pivot_df = pivot_df.fillna(pivot_df_mean_values)
pivot_df.head()
#Plik zapisany w output od kaggle

new_csv = pivot_df.copy()
new_csv.to_csv('new_data_file.csv')
# Odszukanie pliku w folderach chmury kaggle

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Odczyt uzyskanego pliku

melt_new_csv = pd.read_csv('../working/new_data_file.csv')
melt_new_csv
# Użycie melt, ramka została rozpisana teraz jako mapa zmiennej i wartości:

melt_new_csv.melt()
# Użycie melt na oryginalnej ramce pivot, struktura ramki wróciła do stanu niemal początkowego (fillna zostały zapisane)

almost_original_df = pivot_df.melt()
almost_original_df.head()
# Zapisanie pliku csv z separatorem pipe "|" oraz z kodowaniem ascii

almost_original_df.to_csv(decimal = '|', encoding = 'ascii')