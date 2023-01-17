import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
fruits = pd.DataFrame(data = {
    'Apples': [30],
    'Bananas': [21]
})
check_q1(fruits)
# Your code here
apples = [35, 41]
bananas = [21, 34]
sales = ['2017 Sales', '2018 Sales']

fruit_sales = pd.DataFrame(data = {
    'Apples': apples,
    'Bananas': bananas
}, index=sales)
check_q2(fruit_sales)
# Your code here
ingredients = ['Flour', 'Milk', 'Eggs', 'Spam']
amounts = ['4 cups', '1 cup', '2 large', '1 can']

dinner = pd.Series(data = amounts, index = ingredients, name = 'Dinner')
check_q3(dinner)
# Your code here 
wine = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
check_q4(wine)
# Your code here
pregnant = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name="Pregnant Women Participating")
check_q5(pregnant)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
# Your Code Here