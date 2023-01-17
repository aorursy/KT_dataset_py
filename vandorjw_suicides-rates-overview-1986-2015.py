import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/master.csv')

# Drop the columns I am not going to use.

df = df.drop([

    'country-year',

    'HDI for year', 

    ' gdp_for_year ($) ',

    'generation',

    'gdp_per_capita ($)',

    'population'], 

    axis=1)

male_female_suicides = df.groupby(['sex', 'year']).sum()['suicides_no']
suicides_by_year = df.groupby('year')['suicides_no'].agg('sum')
plt.plot(suicides_by_year, label='All')

plt.plot(male_female_suicides['male'], label='male')

plt.plot(male_female_suicides['female'], label='female')

plt.ylabel('Number of Suicides')

plt.legend(

    bbox_to_anchor=(1.04,0.5),

    loc="center left",

    borderaxespad=0

)

plt.xlim(right=2015)

plt.show()
age_dist_suicides = df.groupby(['age', 'year'])['suicides_no'].agg('sum')



plt.plot(age_dist_suicides['5-14 years'], label='5-14 years')

plt.plot(age_dist_suicides['15-24 years'], label='15-24 years')

plt.plot(age_dist_suicides['25-34 years'], label='25-34 years')

plt.plot(age_dist_suicides['35-54 years'], label='35-54 years')

plt.plot(age_dist_suicides['55-74 years'], label='55-74 years')

plt.plot(age_dist_suicides['75+ years'], label='75+ years')

plt.ylabel('Number of Suicides')

plt.legend(

    bbox_to_anchor=(1.04,0.5),

    loc="center left",

    borderaxespad=0

)

plt.xlim(right=2015)

plt.show()
country_data = df.groupby(['country', 'year'])['suicides_no'].agg('sum')
def country_graph(countries):

    for cntry in countries:

        plt.plot(country_data[cntry], label=cntry)



    plt.ylabel('Number of Suicides')

    plt.legend(

        bbox_to_anchor=(1.04,0.5),

        loc="center left",

        borderaxespad=0

    )

    plt.xlim(right=2015)

    plt.show()
country_suicide_sum = df.groupby('country')['suicides_no'].agg('sum')

country_suicide_sum.sort_values(ascending=False, inplace=True)
countries = list(country_suicide_sum[0:3].keys())

country_graph(countries)
countries = list(country_suicide_sum[3:6].keys())

country_graph(countries)
countries = list(country_suicide_sum[6:10].keys())

country_graph(countries)
countries = list(country_suicide_sum[15:20].keys())

country_graph(countries)