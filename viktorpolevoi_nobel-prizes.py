import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid", palette="Paired")
plt.rcParams['figure.dpi'] = 120
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/nobel-prizes/nobel.csv')
data[:5]
sns.heatmap(data.isnull(), yticklabels=False, cmap="viridis")
by_year = data.year.value_counts()
by_year = by_year[by_year > 11]
by_year
top_10_county = data.birth_country.value_counts()[:10].plot.bar()
for p in top_10_county.patches:
    top_10_county.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 4),
                         textcoords = 'offset points')
plt.xlabel('Top 10 countries')
top_10_organizations = data.organization_name.value_counts()[:10].plot.bar()
for p in top_10_organizations.patches:
    top_10_organizations.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 4),
                         textcoords = 'offset points')
plt.xlabel('Top 10 organizations')
plt.figure(figsize=(7,5))
category_count_total = data.category.value_counts()
category_count_total_pie = plt.pie(category_count_total, labels=category_count_total.index, autopct='%1.1f%%')
yr_md = data.year.median()
yr_md
data['Year group'] = data.year.map(lambda year: 0 if year<yr_md else 1)
category_by_year_gr = sns.countplot(x="category", hue="Year group", data=data)
plt.legend([f'Before {int(yr_md)}', f'After {int(yr_md)}'], bbox_to_anchor=(1.0, 0.7))
for p in category_by_year_gr.patches:
    category_by_year_gr.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 5),
                         textcoords = 'offset points')
laureate_2x = data[data.laureate_type == 'Individual'].full_name.value_counts()
laureate_2x = laureate_2x[laureate_2x > 1]
laureate_2x
data[data.full_name.isin(laureate_2x.index)].sort_values(by=['full_name'])
laureate_org = data[data.laureate_type == 'Organization'].full_name.value_counts()
laureate_org = laureate_org[laureate_org > 1]
laureate_org
data[data.full_name.isin(laureate_org.index)].sort_values(by=['full_name'])
data[data.birth_country == 'Ukraine']
