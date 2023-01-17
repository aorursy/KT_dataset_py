import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_set_df = pd.read_csv('../input/train_set.csv')

test_set_df = pd.read_csv('../input/test_set.csv')

tube_df = pd.read_csv('../input/tube.csv')

bill_of_materials_df = pd.read_csv('../input/bill_of_materials.csv')

specs_df = pd.read_csv('../input/specs.csv')
train_set_df['tube_assembly_id'] = pd.Series(train_set_df['tube_assembly_id'], dtype = 'category')

train_set_df['supplier'] = pd.Series(train_set_df['supplier'], dtype = 'category')

train_set_df['bracket_pricing'] = pd.Series(train_set_df['bracket_pricing'], dtype = 'category')
train_set_df.info()
df = pd.to_datetime(train_set_df['quote_date'])

df = df.to_frame()

df['first_date'] = pd.Timestamp('19820922')

train_set_df['quote_date'] = (df['quote_date'] - df['first_date']).dt.days
sns.pairplot(train_set_df, markers= 'o',diag_kind="kde",diag_kws=dict(shade=True),  

              plot_kws=dict(s=50, edgecolor="cyan", linewidth=0.8))
#Now lets fix the tube data frame. 

#By making some coloumns as catagorical.

tube_df['tube_assembly_id'] = pd.Series(tube_df['tube_assembly_id'], dtype = 'category')

tube_df['material_id'] = pd.Series(tube_df['material_id'], dtype = 'category')

tube_df['end_a_1x'] = pd.Series(tube_df['end_a_1x'], dtype = 'category')

tube_df['end_a_2x'] = pd.Series(tube_df['end_a_2x'], dtype = 'category')

tube_df['end_x_1x'] = pd.Series(tube_df['end_x_1x'], dtype = 'category')

tube_df['end_x_2x'] = pd.Series(tube_df['end_x_2x'], dtype = 'category')

tube_df['end_a'] = pd.Series(tube_df['end_a'], dtype = 'category')

tube_df['end_x'] = pd.Series(tube_df['end_x'], dtype = 'category')
tube_df.describe()
#Lets explore the numerical columns of the tube data frame

#Diameter

diameter = tube_df['diameter']
diameter.hist(bins = 30, figsize=(10,6), color = 'firebrick', alpha = 0.75)

plt.xlabel('Diameter')

plt.ylabel('Count')

#Maximun number of diameters of the tube is between 0-30
#Wall 

wall = tube_df['wall']
wall.hist(figsize=(10,6), bins = 15, color = 'olive', alpha = 0.90)

plt.xlabel('Wall')

plt.ylabel('Count')

#Maximum off the wall sizes are between 0.5-1.5
#Length

tube_df['length'].hist(figsize = (10,6), bins = 50, color='tomato')

plt.xlabel('Length')

plt.ylabel('Count')

#Here we see that shorter tubes less than 250 are dominant.
tube_df.describe(include = 'all')

# num_boss, num_bracket, and other are mostly 0, with small maximun values.
#Lets see the pairplot between these three components(diameter,length,wall)

sns.pairplot(tube_df[['diameter', 'wall', 'length']], aspect = 2, markers= 's',diag_kind="kde", 

             diag_kws=dict(shade=True),  plot_kws=dict(s=50, edgecolor="turquoise", linewidth=0.8))
tube_df.corr()
#Now lets analyze the bill of materials dataframe.

bill_of_materials_df.head(10)
#Seperating the quantity and component of tubes and analysing both.



bill_comp_types_df = bill_of_materials_df.iloc[:,[1,3,5,7,9,11,13,15]]
bill_comp_types_logical_df = ~bill_comp_types_df.isnull()
component_series = bill_comp_types_logical_df.sum(axis = 1)
component_series.hist(figsize=(10,6), color = 'darkcyan',alpha=0.75)

plt.xlabel('Components')

plt.ylabel('Count')

#Here we see that most of the components are allign on the 2 types.
(sum(component_series == 0) + sum(component_series == 1) + sum(component_series == 2) \

 + sum(component_series == 3))/float(component_series.count())

#Here we see that almost 97% of tube have components in range 0-3 
#Now lets check same for the 

bill_comp_quants_df = bill_of_materials_df.iloc[:, [2,4,6,8,10,12,14,16]]
quants_series = bill_comp_quants_df.sum(axis = 1)
quants_series.hist(bins = 15, figsize=(10,6), color = 'orchid', alpha =1, label= 'Quant_Series')

plt.xlabel('Quantities')

plt.ylabel('Count')

plt.legend()
(sum(quants_series == 0) + sum(quants_series == 1) + sum(quants_series == 2) \

 + sum(quants_series == 3) + sum(quants_series == 4)) / float(np.shape(bill_of_materials_df)[0])

#98% of all tube assemblies have 0-4 total components
specs_only_df = specs_df.iloc[:, 1:11]
specs_logical_df = ~specs_only_df.isnull()

spec_totals = specs_logical_df.sum(axis = 1)
spec_totals.hist(figsize=(10,6), color = "darkorchid", bins = 10) 

plt.xlabel('Specifications')

plt.ylabel('Count')

plt.legend()

#Almost half of all tube assemblies have exactly 2 types of components
#Now lets join our data frames into a single dataframe.

#1st Applying Left join on train set dataframe and tube dataframe on tube_assembly_id.

join_1 = pd.merge(train_set_df, tube_df, left_on = 'tube_assembly_id', right_on = 'tube_assembly_id',

                    how='left', sort=False)
join_1.head()
#2nd Applying Left join on join_1 dataframe(train_set_df and tube_df) and specification dataframe(specs_totals). 

specs_with_totals_df = specs_df.copy()

specs_with_totals_df['spec_totals'] = spec_totals



join_2 = join_1.merge(specs_with_totals_df[['tube_assembly_id', 'spec_totals']])
join_2.head()
#3rd Applying left join with join_2(train, tube, and spec_totals) with bill_of_materials_summary_df

bill_of_materials_summary_df = bill_of_materials_df.copy()

bill_of_materials_summary_df['type_totals'] = component_series

bill_of_materials_summary_df['component_totals'] = quants_series



join_3 = join_2.merge(bill_of_materials_summary_df[['tube_assembly_id', 'type_totals', 'component_totals']])

join_3.head()
#Now lets make heatmaps and cluster maps of the correlations of the numerical colunms of join_3 data frame..

corr_df= join_3.corr()

plt.figure(figsize=(15,12))

sns.heatmap(corr_df, vmax = 1, vmin = -1, annot=True,linewidths=.5, cmap = 'winter')
cm = sns.clustermap(corr_df, vmax = 1, vmin = -1, annot=True,linewidths=.5, cmap = 'summer', figsize=(15,12))

cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), rotation=0)
#Lets plot the more correlated columns

#join_3.columns

more_correlated = pd.DataFrame(join_3.iloc[:,[7,9,10, 11, 12,13, 14,15,16,17,24,25]])
plt.figure(figsize=(12,8))

sns.heatmap(more_correlated.corr(), annot=True,  linewidths=.5, cmap = 'YlGnBu')
#cluster map on the more correlated columns

g = sns.clustermap(more_correlated.corr(), annot=True, cmap= 'summer', figsize=(12,10))

g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

sns.lmplot(x= 'diameter', y='bend_radius', data=join_3, fit_reg=False,hue = 'bracket_pricing', palette='Set1', col='end_a_1x',

          row = 'end_x_1x', x_jitter=.2)

#plt.title('Diameter of tube vs Bend_radius')
fig= plt.figure(figsize=(12,6))

sns.pointplot(x ='quantity', y = 'cost', data = join_3,color="#bb3f3f", markers="D")

plt.xlim((0,20))

plt.title('Mean(Cost) per Quantity')
sns.lmplot(x = 'quantity', y = 'cost', data = join_3, aspect=2, fit_reg=False, markers='o',

          x_jitter=0.5)

plt.xlim((0,50))

plt.title('Cost per volume quantity')

#Most of the quantities are in the range of 1-5

#And as the quanitity is increasing the cost is decreasing.
supplier= pd.Series(join_3['supplier'], dtype='category')
fig= plt.figure(figsize=(12,6))

sns.pointplot(x ='num_bends', y = 'cost', data = join_3,color="#bb3f3f", markers="D")

plt.xlim((0,20))

plt.title('Mean(Cost) per Bend')