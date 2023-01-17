import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

% matplotlib inline
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df.head()
df.rename(columns={'PatientId': 'PatientID', 'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
df.info()
sum(df.duplicated())
# Converting data type
df['PatientID'] = df['PatientID'].astype('int64')
# Check to see if the conversion was successful
df.head(1)
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df.head(1)
df['AppointmentDay'].unique()
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df.head(1)
# Check the values in the 'Age' column
np.sort(df['Age'].unique())
df[df['Age'] == -1].count()['Age']
# Find out which row contains the value -1 in the Age column.
df.loc[df['Age'] == -1]
# Drop row 99832, which carries the value -1.
df.drop(index=99832, inplace=True)
# Check to see if the row has been dropped successfully.
np.sort(df['Age'].unique())
# Print the unique values in the columns
print('Scholarship:')
print(df['Scholarship'].unique())
print('Hypertension:')
print(df['Hypertension'].unique())
print('Diabetes:')
print(df['Diabetes'].unique())
print('Alcoholism:')
print(df['Alcoholism'].unique())
print('Handicap:')
print(df['Handicap'].unique())
print('SMS_received:')
print(df['SMS_received'].unique())
# Count the number of patients of each gender
df['Gender'].value_counts()
plt.pie([71840, 38687], labels = ['Female', 'Male'], colors = ['lightcoral', 'lightskyblue'])
plt.axis('equal')
plt.title('Number of Female and Male Patients');
# Group patients by gender and whether they showed up
gender_counts = df.groupby(['Gender', 'No-show']).count()['Age']
gender_counts
female_show_up_proportion = gender_counts['F','No'] / df['Gender'].value_counts()['F']
female_show_up_proportion
male_show_up_proportion = gender_counts['M', 'No'] / df['Gender'].value_counts()['M']
male_show_up_proportion
plt.bar([1, 2],[female_show_up_proportion, male_show_up_proportion], width=0.6, color = ['seagreen'])
plt.xlabel('Gender')
plt.ylabel('Proportion')
plt.xticks([1, 2], ['Female', 'Male'])
plt.title('Proportion of Patients that Showed Up by Gender');
df['Age'].describe()
# Set bin edges that will be used to "cut" the data into age groups
bin_edges = [0, 20, 40, 60, 80, 115]
# Create labels for each age group
bin_names = ['<20', '20-39', '40-59', '60-79', '>=80']

# Create the 'AgeGroup' column
df['AgeGroup'] = pd.cut(df['Age'], bin_edges, labels=bin_names, right=False, include_lowest=True)

# Check for successful creation of the column
df.head()
age_group_counts = df.groupby(['AgeGroup', 'No-show']).count()['Age']
age_group_counts
# Calculate proportion of patients that show up according to age groups
below_twenty = age_group_counts['<20', 'No'] / df['AgeGroup'].value_counts()['<20']
twenty_to_thirty_nine = age_group_counts['20-39', 'No'] / df['AgeGroup'].value_counts()['20-39']
forty_to_fifty_nine = age_group_counts['40-59', 'No'] / df['AgeGroup'].value_counts()['40-59']
sixty_to_seventy_nine = age_group_counts['60-79', 'No'] / df['AgeGroup'].value_counts()['60-79']
eighty_and_above = age_group_counts['>=80', 'No'] / df['AgeGroup'].value_counts()['>=80']
# Plot the graph accordingly
proportions = [below_twenty, twenty_to_thirty_nine, forty_to_fifty_nine, sixty_to_seventy_nine, eighty_and_above]
plt.bar([1, 2, 3, 4, 5], proportions, width=0.3)
plt.xlabel('Age Group')
plt.ylabel('Proportion')
plt.xticks([1, 2, 3, 4, 5], ['<20', '20-39', '40-59', '60-79', '>=80'])
plt.title('Proportion of Patients that Showed Up According to Age Groups');
scholarship_counts = df.groupby(['Scholarship', 'No-show']).count()['Age']
scholarship_counts
not_enrolled_show_up_proportion = scholarship_counts[0, 'No'] / df['Scholarship'].value_counts()[0]
not_enrolled_show_up_proportion
enrolled_show_up_proportion = scholarship_counts[1, 'No'] / df['Scholarship'].value_counts()[1]
enrolled_show_up_proportion
plt.bar([1, 2], [not_enrolled_show_up_proportion, enrolled_show_up_proportion], width=0.6, color=['seagreen'])
plt.xlabel('Enrolment in the Bolsa Família Program')
plt.ylabel('Proportion')
plt.xticks([1, 2], ['Not enrolled', 'Enrolled'])
plt.title('Proportion of Patients that Showed Up, According to Enrolment in the Program');
hypertension_counts = df.groupby(['Hypertension', 'No-show']).count()['Age']
hypertension_counts
non_hypertensive = hypertension_counts[0, 'No'] / df['Hypertension'].value_counts()[0]
hypertensive = hypertension_counts[1, 'No'] / df['Hypertension'].value_counts()[1]
diabetes_counts = df.groupby(['Diabetes', 'No-show']).count()['Age']
diabetes_counts
non_diabetic = diabetes_counts[0, 'No'] / df['Diabetes'].value_counts()[0]
diabetic = diabetes_counts[1, 'No'] / df['Diabetes'].value_counts()[1]
alcoholism_counts = df.groupby(['Alcoholism', 'No-show']).count()['Age']
alcoholism_counts
non_alcoholic = alcoholism_counts[0, 'No'] / df['Alcoholism'].value_counts()[0]
alcoholic = alcoholism_counts[1, 'No'] / df['Alcoholism'].value_counts()[1]
ind = np.array([1, 2, 3])
width = 0.3
plt.bar(ind, [non_hypertensive, non_diabetic, non_alcoholic], width=width, color='seagreen', label='Without the condition')
plt.bar(ind+width, [hypertensive, diabetic, alcoholic], width=width, color='brown', label='With the condition')

plt.xlabel('Conditions')
plt.ylabel('Proportion that showed up')

locations = ind + width / 2
plt.xticks(locations, ['Hypertension', 'Diabetes', 'Alcoholism'])
plt.title('Proportion of Patients that Showed Up, According to Conditions')

plt.legend(bbox_to_anchor=(1,1));
handicap_counts = df.groupby(['Handicap', 'No-show']).count()['Age']
handicap_counts
handicap_zero = handicap_counts[0, 'No'] / df['Handicap'].value_counts()[0]
handicap_zero
handicap_one = handicap_counts[1, 'No'] / df['Handicap'].value_counts()[1]
handicap_one
handicap_two = handicap_counts[2, 'No'] / df['Handicap'].value_counts()[2]
handicap_two
handicap_three = handicap_counts[3, 'No'] / df['Handicap'].value_counts()[3]
handicap_three
handicap_four = handicap_counts[4, 'No'] / df['Handicap'].value_counts()[4]
handicap_four
plt.bar([1, 2, 3, 4, 5], [handicap_zero, handicap_one, handicap_two, handicap_three, handicap_four], width=0.6)
plt.xlabel('Handicap Type')
plt.ylabel('Proportion')
plt.xticks([1, 2, 3, 4, 5], ['0', '1', '2', '3', '4'])
plt.title('Proportion of Patients that Showed Up by Handicap Type');
sms_counts = df.groupby(['SMS_received', 'No-show']).count()['Age']
sms_counts
no_sms_show_up_proportion = sms_counts[0, 'No'] / df['SMS_received'].value_counts()[0]
no_sms_show_up_proportion
sms_show_up_proportion = sms_counts[1, 'No'] / df['SMS_received'].value_counts()[1]
sms_show_up_proportion
plt.bar([1, 2], [no_sms_show_up_proportion, sms_show_up_proportion], width=0.6, color=['brown'])
plt.xlabel('Received SMS')
plt.ylabel('Proportion')
plt.xticks([1, 2], ['No', 'Yes'])
plt.title('Proportion of Patients that Showed Up: SMS vs No SMS');