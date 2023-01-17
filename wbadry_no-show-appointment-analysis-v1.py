# Libraries to be used
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# More display width
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Print libraries version by date of submission
print('Libraries version by date of submission')
print('pandas\t\t : {:}'.format(pd.__version__))
print('numpy\t\t : {:}'.format(np.__version__))
print('seaborn\t\t : {:}'.format(sns.__version__))
print('matplotlib\t : {:}'.format(matplotlib.__version__))
# Read csv file
df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')

# Show few dataset size
print('The dataset has {:} records and {:} columns'.format(df.shape[0],df.shape[1]))
for i,c in enumerate(df.columns):
    print('{:} : {:}'.format(i+1,c))
# Show number of filled records
df.info()
print('Number of null cells in the dataset : {:} cells'.format(df.isnull().any().sum()))
df.head()
df.describe()
# Drop insignificant columns
df.drop(columns=['PatientId', 'AppointmentID'],axis=1,inplace=True)

# Check if dropped
df.head()
# Check if there are duplicate records
print('Found {:} duplated records'.format(df.duplicated().sum()))

# Number of records before
print('Number of records with duplicates : {:}'.format(df.shape[0]))

# drop duplicated records
df.drop_duplicates(keep='first',inplace=True)

# Check remaining number of records
print('Number of records after duplicates removed : {:}'.format(df.shape[0]))
# Remove incorrect age entries
incorrect_age_entry = df.Age[df.Age < 0].count()

# Show records found
if (incorrect_age_entry == 1):
    print('Found {:d} record with incorrect age entry'.format(incorrect_age_entry))
else:
    print('Found {:d} records with incorrect age entry'.format(incorrect_age_entry))
    
# Remove the incorrect records from the dataframe
df = df[df.Age >= 0]

# Check remaining number of records
print('Number of records after incorrect age entries removed : {:}'.format(df.shape[0]))

# Swap dates if ScheduledDay > AppointmentDay 
incorrect_date_entry = df.Gender[df.AppointmentDay < df.ScheduledDay].count()

if incorrect_date_entry >= 1 :
    # Swap column values if reservation (scheduled) day is later than appointment day
    df.AppointmentDay, df.ScheduledDay = np.where(df.AppointmentDay < df.ScheduledDay, 
                                                  [df.ScheduledDay, df.AppointmentDay], 
                                                  [df.AppointmentDay, df.ScheduledDay])

    # Show records found
if (incorrect_date_entry == 1):
    print('{:d} record with with invalid dates entry was corrected'.format(incorrect_date_entry))
else:
    print('Found {:d} records with invalid dates entry was corrected'.format(incorrect_date_entry))
    
# Columns to be renamed
df.rename(columns = {'HiperTension': 'Hypertension',
                     'Handcap': 'Handicap'}, inplace = True)

# Get columns names after renaming
print('Column names:')
for i,c in enumerate(df.columns):
    print('{:} : {:}'.format(i+1,c))
# Convert date columns into datetime datatype
df['ScheduledDay']= pd.to_datetime(df['ScheduledDay'],utc=False).dt.normalize()
df['AppointmentDay']= pd.to_datetime(df['AppointmentDay'],utc=False).dt.normalize()
df.info()
# Validation of changes
df.head()
# Another validation
df.tail()
# Create IsSameDay column
df['IsSameDay'] = (df.AppointmentDay == df.ScheduledDay).astype(int)

# Create DaysDifference column
df['DaysDifference'] = df.AppointmentDay - df.ScheduledDay
df.DaysDifference = df.DaysDifference.apply(lambda x: x.days)

# Move column to a better place
col_name="DaysDifference"
first_col = df.pop(col_name)
df.insert(3, col_name, first_col)

col_name="IsSameDay"
first_col = df.pop(col_name)
df.insert(4, col_name, first_col)

# Check results
df.tail()
# Rename column 
df.rename(columns={'No-show':'IsAttended'},inplace=True )

# Swap values
df.IsAttended  = df.IsAttended.apply(lambda x: 'Yes' if x=='No' else 'No')

# Check reult
df.head()
# Get number of each gender
labels = df.Gender.unique()
total_females = df['Gender'].value_counts()[0]
total_males = df['Gender'].value_counts()[1]
total = total_females + total_males

# Get percentage of each gender with respect to tatal number patients
female_percentage  = (total_females / total) * 100
male_percentage  = (total_males / total) * 100
count = [total_females,total_males]

# Visualize ratio
fig = plt.figure()
plt.pie(x= count ,labels=labels, autopct='%1.1f%%',radius=1.2,
        colors  = ['m','y'],shadow=True,explode=(0.1,0),startangle=220)
plt.title('Females vs Males',fontsize=14);
print('The dataset has {:} female records'.format(total_females))
print('The dataset has {:} male records'.format(total_males))
# Age distribution of each gender in the dataset
df_age = df.groupby('Gender')['Age'].value_counts().to_frame()

# Organize result into columns
df_age.rename(columns={'Age': 'Count'}, inplace=True)
df_age.reset_index(inplace=True)

# Age into categories
df_age['AgeByDecade'] = pd.cut(x=df_age.Age, bins=[0,2, 11, 17, 64, 150], 
                                  labels=['Infants','Children', 'Teens', 'Adults', 'Elderly'])

# Plot the distribution
ax = sns.barplot(x='AgeByDecade', y='Count', data=df_age, hue='Gender')
ax.set_title('Males vs Females Dataset Distribution');
# Get number of females came to appointment
femelaes_attended = (df.Gender[(df.Gender=='F') & (df.IsAttended=='Yes')].count() / total_females) * 100
melaes_attended = (df.Gender[(df.Gender=='M') & (df.IsAttended=='Yes')].count() / total_males) * 100

ax = sns.barplot(x=['Females','Males'],y=[femelaes_attended, melaes_attended])
ax.set_xlabel('Gender')
ax.set_ylabel('%')
ax.set_title('Percentage of attended males and females per own gender')

print("Who attended the most?")
print('{:1.1f} of total females attended the appointment'.format(femelaes_attended))
print('{:1.1f} of total males attended the appointment'.format(melaes_attended))
print()
# Get the query
df_ages_attended = df.groupby(['Gender','IsAttended'])['Age'].value_counts().to_frame()

# Organize result into columns
df_ages_attended.rename(columns={'Age': 'Count'}, inplace=True)
df_ages_attended.reset_index(inplace=True)

# Age into categories
df_ages_attended['AgeByDecade'] = pd.cut(x=df_ages_attended.Age, bins=[-5,2, 11, 17, 64, 150], 
                                  labels=['Infants','Children', 'Teens', 'Adults', 'Elderly'])

# Plot the distribution
ax = sns.barplot(x='AgeByDecade', y='Count', data=df_ages_attended, hue='IsAttended')
ax.set_title('Attendance to appointments by Age categories');
# Function  to plot insights of patients' attendance with diseases
def attendance_with_disease(df, logical_disease_field, printable_name):
    '''Plots and shows patients' attendance of appointments
    with specific disease
    
    Parameters:
    df : Pandas dataframe of patient dataset
    logical_disease_field:  Disease column with:
                            0 : not sick
                            1 : sick
    printable_name : Friendly name of the field
    '''
    # Get diabetes data
    attended = df[logical_disease_field][(df[logical_disease_field]==1) & 
                                                  (df.IsAttended=='Yes')].count()
    total  = df[logical_disease_field].count()
    missed = total - attended

    # Plot findings
    ax = sns.barplot(x=['Attended','Missed'],y=[(attended / total)*100,
                                            (missed / total)*100],palette= 'rocket')
    ax.set_xlabel('Status')
    ax.set_ylabel('%')
    ax.set_title('Percentage of attended {:} to appointments'.format(printable_name));
    print('Does {:} affect attendance percentage?'.format(printable_name))
    print('Only {:1.1f} % of {:} attended their appointment'.format((attended / total)*100, printable_name))
    print()
# Get insights of attendance for diabetes
attendance_with_disease(df, 'Diabetes', 'diabetes')
# Get insights of attendance for alcoholics
attendance_with_disease(df, 'Alcoholism', 'alcoholics')
# Get data of patients attended after reservation
df_same_day_attended = df.groupby(['IsAttended']).IsSameDay.value_counts().to_frame()

# Organize result into columns
df_same_day_attended.rename(columns={'IsSameDay': 'Count'}, inplace=True)
df_same_day_attended.reset_index(inplace=True)

# Plot the result
ax = sns.barplot(x='IsSameDay', y='Count', data=df_same_day_attended, hue='IsAttended')
ax.set_title('Attendance to appointments by Same Day Registration');
# Get categorical attenace by nigbourhood
df_neigbour_attended  = df.groupby('IsAttended').Neighbourhood.value_counts().to_frame()

# Organize result into columns
df_neigbour_attended.rename(columns={'Neighbourhood': 'Count'}, inplace=True)
df_neigbour_attended.reset_index(inplace=True)

# Get the top 5 neigborhood
df_n_top5_yes = df_neigbour_attended[df_neigbour_attended['IsAttended']=='Yes'].head(5)
df_n_top5_yes.reset_index()
df_n_top5_no = df_neigbour_attended[df_neigbour_attended['IsAttended']=='No'].head(5) 
df_n_top5_yes.reset_index()

# Merge both into one dataframe
df_top_5 = pd.merge(left=df_n_top5_yes,right=df_n_top5_no,how='outer',on=['IsAttended','Count','Neighbourhood'])

# Visualize result
ax = sns.barplot(x='Neighbourhood', y='Count', data=df_top_5, hue='IsAttended')
ax.set_title('Top 5 neighbourhoods for attendance/absence of appointments')
plt.xticks(rotation=45);