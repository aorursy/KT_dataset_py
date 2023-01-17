# Initial library declarations

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



# Store Philadelphia's crime data

crime_data = pd.read_csv('../input/crime.csv')



# How large is data set?

print("Data dimensions: ")

print(crime_data.shape)
# Explore column names, max values, and min values

print("Column values:")

print(crime_data.columns.values)

print("\n")



pd.options.display.float_format = '{:,.2f}'.format

print("Data Set Description:")

print(crime_data.describe())
# Look at some distinct data

dc_dist_distinct = crime_data.groupby('Dc_Dist')['Dc_Dist'].count()

print(dc_dist_distinct)

print("\n")



ucr_text_distinct = crime_data.groupby(['UCR_General', 'Text_General_Code']).size()

print(ucr_text_distinct)
import warnings

warnings.filterwarnings("ignore", 'This pattern has match groups')



# Check dates and times for consistent formatting

def does_column_match_pattern(data_frame, column_name, pattern):

    ''' Returns true if every value of a data frame column matches a pattern

        data_frame = the pandas data frame to be checked

        column_name = name of column to be checked

        pattern = regular expression to be matched

    '''

    data = data_frame[column_name]

    return data.str.contains(pattern).all()



# Patterns copied from http://regexlib.com/DisplayPatterns.aspx?cattabindex=4&categoryId=5&AspxAutoDetectCookieSupport=1

date_time_re = '20\d{2}-((0[1-9])|(1[0-2]))-((0[1-9])|([1-2][0-9])|(3[0-1]))(\s)(([0-1][0-9])|(2[0-3])):([0-5][0-9]):([0-5][0-9])'

date_re = '20\d{2}-((0[1-9])|(1[0-2]))-((0[1-9])|([1-2][0-9])|(3[0-1]))'

time_re = '^(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(:([0-5]?[0-9]))?$'

print("Do all elements in columns match expected pattern?")

print("Dispatch_Date_Time: ", does_column_match_pattern(crime_data, 'Dispatch_Date_Time', date_time_re))

print("Dispatch_Date: ", does_column_match_pattern(crime_data, 'Dispatch_Date', date_re))

print("Dispatch_Time: ", does_column_match_pattern(crime_data, 'Dispatch_Time', time_re))

print("Hour: ", ((crime_data['Hour'] >= 0) & (crime_data['Hour'] <= 23)).all())

print("Month: ", does_column_match_pattern(crime_data, 'Month', '[0-9]{4}-[0-9]{2}'))
# Remove incidents from 2016 and 2017 since there are large gaps in the data

years_included = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

crime_data = crime_data[crime_data['Dispatch_Date_Time'].astype(str).str.startswith(tuple(years_included))]



# How large is data set now?

print("Data dimensions: ")

print(crime_data.shape)
# Exclude rows without a crime type or description

crime_data.dropna(subset=['UCR_General', 'Text_General_Code'], inplace = True)



# How large is data set now?

print("Data dimensions: ")

print(crime_data.shape)
# Check before merger

print("Before district merger:")

dc_dist_distinct = crime_data.groupby('Dc_Dist')['Dc_Dist'].count()

print(dc_dist_distinct)                             

print("\n")



# Merge districts

crime_data.loc[crime_data.Dc_Dist == 4, 'Dc_Dist'] = 3

crime_data.loc[crime_data.Dc_Dist == 23, 'Dc_Dist'] = 22

crime_data.loc[crime_data.Dc_Dist == 77, 'Dc_Dist'] = 12

crime_data.loc[crime_data.Dc_Dist == 92, 'Dc_Dist'] = 14



# Check the result of the merger

print("After district merger:")

dc_dist_distinct = crime_data.groupby('Dc_Dist')['Dc_Dist'].count()

print(dc_dist_distinct)

print("\n")
import re



# Regular expressions

block_re = '^(\d+)(\s)BLOCK(\s)(.+)(\s)(\w+)$'

block_missed_re = '^(\d+)(\s)(.+)(\s)(\w+)$'

block_street_missed_re = '^(\d+)(.+)$'

street_missed_re = '^(\d+)(\s)BLOCK(\s)(.+)$'

int_amp_re = '^(.+)(\s)(\w+)(\s)&(.+)(\s)(\w+)$'

int_slash_re = '^(.+)(\s)(\w+)(\s)(\/)(.+)(\s)(\w+)$'



# Location formatting function

def format_location_data(item):

    ''' Format data set's location block column (invoked via apply method)

        item = one item from location block column

    '''  

    # Initial trimming

    item = item.strip()

    

    # Remove extraneous first characters

    if (item[0] == '`' or item[0] == '/'):

        item = item[1:]

        

    # Remove item with no whitespace

    if (len(item.split()) == 1):

        item = 'None'

    

    # We will not look at establishment, only the street

    if ('@' in item):

        at_index = item.index('@')

        item = item[at_index+1:] # also remove '@'

        item = item.strip()



    # BLOCK and SLASH patterns accepted at this stage

    if (re.fullmatch(block_re, item) or re.fullmatch(int_slash_re, item)):

        pass

    

    elif (re.fullmatch(street_missed_re, item)):

        # Add default street type

        item += " ST"

    

    elif (re.fullmatch(block_missed_re, item)):

        # Add BLOCK after number

        tokens = item.split()

        item = tokens[0] + " BLOCK"

        

        # Reconstruct string

        for i in range(1,len(tokens)):

            item = item + " " + tokens[i]

    

    elif (re.fullmatch(block_street_missed_re, item)):

        # Add BLOCK after number

        tokens = item.split()

        item = tokens[0] + " BLOCK"

        

        # Reconstruct string

        for i in range(1,len(tokens)):

            item = item + " " + tokens[i]

            

        # Add default street type

        item += " ST"

    

    elif (re.fullmatch(int_amp_re, item)):

        # Replace ampersand

        item = item.replace("&", "/")

    

    return item



# Street Types

STREET_TYPE_EXPECTED = ["AV", "BLVD", "CIR", "CT", "DR", "LN", "PKWY", "PL", "RD", "ROW", "ST", "TER", "WAY"]



STREET_TYPE_MAPPING = { "AVE": "AV",

                        "AVENUE": "AV",

                        "AVE": "AV",

                        "BLD": "BLVD",

                        "BLV": "BLVD",

                        "BDV": "BLVD",

                        "BOULEVARD": "BLVD",

                        "CI": "CIR",

                        "CIRCLE": "CIR",

                        "CRT": "CT",

                        "COURT": "CT",

                        "DRIVE": "DR",

                        "LANE": "LN",

                        "PKY": "PKWY",

                        "PWY": "PKWY",

                        "PARKWAY": "PKWY",

                        "PLA": "PL",

                        "PLACE": "PL",

                        "RDS": "RD",

                        "ROAD": "RD",

                        "STR": "ST",

                        "STT": "ST",

                        "STREET": "ST",

                        "TRCE": "TER",

                        "WA": "WAY"

                      }



# Street Type Formatting function

def format_street_type(item):

    ''' Format street type in data set's location block column (invoked via apply method)

        This function is invoked agter format_location_data

        item = one item from location block column

    '''

    # BLOCK pattern

    if (re.fullmatch(block_re, item)):

        tokens = item.split()

    

        # Check the hundred block

        if (tokens[0].isdigit()):

            # Check the hundred block

            tokens = item.split()

            block_num = int(tokens[0])

            block_num = (block_num // 100) * 100

            item = str(block_num)

        else:

            item = tokens[0]

            

        # Keep BLOCK and name of street

        item = item + " " + tokens[1] + " " + tokens[2]

    

        # Check remaining tokens for street type

        for i in range(3, len(tokens)):

            if (tokens[i] in STREET_TYPE_MAPPING):

                item = item + " " + STREET_TYPE_MAPPING[tokens[i]]

                break

            elif (tokens[i] in STREET_TYPE_EXPECTED):

                item = item + " " + tokens[i]

                break

            else:

                item = item + " " + tokens[i]

    

    # SLASH pattern

    elif (re.fullmatch(int_slash_re, item)):

        street_names = item.split('/')

        

        # Go through each part

        this_part = ""

        for i in range(0, len(street_names)):

            if (i>0):

                # Previous part

                item = this_part + " / "

                this_part = ""

                

            # Initial trimming

            street_names[i].strip()

            

            # Split this part

            tokens = street_names[i].split()

            

            # Empty item?

            if (len(tokens) == 0):

                break

        

            # Keep first token (for case like STREET RD)

            this_part = this_part + tokens[0]

            

            # Check remaining tokens for street type

            for j in range(1, len(tokens)):

                if (tokens[j] in STREET_TYPE_MAPPING):

                    this_part = this_part + " " + STREET_TYPE_MAPPING[tokens[j]]

                    break

                elif (tokens[j] in STREET_TYPE_EXPECTED):

                    this_part = this_part + " " + tokens[j]

                    break

                else:

                    this_part = this_part + " " + tokens[j]

            

        # Last part

        item = item + this_part



    return item



# Standardize the location data

crime_data['Location_Block'] = crime_data['Location_Block'].apply(format_location_data)



# Dump items with no information

crime_data = crime_data[crime_data['Location_Block'] != 'None']



# Standardize street type

crime_data['Location_Block'] = crime_data['Location_Block'].apply(format_street_type)



# How large is data set now?

print("Data dimensions: ")

print(crime_data.shape)
# Try groupby to remove duplicates

crime_data = crime_data.drop_duplicates(['Dispatch_Date_Time', 'Location_Block', 'UCR_General', 'Text_General_Code'])



# How large is data set now?

print("Data dimensions: ")

print(crime_data.shape)
def get_year(year_month_item):

    ''' Apply function to add only the year as a field in the data set'''

    return year_month_item[:4]



def get_month(year_month_item):

    ''' Apply function to add only the month as a field in the data set'''

    return year_month_item[5:]



# See if we can get year, month as separate fields in dataset - easier to plot these!

crime_data['Year'] = crime_data['Month'].apply(get_year)

crime_data['Actual_Month'] = crime_data['Month'].apply(get_month)
crime_type_dict = {100.0: 'Homicide',

                   200.0: 'Rape',

                   300.0: 'Robbery',

                   400.0: 'Aggravated Assault',

                   500.0: 'Burglary',

                   600.0: 'Theft',

                   700.0: 'Motor Vehicle Theft',

                   800.0: 'Other Assaults',

                   900.0: 'Arson',

                   1000.0: 'Forgery and Counterfeiting',

                   1100.0: 'Fraud',

                   1200.0: 'Embezzlement',

                   1300.0: 'Receiving Stolen Property',

                   1400.0: 'Vandalism',

                   1500.0: 'Weapon Violations',

                   1600.0: 'Prostitution',

                   1700.0: 'Other Sex Offenses',

                   1800.0: 'Drug Violations',

                   1900.0: 'Gambling Violations',

                   2000.0: 'Offenses against Families',

                   2100.0: 'Driving Under the Influence',

                   2200.0: 'Liquor Law Violations',

                   2300.0: 'Public Drunkenness',

                   2400.0: 'Disorderly Conduct',

                   2500.0: 'Vagrancy/Loitering',

                   2600.0: 'All Other Offenses'}



def replace_crime_type(ucr_general):

    '''Replace UCR General code with crime type'''

    if (ucr_general in crime_type_dict):

        ucr_general = crime_type_dict[ucr_general]

    return ucr_general



crime_data['UCR_General'] = crime_data['UCR_General'].apply(replace_crime_type)
# What parts of the city have the most crime?

dist_bins = np.array([1, 2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 18, 19, 22, 24, 25, 26, 35, 39])

sns.countplot(crime_data['Dc_Dist'], color='skyblue')

plt.title("Philadelphia Crime Counts by Police District (2006-2015)")

plt.xlabel("Police District")

plt.ylabel("Incident Count")

plt.show()
# Which crimes are most frequently committed?

ucr_bins = np.arange(50,2650,100)

sns.countplot(crime_data['UCR_General'], color='skyblue')

plt.title("Philadelphia Crime Counts by Category (2006-2015)")

plt.xlabel("UCR General Code")

plt.ylabel("Incident Count")

plt.xticks(rotation=90)

plt.show()
# How has crime changed over time?

sns.countplot(crime_data['Year'], color='skyblue')

plt.title("Philadelphia Crime Counts by Year")

plt.xlabel("Year")

plt.ylabel("Incident Count")

plt.show()
# What times of year do we see the most crimes committed?

sns.countplot(crime_data['Actual_Month'], color='skyblue')

plt.title("Philadelphia Crime Counts by Month (2006-2015)")

plt.xlabel("Month")

plt.ylabel("Incident Count")

plt.show()
# What times of day do we see the most crimes committed?

sns.countplot(crime_data['Hour'], color='skyblue')

plt.title("Philadelphia Crime Counts by Hour (2006-2015)")

plt.xlabel("Hour")

plt.ylabel("Incident Count")

plt.show()
## Look at most frequent streets and intersections - where are these located in the city?

location_distinct = crime_data.groupby(['Location_Block', 'Dc_Dist']).size()

location_distinct.sort_values(ascending=False, inplace=True)

print(location_distinct[:30])
# Crime types by police district

crime_data_type_dist = crime_data.groupby(['UCR_General', 'Dc_Dist']).size()

crime_data_type_dist = crime_data_type_dist.apply(int)

crime_data_type_dist_df = crime_data_type_dist.to_frame()

crime_data_type_dist_pt = pd.pivot_table(crime_data_type_dist_df, index=['UCR_General'], columns=['Dc_Dist'])[0] # Get rid of '0 column'



# Heatmap representation

f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data=crime_data_type_dist_pt, annot=True, linewidths=0.1, fmt='g', cmap="YlOrRd", ax=ax)

plt.title("Philadelphia Crime Types By Police District (2006-2015)")

plt.xlabel("Police District")

plt.ylabel("Crime Type Code")

plt.show()
# Crime types by year

crime_data_type_year = crime_data.groupby(['UCR_General', 'Year']).size()

crime_data_type_year = crime_data_type_year.apply(int)

crime_data_type_year_df = crime_data_type_year.to_frame()

crime_data_type_year_pt = pd.pivot_table(crime_data_type_year_df, index=['UCR_General'], columns=['Year'])[0] # Get rid of '0 column'



# Heatmap representation

f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data=crime_data_type_year_pt, annot=True, linewidths=0.1, fmt='g', cmap="YlOrRd", ax=ax)

plt.title("Philadelphia Crime Types Per Year")

plt.xlabel("Year")

plt.ylabel("Crime Type Code")

plt.show()
# Police district crime counts by year

crime_data_dist_year = crime_data.groupby(['Dc_Dist', 'Year']).size()

crime_data_dist_year = crime_data_dist_year.apply(int)

crime_data_dist_year_df = crime_data_dist_year.to_frame()

crime_data_dist_year_pt = pd.pivot_table(crime_data_dist_year_df, index=['Dc_Dist'], columns=['Year'])[0] # Get rid of '0 column'



# Heatmap representation

f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data=crime_data_dist_year_pt, annot=True, linewidths=0.1, fmt='g', cmap="YlOrRd", ax=ax)

plt.title("Philadelphia Crime Counts By Police District Per Year")

plt.xlabel("Year")

plt.ylabel("Police District")

plt.show()
# Sample districts and common crime types

police_districts = [3, 17, 25, 15, 19]



common_crime_types = ['Theft', 'Vandalism', 'Motor Vehicle Theft', 'Drug Violations', 'Other Assaults', 'All Other Offenses']



# ColorBrewer color scheme

line_colors = ['#d53e4f', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd']



# Loop through the police districts

for district in police_districts:

    crime_data_dist = crime_data[crime_data['Dc_Dist'] == district]

    crime_data_dist = crime_data_dist.loc[(crime_data_dist['UCR_General']).isin(common_crime_types)]

    crime_data_dist = crime_data_dist.groupby(['UCR_General', 'Year']).size()

    crime_data_dist = crime_data_dist.apply(int)

    crime_data_dist_df = crime_data_dist.to_frame()

    crime_data_dist_pt = pd.pivot_table(crime_data_dist_df, index=['Year'], columns=['UCR_General'])[0] # Get rid of '0 column'



    ax = crime_data_dist_pt.plot(kind = 'line', title='Police District ' + str(district) + ' Common Crime Counts by Year')

    ax.legend(bbox_to_anchor=(1.0, 1.0))

    ax.patch.set_facecolor('white')

    ax.grid(color='gray')

    plt.rcParams["axes.prop_cycle"] = plt.cycler('color', line_colors)

    plt.show()
# Sample districts and common crime types

worst_districts = [12, 15, 22, 24, 25, 35]



common_crime_types = ['Aggravated Assault', 'Homicide', 'Motor Vehicle Theft', 'Rape']



# ColorBrewer color scheme

line_colors = ['#41b6c4', '#e31a1c', '#fecc5c', '#fd8d3c']



# Loop through the police districts

for district in worst_districts:

    crime_data_dist = crime_data[crime_data['Dc_Dist'] == district]

    crime_data_dist = crime_data_dist.loc[(crime_data_dist['UCR_General']).isin(common_crime_types)]

    crime_data_dist = crime_data_dist.groupby(['UCR_General', 'Year']).size()

    crime_data_dist = crime_data_dist.apply(int)

    crime_data_dist_df = crime_data_dist.to_frame()

    crime_data_dist_pt = pd.pivot_table(crime_data_dist_df, index=['Year'], columns=['UCR_General'])[0] # Get rid of '0 column'



    ax = crime_data_dist_pt.plot(kind = 'line', title='Police District ' + str(district) + ' Common Crime Counts by Year')

    ax.legend(bbox_to_anchor=(1.0, 1.0))

    ax.patch.set_facecolor('white')

    ax.grid(color='gray')

    plt.rcParams["axes.prop_cycle"] = plt.cycler('color', line_colors)

    plt.show()