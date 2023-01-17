# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

pd.set_option('display.max_columns', 500) # show all columns

pd.options.display.max_rows = 200 # show 200 rows

        

import warnings

warnings.filterwarnings("ignore") #remove warning messages during csv import



# Any results you write to the current directory are saved as output.



symbols = '!@#$%^&*()_+[]-–'

letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

numbers = '0123456789'



pd.options.display.float_format = '{:.2f}'.format



%matplotlib inline
raw_data_original = pd.read_csv("/kaggle/input/fullspecs-fetched-data/fullspecs_fetched_data.csv",index_col=0).transpose()



raw_data = raw_data_original.copy()
raw_data.shape
curb_cols = [col for col in raw_data.columns.str.lower() if 'curb' in col]

print(curb_cols)
#Hybrid / Electric

raw_data.shape

raw_data['Full Name'] = raw_data.index

raw_data['Hybrid'] = raw_data['Full Name'].str.lower().str.contains('hyb')

raw_data['Electric'] = (raw_data['Engine Type'].str.lower().str.contains('elect') | raw_data['Full Name'].str.lower().str.contains('elect'))

raw_data['Electric'].loc[raw_data['Electric']==1]
raw_data['Turning Diameter - Curb to Curb (ft)'] = raw_data['Turning Diameter - Curb to Curb (ft)'].str.strip('-')

raw_data['Turning Diameter - Curb to Curb (ft)'] = raw_data['Turning Diameter - Curb to Curb (ft)'].str.split().str.get(0)

raw_data['Turning Diameter - Curb to Curb (ft)'] = raw_data['Turning Diameter - Curb to Curb (ft)'].str.replace('TBD', 

                                                            '').replace(r'^\s*$', np.nan, regex=True).astype(float)
raw_data['Turning Diameter - Curb to Curb (ft)']
'''raw_data['Reverse Ratio (:1)'] = raw_data['Reverse Ratio (:1)'].str.split('/').str.get(0).str.strip('-TBD-')

raw_data['Reverse Ratio (:1)'] = raw_data['Reverse Ratio (:1)'].str.split('-').str.get(0).str.strip('Variable')

raw_data['Reverse Ratio (:1)'] = raw_data['Reverse Ratio (:1)'].str.split().str.get(0).astype(float)'''
#### COMPANY NAMES



raw_data['Full Name'] = raw_data.index

raw_data['Company Name'] = raw_data['Full Name'].str.split(" ",expand=True)[1]



del raw_data['Full Name']



raw_data['Company Name'] = raw_data['Company Name'].str.replace("Alfa", "Alfa Romeo")

raw_data['Company Name'] = raw_data['Company Name'].str.replace("Aston", "Aston Martin")

raw_data['Company Name'] = raw_data['Company Name'].str.replace("Land", "Land Rover")

raw_data['Company Name'] = raw_data['Company Name'].str.replace("smart", "Smart")





# -------- replace na and tbd with np nan



raw_data.replace("NA", np.nan)

raw_data = raw_data.replace("- TBD –", 'NA')

raw_data = raw_data.replace("- TBD -", 'NA')

raw_data['EPA Fuel Economy Est - City (MPG)'] = raw_data['EPA Fuel Economy Est - City (MPG)'].str.replace(r"\(.*\)","")

raw_data = raw_data.replace("NA", np.nan)



# -------- cols with forbidden charac



raw_data = raw_data.rename(columns=lambda x: x.split(" (ft")[0])

raw_data['Passenger Volume'] = raw_data['Passenger Volume'].str.replace(r"\(.*\)","")



# -------- Clean MSRP and convert to float



raw_data.MSRP = raw_data.MSRP.str.replace("$", "")

raw_data.MSRP = raw_data.MSRP.str.replace(",", "")



# -------- Clean basic miles and convert to float



raw_data['Basic Miles/km'] = raw_data['Basic Miles/km'].str.replace(",", "")

raw_data['Basic Miles/km'] = raw_data['Basic Miles/km'].str.replace("Unlimited", "150000")

raw_data['Basic Miles/km'] = raw_data['Basic Miles/km'].str.replace("49999", "50000")



# -------- Clean Drivetrain Miles and convert to float



raw_data['Drivetrain Miles/km'] = raw_data['Drivetrain Miles/km'].str.replace(",", "")

raw_data['Drivetrain Miles/km'] = raw_data['Drivetrain Miles/km'].str.replace("Unlimited", "150000")



# -------- get Roadside Assistance Miles/km miles  as integer



raw_data['Roadside Assistance Miles/km'] = raw_data['Roadside Assistance Miles/km'].str.replace(",", "")

raw_data['Roadside Assistance Miles/km'] = raw_data['Roadside Assistance Miles/km'].str.replace("Unlimited", "100000")



# -------- get number of gears



raw_data['Transmission'] = raw_data['Transmission'].str.lower()

raw_data['Gears'] = raw_data['Transmission'].str.split("-speed", expand=True, n = 1)[0].str[-2:].str.strip()

raw_data.Gears = raw_data['Gears'].str.replace("le", "1")

raw_data.Gears = raw_data['Gears'].str.replace("ed", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("ic", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("es", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("er", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("ls", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("ve", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("to", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("de", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("ch", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("ct", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("rs", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("ft", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("al", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("s,", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("on", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("NA", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("co", "NA")

raw_data.Gears = raw_data['Gears'].str.replace("/8", "NA")





# -------- get max horsepower 

################ FIX THE FIRST SPLIT TO KEEP ONLY THE NUMBER



raw_data['Net Horsepower'] = raw_data['SAE Net Horsepower @ RPM'].str.split(" ",expand=True)[0].value_counts()

raw_data['Net Horsepower'] = raw_data['Net Horsepower'].astype(float)

raw_data.replace("NA", np.nan, inplace=True)





# -------- get max horsepower rpm 

############### FIX

raw_data['Net Horsepower RPM'] = raw_data['SAE Net Horsepower @ RPM'].str.split("@",expand=True)[1].str.strip()

raw_data['Net Horsepower RPM'] = raw_data['Net Horsepower RPM'].str.replace("- TBD -", "NA").str.replace("-TBD-", "NA")

raw_data['Net Horsepower RPM'] = raw_data['Net Horsepower RPM'].str[:4]



# -------- get max torque



raw_data['Net Torque'] = raw_data['SAE Net Torque @ RPM'].str.split(" ", expand=True)[0]

raw_data.replace("NA", np.nan, inplace=True)

raw_data['Net Torque'] = raw_data['Net Torque'].astype(float)


list(raw_data.columns)
raw_data['Transmission'] = raw_data['Transmission'].str.lower()

raw_data['Gears'] = raw_data['Transmission'].str.split("-speed", expand=True, n = 1)[0].str[-2:].str.strip()

raw_data['Gears'] = raw_data['Gears'].str.strip(letters).str.strip('/').str.strip(',').replace(r'^\s*$', np.nan, regex=True)

raw_data['Gears'] = raw_data['Gears'].astype(float)

raw_data['Gears'].value_counts()
raw_data['Cylinders'] = raw_data['Engine Type'].str.strip(letters).str.strip(symbols).str.strip(letters).str.strip(symbols)

raw_data['Cylinders'] = raw_data['Cylinders'].str.strip().str.split().str.get(-1)

raw_data['Cylinders'] = raw_data['Cylinders'].str.replace("-", "").str.replace("/", "")

raw_data['Cylinders'] = raw_data['Cylinders'].str.lstrip(letters)

raw_data['Cylinders'] = raw_data['Cylinders'].replace(r'^\s*$', np.nan, regex=True).replace('4Cyl', '4')

raw_data['Cylinders'] = raw_data['Cylinders'].astype(float)
config = ['turbo', 'supercharger', 'regular', 'unleaded', 'premium', 'gas', 'electric', 'turbocharged', 'flexible',

          'intercooled', 'twin', 'unleaded', 'charged', 'ethanol', 'natural', 'high pressure', 'low pressure',

          'ecotec', 'cyl', 'diesel', 'compressed', 'super', 'vortec', '4', '6', '8', '5', '(']

raw_data['Engine Configuration'] = raw_data['Engine Type'].str.lower() 

raw_data['Engine Configuration'] = raw_data['Engine Configuration'].str.strip(numbers).str.lower()

raw_data['Engine Configuration'] = raw_data['Engine Configuration'].str.replace('-', " ").str.replace('/', " ")

raw_data['Engine Configuration'] = raw_data['Engine Configuration'].str.strip(symbols).str.rstrip(numbers)



for i in config:

    raw_data['Engine Configuration'] = raw_data['Engine Configuration'].str.replace(i, " ")

raw_data['Engine Configuration'] = raw_data['Engine Configuration'].str.strip().str[-1]

raw_data['Engine Configuration'] = raw_data['Engine Configuration'].str.upper().str.replace('T', 'FLAT').replace('L', np.nan)

raw_data['Engine Configuration'].value_counts()
raw_data["Rear Tire Width"] = raw_data["Rear Tire Size"].str.split("/").str.get(0).str[-3:].str.strip()

raw_data["Rear Tire Width"] = raw_data["Rear Tire Width"].replace('R20', np.nan).replace('R18', np.nan)

raw_data["Rear Tire Width"] = raw_data["Rear Tire Width"].replace('D -', np.nan).replace('R15', np.nan).replace('R15', np.nan)

raw_data["Rear Tire Width"] = raw_data["Rear Tire Width"].replace('18"', np.nan).replace('60A', np.nan)

raw_data["Rear Tire Width"] = raw_data["Rear Tire Width"].astype(float)

raw_data["Rear Tire Width"].value_counts().head(8)
raw_data["Front Tire Width"] = raw_data["Front Tire Size"].str.split("/").str.get(0).str[-3:].str.strip()

raw_data["Front Tire Width"] = raw_data["Front Tire Width"].replace('R20', 'NA').str.strip(letters).str.strip(symbols)

raw_data["Front Tire Width"] = raw_data["Front Tire Width"].replace('18"', 'NA').replace('R15', '').replace('D -', '')

raw_data["Front Tire Width"] = raw_data["Front Tire Width"].replace("NA", np.nan).replace("60", np.nan).replace("15", np.nan)

raw_data["Front Tire Width"] = raw_data["Front Tire Width"].replace(r'^\s*$', np.nan, regex=True).astype(float)

raw_data["Front Tire Width"].value_counts().head(8)
raw_data["Tire Rating"] = raw_data["Front Tire Size"].str.split("/").str.get(-1).str.strip(numbers).str[0].str.upper()

raw_data["Tire Rating"] = raw_data["Tire Rating"].replace(r'^\s*$', np.nan, regex=True).replace('-', np.nan)

raw_data["Tire Rating"] = raw_data["Tire Rating"].replace('"', np.nan).replace('P', np.nan).replace('X', np.nan)

raw_data["Tire Rating"].value_counts()
raw_data['Country'] = raw_data['Company Name']



raw_data['Country'] = raw_data['Country'].replace(['Ford', 'Chevrolet', 'GMC', 'Ram', 'Jeep', 'Cadillac', 'Dodge',

                                      'Buick', 'Lincoln', 'Chrysler', 'Tesla'], 'USA')

raw_data['Country'] = raw_data['Country'].replace(['Jaguar', 'Land Rover', 'Bentley', 'Rolls-Royce', 

                                       'Aston Martin', 'Lotus', 'McLaren', 'Mini', 'MINI'], 'UK')

raw_data['Country'] = raw_data['Country'].replace(['Toyota', 'Nissan', 'Honda', 'Subaru', 'Mazda', 'Acura', 

                                       'Mitsubishi', 'Lexus', 'Infiniti', 'INFINITI'], 'Japan')

raw_data['Country'] = raw_data['Country'].replace(['Volkswagen', 'BMW', 'Audi', 'Mercedes-Benz', 'Porsche', 'Smart'], 'Germany')

raw_data['Country'] = raw_data['Country'].replace(['Hyundai', 'Kia', 'Genesis'], 'Korea')

raw_data['Country'] = raw_data['Country'].replace(['Volvo'], 'Sweden')

raw_data['Country'] = raw_data['Country'].replace(['Fiat', 'Maserati', 'Alfa Romeo', 'Lamborghini', 'Ferrari', 'FIAT'], 'Italy')



raw_data['Country Code'] = raw_data['Country'].astype("category").cat.codes

raw_data['Displacement (L)'] = raw_data['Displacement'].str.split("/", expand=True)[0].str[:3]

raw_data['Displacement (L)'] = raw_data['Displacement (L)'].str.replace('39.', '3.9')





# -------- displacement - cc



raw_data['Displacement (cc)'] = raw_data['Displacement'].str.split("/", expand=True)[1]

raw_data['Displacement (cc)'] = raw_data['Displacement (cc)'].str.replace('- TBD -', 'NA')

raw_data['Displacement (cc)'] = raw_data['Displacement (cc)'].str.replace('- TBD –', 'NA')

raw_data['Displacement (cc)'] = raw_data['Displacement (cc)'].str.replace('302 CID', 'NA')

raw_data['Displacement (cc)'] = raw_data['Displacement (cc)'].str.replace(' NA', 'NA')

# raw_data.loc['2018 Buick Envision Specs: AWD 4-Door Essence':'2018 Buick Envision Specs: AWD 4-Door Preferred',

# "Displacement (cc)"] = 'NA'



# -------- get rear tire width



raw_data["Rear Tire Width"] = raw_data["Rear Tire Size"].str.split("/").str.get(0).str[-3:].str.strip()

raw_data["Rear Tire Width"] = raw_data["Rear Tire Width"].replace('R20', 'NA').replace('18\"', 'NA').replace('R15', 'NA').replace('60A', 'NA').replace('R18', 'NA')

raw_data.replace("NA", np.nan, inplace=True)

raw_data["Rear Tire Width"] = raw_data["Rear Tire Width"].astype(float)



# -------- get front tire width



raw_data["Front Tire Width"] = raw_data["Front Tire Size"].str.split("/").str.get(0).str[-3:].str.strip()

#### FIX

raw_data["Front Tire Width"] = raw_data["Front Tire Width"].replace('R20', 'NA').replace('18\"', 'NA').replace('R15', 'NA').replace('60A', 'NA').replace('R18', 'NA')

raw_data.replace("NA", np.nan, inplace=True)

raw_data["Front Tire Width"] = raw_data["Front Tire Width"].astype(float)



# -------- get rear wheel size

#### FIX

raw_data["Rear Wheel Size"] = raw_data["Rear Wheel Size (in)"].str[:2].replace('P2', 'NA')

raw_data.replace("NA", np.nan, inplace=True)

raw_data["Rear Wheel Size"] = raw_data["Rear Wheel Size"].astype(float)

# -------- get front wheel size

#### FIX

raw_data["Front Wheel Size"] = raw_data["Front Wheel Size (in)"].str[:2].replace('P2', 'NA')

raw_data.replace("NA", np.nan, inplace=True)

raw_data["Front Wheel Size"]= raw_data["Front Wheel Size"].astype(float)

# -------- get tire rating



raw_data["Tire Rating"] = raw_data["Front Tire Size"].str.split("/").str.get(-1).str[-4]

raw_data["Tire Rating"] = raw_data["Tire Rating"].replace('5', 'NA')

raw_data["Tire Rating"] = raw_data["Tire Rating"].replace('0', 'NA')

raw_data["Tire Rating"] = raw_data["Tire Rating"].replace('1', 'NA')

raw_data["Tire Rating"] = raw_data["Tire Rating"].replace('2', 'NA')







# -------- get width ratio



raw_data["Tire Width Ratio"] = raw_data["Rear Tire Width"]/raw_data["Front Tire Width"]



# -------- get size ratio



raw_data["Wheel Size Ratio"] = raw_data["Rear Wheel Size"] / raw_data["Front Wheel Size"]



# -------- get tire ratio



raw_data["Tire Ratio"] = raw_data["Front Tire Size"].str.split("/").str.get(1).str[0]

raw_data["Tire Ratio"] = raw_data["Tire Ratio"].replace('Y', 'NA')



# -------- get year



raw_data["Year"] = raw_data.index.str[:4].astype(float)


# -------- edit drivetrain values



raw_data['Drivetrain'] = raw_data['Drivetrain'].str.replace('4-Wheel Drive', 'Four Wheel Drive')

raw_data['Drivetrain'] = raw_data['Drivetrain'].str.replace('Front wheel drive', 'Front Wheel Drive')

raw_data['Drivetrain'] = raw_data['Drivetrain'].str.replace('Four-Wheel Drive', 'Four Wheel Drive')



# -------- edit fuel system values



raw_data['Fuel System'] = raw_data['Fuel System'].str.replace('Turbocharged EFI', 'Electronic Fuel Injection')

raw_data['Fuel System'] = raw_data['Fuel System'].str.replace('Electric', 'Electronic Fuel Injection')

raw_data['Fuel System'] = raw_data['Fuel System'].str.replace('Sequential MPI (injection)', 'Sequential MPI')

raw_data['Fuel System'] = raw_data['Fuel System'].str.replace('SMPI', 'Sequential MPI')

raw_data['Fuel System'] = raw_data['Fuel System'].str.replace('EFI', 'Electronic Fuel Injection')

raw_data['Fuel System'] = raw_data['Fuel System'].str.replace('Direct Gasoline Injection', 'Direct Injection')
raw_data['Sixth Gear Ratio (:1)'] = raw_data['Sixth Gear Ratio (:1)'].str.strip(letters).str.strip(symbols)

raw_data['Sixth Gear Ratio (:1)'] = raw_data['Sixth Gear Ratio (:1)'].str.strip(' TBD ').replace(r'^\s*$', np.nan, regex=True).astype(float)
raw_data = raw_data.rename(columns=lambda x: x.split(" (ft")[0])

raw_data['EPA Fuel Economy Est - City (MPG)'] = raw_data['EPA Fuel Economy Est - City (MPG)'].str.replace(r"\(.*\)","")

raw_data['Passenger Volume'] = raw_data['Passenger Volume'].str.replace(r"\(.*\)","")
raw_data['Cylinders'] = raw_data['Engine Type'].str.split("-", expand=True)[1]

raw_data['Cylinders'] = raw_data['Cylinders'].str.replace("Cyl", "4")

raw_data['Cylinders'] = raw_data['Cylinders'].str.replace("in Electric I4", "4")





# -------- replace na by npnan



raw_data.replace("NA", np.nan, inplace=True)



# -------- convert all to float



raw_data.MSRP = raw_data.MSRP.astype(float)

raw_data["Tire Ratio"] = raw_data["Tire Ratio"].astype(float)

raw_data['Displacement (cc)'] = raw_data['Displacement (cc)'].astype(float)

raw_data['Displacement (L)'] = raw_data['Displacement (L)'].astype(float)



raw_data['Cylinders'] = raw_data['Cylinders'].str.replace('cyl', 'NA').str.replace('Pressure Turbo Gas I5', 'NA').str.replace('Turbocharged Gas V12', 'NA').str.replace('Scroll Turbocharged Gas I6', 'NA').str.replace('4 Turbocharged', 'NA').str.replace('Turbocharged Gas V8', 'NA')

raw_data.replace("NA", np.nan, inplace=True)

raw_data['Cylinders'] = raw_data['Cylinders'].astype(float)



raw_data['Net Horsepower RPM'] = raw_data['Net Horsepower RPM'].astype(float)
raw_data['Gears'] = raw_data['Gears'].astype(float)

raw_data['Roadside Assistance Miles/km'] = raw_data['Roadside Assistance Miles/km'].astype(float)

raw_data['Drivetrain Miles/km'] = raw_data['Drivetrain Miles/km'].astype(float)

raw_data['Basic Miles/km'] = raw_data['Basic Miles/km'].astype(float)
raw_data.head()
raw_data['Net Torque RPM'] = raw_data['SAE Net Torque @ RPM'].str.split().str.get(-1).str[-4:].str.strip()

raw_data['Net Torque RPM'] = raw_data['Net Torque RPM'].str.replace("- TBD -", "NA").str.replace('-', 'NA').str.replace('ined', 'NA').str.replace('E85', 'NA').str.replace('NA\)', 'NA').str.replace('est\)', 'NA')

raw_data.replace("NA", np.nan, inplace=True)

raw_data['Net Torque RPM'] = raw_data['Net Torque RPM'].astype(float)

raw_data['Net Torque RPM'] = raw_data['Net Torque RPM'].clip(lower=1000)







# -------- replace na by npnan



raw_data.replace("NA", np.nan, inplace=True)



# -------- convert all to float



raw_data.MSRP = raw_data.MSRP.astype(float)

raw_data["Tire Ratio"] = raw_data["Tire Ratio"].astype(float)

raw_data['Displacement (cc)'] = raw_data['Displacement (cc)'].astype(float)

raw_data['Displacement (L)'] = raw_data['Displacement (L)'].astype(float)

raw_data['Cylinders'] = raw_data['Cylinders'].astype(float)

raw_data['Net Horsepower RPM'] = raw_data['Net Horsepower RPM'].astype(float)

raw_data['Gears'] = raw_data['Gears'].astype(float)

raw_data['Roadside Assistance Miles/km'] = raw_data['Roadside Assistance Miles/km'].astype(float)

raw_data['Drivetrain Miles/km'] = raw_data['Drivetrain Miles/km'].astype(float)

raw_data['Basic Miles/km'] = raw_data['Basic Miles/km'].astype(float)



# -------- converet numeric



specs_to_numeric = ['MSRP', 'Passenger Capacity', 'Passenger Doors',

                    'Base Curb Weight (lbs)', 'Second Shoulder Room (in)',

                    'Second Head Room (in)', 'Front Shoulder Room (in)',

                    'Second Hip Room (in)', 'Front Head Room (in)', 'Second Leg Room (in)', 'Front Hip Room (in)',

                    'Front Leg Room (in)', 'Width, Max w/o mirrors (in)', 'Track Width, Rear (in)',

                    'Height, Overall (in)', 'Wheelbase (in)', 'Track Width, Front (in)',

                    'Fuel Tank Capacity, Approx (gal)', 'EPA Fuel Economy Est - City (MPG)',

                    'EPA Fuel Economy Est - Hwy (MPG)',

                    'Fuel Economy Est-Combined (MPG)', 'Fourth Gear Ratio (:1)',

                    'Second Gear Ratio (:1)', 'Reverse Ratio (:1)', 'Fifth Gear Ratio (:1)',

                    'Third Gear Ratio (:1)', 'Final Drive Axle Ratio (:1)', 'First Gear Ratio (:1)',

                    'Sixth Gear Ratio (:1)', 'Passenger Volume',

                    'Front Brake Rotor Diam x Thickness (in)', 'Disc - Front (Yes or   )',

                    'Rear Brake Rotor Diam x Thickness (in)', 'Rear Wheel Size (in)',

                    'Rear Wheel Material', 'Spare Wheel Size (in)', 'Front Wheel Size (in)', 'Basic Miles/km',

                    'Basic Years', 'Corrosion Years', 'Drivetrain Miles/km', 'Drivetrain Years',

                    'Roadside Assistance Miles/km', 'Roadside Assistance Years', 'Year', 'Tire Ratio',

                    'Front Tire Width', 'Rear Tire Width', 'Displacement (cc)', 'Displacement (L)', 'Net Torque RPM',

                    'Net Torque', 'Gears', 'Net Horsepower', 'Net Horsepower RPM', 'Cylinders']



for i in specs_to_numeric:

    raw_data[i] = pd.to_numeric(raw_data[i], errors='coerce')



raw_data.head()
raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Compact Cars', 'Compact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Mid-Size Cars', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Small Sport Utility Vehicles 4WD', 'Small SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('4WD Sport Utility Vehicle', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Small Sport Utility Vehicles 2WD', 'Small SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Mid-Size', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Two-Seaters', 'Two-Seater')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Sub-Compact', 'Subcompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Small Station Wgn', 'Small Station Wagon')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Large Cars', 'Large')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Subcompact Cars', 'Subcompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('4WD Sport Utility', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Standard Sport Utility Vehicles 4WD', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('SUV 4WDs', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Sport Utility Vehicle - 4WD', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Sport Utility Vehicle - 2WD', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('SUV 4WDs', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Large Car', 'Large')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('2WD Sport Utility', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Subcompact car', 'Subcompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('2WD Sport Utility Vehicles', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Standard Sport Utility Vehicles 2WD', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Mini-Compact', 'Minicompact Cars')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Midsize Car', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('AWD Sport Utility', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('SUV 2WD Vehicle', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Two-Seater', 'Two Seater')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Sport Utility Vehicle', 'SUV')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Midsizes', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('2WD Minivans', '2WD Minivan')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Special Purpose Vehicle', 'Special Purpose')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Small Station Wagons', 'Small Station Wagon')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('SUV 2WDs', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Midsize Sedan', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('2WD Sport Utililty', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('FWD SUV', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Midsize cars', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('SUV - AWD', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Minicompact Car', 'Minicompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Minicompact Cars ', 'Minicompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Mid-size', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('FWD Sport Utility', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Sub-compact', 'Subcompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('SUVs', 'SUV')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('2 Seater', 'Two Seater')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('SUV 4WD Vehicle', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('2WD sport Utility Vehicle', 'SUV 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Sport Utility', 'SUV')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('MidSize Cars', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Small station wagon', 'Small Station Wagon')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Minicompacts Car', 'Minicompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Sub Compact', 'Subcompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Mini-compact', 'Minicompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('SUV 4WDs', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('large', 'Large')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Two seaters', 'Two Seater')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Midsize sedan', 'Midsize')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Subcompacts', 'Subcompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Compact Car', 'Compact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Midsize Station Wagons', 'Midsize Station Wagon')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Two seater', 'Two Seater')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('4WD sport Utility Vehicle', 'SUV 4WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Compact Sedan', 'Compact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Large car', 'Large')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Subcompact Car', 'Subcompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Two Seaters', 'Two Seater')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Minivan - 2WD', '2WD Minivan')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('4WD Minivans', '4WD Minivan')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('2WD Special Purpose', 'Special Purpose 2WD')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Midsize S/W', 'Midsize Station Wagon')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Midsize Wagon', 'Midsize Station Wagon')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('2WD Van', '2WD Minivan')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Minicompacts', 'Minicompact')

raw_data['EPA Classification'] = raw_data['EPA Classification'].str.replace('Pick-up Truck', 'Truck')





raw_data.loc[raw_data['EPA Classification'] == 'Full Size', 'EPA Classification'] = 'Midsize'

raw_data.loc[raw_data['EPA Classification'] == 'Wagon', 'EPA Classification'] = 'Small Station Wagon'

raw_data.loc[raw_data['EPA Classification'] == 'Small SUV', 'EPA Classification'] = 'SUV'

raw_data.loc[raw_data['EPA Classification'] == 'Pickup Trucks', 'EPA Classification'] = np.nan

raw_data.loc[raw_data['EPA Classification'] == 'Light-Duty Truck', 'EPA Classification'] = np.nan

raw_data.loc[raw_data['EPA Classification'] == '4WD Pickup Trucks', 'EPA Classification'] = np.nan

raw_data.loc[raw_data['EPA Classification'] == '4WD Standard Pickup Truck', 'EPA Classification'] = np.nan





del raw_data['Other Features']





raw_data['Corrosion Miles/km'] = raw_data['Corrosion Miles/km'].str.replace("Unlimited", "10000000")





raw_data.head()
#final prep-processing





raw_data['Min Ground Clearance (in)'] = raw_data['Min Ground Clearance (in)'].str.slice(stop=2).astype(float)



raw_data['Corrosion Miles/km']= raw_data['Corrosion Miles/km'].str.replace('50,000', '50000')

raw_data['Corrosion Miles/km']= raw_data['Corrosion Miles/km'].str.replace('60,000', '60000')

raw_data['Corrosion Miles/km']= raw_data['Corrosion Miles/km'].str.replace('100,000', '100000')

raw_data['Corrosion Miles/km'] = raw_data['Corrosion Miles/km'].astype(float)



del raw_data['Maximum Alternator Capacity (amps)']



del raw_data['Cold Cranking Amps @ 0° F (Primary)'] 

del raw_data['Wt Distributing Hitch - Max Tongue Wt. (lbs)'] 

raw_data['Drivetrain'].value_counts()
raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('Rear wheel drive', 'Rear Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('Rear wheel drive ', 'Rear Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('Rear-Wheel Drive', 'Rear Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('REAR WHEEL DRIVE', 'Rear Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('RWD', 'Rear Wheel Drive')



raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('Front-Wheel Drive', 'Front Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('Front-wheel Drive', 'Front Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('Front-wheel drive', 'Front Wheel Drive')



raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('4 Wheel Drive', 'Four Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('4WD', 'Four Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('Four wheel drive', 'Four Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('4-wheel Drive', 'Four Wheel Drive')



raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('All-Wheel Drive', 'All Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('AWD', 'All Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('All-wheel drive', 'All Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('All wheel drive', 'All Wheel Drive')



raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('2 Wheel Drive', 'Two Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('2WD', 'Two Wheel Drive')

raw_data['Drivetrain']= raw_data['Drivetrain'].str.replace('2-Wheel Drive', 'Two Wheel Drive')
df = raw_data.copy()

#raw_data = df.copy()
# DELETE COLUMNS

specs_to_delete = ['Gas Mileage', 'Engine', 'Engine Type', 'SAE Net Horsepower @ RPM', 'SAE Net Torque @ RPM',

                  'Displacement', 'Trans Description Cont.', 'Rear Tire Size', 'Front Tire Size', 'Rear Wheel Size (in)',

                  'Front Wheel Size (in)', 'Transmission', 'EPA Class', 'Brake ABS System', 'Disc - Front (Yes or   )',

                  'Brake Type', 'Disc - Rear (Yes or   )', 'Spare Tire Size', 'Spare Wheel Size (in)', 'Spare Wheel Material']

raw_data.drop(specs_to_delete, axis=1, inplace=True)





########### FIX TO 75% ########## -------- Identifying columns with NaN totalling more than 75% of elements	



col_to_delete = raw_data.columns[raw_data.isna().sum() >= 0.25*len(raw_data)].tolist()



  ######

#Keep Hybrid columns (['Hybrid/Electric Components Miles/km', 'Hybrid/Electric Components Years', 'Hybrid/Electric Components Note', 'Hybrid'] )

#                 even if they have many missing values

hyb_cols = [col for col in raw_data if 'ybri' in col]



for x in hyb_cols:

    if x in col_to_delete:

        col_to_delete.remove(x)



for x in ['MSRP', 'Year', 'EPA Classification', 'Company Name', 'EPA Fuel Economy Est - City (MPG)', 'EPA Fuel Economy Est - Hwy (MPG)',

          'Base Curb Weight (lbs)', 'Turning Diameter - Curb to Curb', 'Curb Weight - Front (lbs)', 'Curb Weight - Rear (lbs)' ]:

    if x in col_to_delete:

        col_to_delete.remove(x)

######



raw_data.drop(col_to_delete, axis=1, inplace=True)

raw_data.head()
raw_data.columns
raw_data.shape[1]
raw_data['EPA Classification'].isnull().value_counts()
df = raw_data.copy()
raw_data['Name'] = raw_data.index



raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Small", na=False)), 'EPA Classification' ] = "Compact"     

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Truck", na=False)), 'EPA Classification' ] = "Pick-up Truck"

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Minivan", na=False)), 'EPA Classification' ] = "Van" 

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Purpose", na=False)), 'EPA Classification' ] = "Special Purpose"

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Subcompact", na=False)), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Minicompact", na=False)), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Small", na=False)), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Mid-sized", na=False)), 'EPA Classification' ] = "Midsize"

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Mid size", na=False)), 'EPA Classification' ] = "Midsize"

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Special", na=False)), 'EPA Classification' ] = "SUV"                    

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Model X", na=False)), 'EPA Classification' ] = "All Electric SUV"  

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Pick-up", na=False)), 'EPA Classification' ] = "Pick-up Truck"    



raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("SUV 4WD", na=False)), 'EPA Classification' ] = "SUV"                    

raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("SUV 2WD", na=False)), 'EPA Classification' ] = "SUV"                    



raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Midsize Station Wagon", na=False)), 'EPA Classification' ] = "Wagon"  



raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Pick-up Truck", na=False)), 'EPA Classification' ] = "Truck"     





raw_data.loc[ pd.Series(raw_data.Name.str.contains("Atlas ")), 'EPA Classification' ] = "SUV"                         

raw_data.loc[ pd.Series(raw_data.Name.str.contains("4Runner")), 'EPA Classification' ] = "SUV" 

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Ascent")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("GLS")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Navigator")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Lexus Lx")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Range Rover")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Wrangler")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("INFINITY")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Yukon")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Expedition")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Durango")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Suburban")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Escalade")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Land Rover")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Sierra")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Bentayga")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Audi Q8")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Sequoia")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Cayenne")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("G Class")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Blazer")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Suburban")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Land Cruiser")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Cullinan")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Lexus GX")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Chevrolet Tahoe")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("BMW X7")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Mercedes-Benz GLE")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Murano")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Audi Q3")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Ford Explorer")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Pathfinder")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Passport")), 'EPA Classification' ] = "SUV"





raw_data.loc[ pd.Series(raw_data.Name.str.contains("Super Duty")), 'EPA Classification' ] = "Pick-up Truck"  





raw_data.loc[ pd.Series(raw_data.Name.str.contains("3-Series")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("5-Series")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Mirage")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Maxima")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Sentra")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Camry")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Corolla")), 'EPA Classification' ] = "Compact"





raw_data.loc[ pd.Series(raw_data.Name.str.contains("Elantra")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Veloster")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Jetta")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Golf")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Elantra")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Impreza")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Altima")), 'EPA Classification' ] = "Compact"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Supra")), 'EPA Classification' ] = "Compact"





raw_data.loc[ pd.Series(raw_data.Name.str.contains("911")), 'EPA Classification' ] = "Two Seater"





raw_data.loc[ pd.Series(raw_data.Name.str.contains("Tesla")), 'EPA Classification' ] = "All Electric"







raw_data.loc[ pd.Series(raw_data.Name.str.contains("Transit")), 'EPA Classification' ] = "Midsize"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Regal")), 'EPA Classification' ] = "Midsize"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Camaro")), 'EPA Classification' ] = "Midsize"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Legacy")), 'EPA Classification' ] = "Midsize"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Passat")), 'EPA Classification' ] = "Midsize"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Taurus")), 'EPA Classification' ] = "Midsize"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Sonata")), 'EPA Classification' ] = "Midsize"





raw_data.loc[ pd.Series(raw_data.Name.str.contains("Ranger")), 'EPA Classification' ] = "Compact"





raw_data.loc[ pd.Series(raw_data.Name.str.contains("Ram")), 'EPA Classification' ] = "Van"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Sprinter")), 'EPA Classification' ] = "Van"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Nissan NV")), 'EPA Classification' ] = "Van"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Metris")), 'EPA Classification' ] = "Van"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Express")), 'EPA Classification' ] = "Van"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("NV200")), 'EPA Classification' ] = "Van"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Savana")), 'EPA Classification' ] = "Van"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Savana")), 'EPA Classification' ] = "Van"







raw_data.loc[ pd.Series(raw_data.Name.str.contains("Lexus LX")), 'EPA Classification' ] = "Large"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Jaguar XF")), 'EPA Classification' ] = "Large"



raw_data.loc[ pd.Series(raw_data.Name.str.contains("Ridgeline")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Titan")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Frontier")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Ford F")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Sierra")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Canyon")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Silverado")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Tundra")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Tacoma")), 'EPA Classification' ] = "Truck"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Colorado")), 'EPA Classification' ] = "Truck"





raw_data.loc[ pd.Series(raw_data.Name.str.contains("BMW X5")), 'EPA Classification' ] = "SUV"



raw_data.loc[ pd.Series(raw_data.Name.str.contains("Bolt EV")), 'EPA Classification' ] = "All Electric"



raw_data.loc[ pd.Series(raw_data.Name.str.contains("QX80")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Armada")), 'EPA Classification' ] = "SUV"

raw_data.loc[ pd.Series(raw_data.Name.str.contains("Urus")), 'EPA Classification' ] = "SUV"





raw_data.loc[ pd.Series(raw_data['EPA Classification'].str.contains("Pick-up", na=False)), 'EPA Classification' ] = "Truck"    









###REMOVE ALL OTHER NAN

raw_data = raw_data.loc[raw_data['EPA Classification'].notnull()]



raw_data['EPA Classification'].value_counts()
raw_data['EPA Classification'].isnull().value_counts()
#raw_data = raw_data.drop(raw_data[raw_data.Year == 2019].index)
raw_data['Curb Weight - Front (lbs)'] = raw_data['Curb Weight - Front (lbs)'].astype(float)

raw_data['Country Code'] = raw_data['Country Code'].astype("category")
#go one year back for names and year of cars of 2019 and 2018



raw_data.loc[raw_data.Year==2018, 'Year'] = 2017

raw_data.loc[raw_data.Year==2017, 'Name'] = raw_data.loc[raw_data.Year==2017, 'Name'].str.slice_replace(stop=4, repl='2017')



raw_data.loc[raw_data.Year==2019, 'Year'] = 2018

raw_data.loc[raw_data.Year==2018, 'Name'] = raw_data.loc[raw_data.Year==2018, 'Name'].str.slice_replace(stop=4, repl='2018')



raw_data.set_index('Name', inplace=True)



raw_data.Year.value_counts()
raw_data.loc[:,raw_data.select_dtypes(float).columns] = raw_data.select_dtypes(float).fillna(raw_data.select_dtypes(float).mean())

raw_data.loc[:,raw_data.select_dtypes(int).columns] = raw_data.select_dtypes(int).fillna(raw_data.select_dtypes(int).mean())

raw_data.loc[:,raw_data.select_dtypes(object).columns] = raw_data.select_dtypes(object).fillna(raw_data.select_dtypes(object).mode().iloc[0])



raw_data.head()
raw_data['Curb Weight - Front (lbs)'].isnull().value_counts()
df = raw_data.copy()
raw_data.info()
df.to_csv('raw_data_no_dummies_imputed.csv')

df.head()
df['EPA Classification'].value_counts()
# EPA CLASSIFICATION DUMMIES



'''# Get one hot encoding of columns B

one_hot = pd.get_dummies(df['EPA Classification'])

# Drop column B as it is now encoded

df = df.drop('EPA Classification',axis = 1)

# Join the encoded df

df = df.join(one_hot)

df.head()'''
#COMPANY NAME ONE HOT ENCONDING



'''# Get one hot encoding of columns B

one_hot = pd.get_dummies(df['Company Name'])

# Drop column B as it is now encoded

df = df.drop('Company Name',axis = 1)

# Join the encoded df

df = df.join(one_hot)

df.head()'''
df['Year'].value_counts().sort_index(ascending=False)
#here we can select specific time range

'''df = df.loc[df['Year']>2015]

df['Year'].value_counts().sort_index(ascending=False)'''
# if we have more than one categories we have to include on specs_to_dumies the 'EPA Classification'

specs_to_dummies = ['EPA Classification','Company Name']

        

for item in specs_to_dummies:

    dummies = pd.get_dummies(df[item], prefix_sep=': ', prefix=item)

    df = pd.concat([df, dummies], sort=False, axis=1)



df = df.drop(specs_to_dummies, axis=1)
df.head()
# DELETE ALL ROWS WHICH MISSING MSRP

df = df[pd.notnull(df['MSRP'])]

df[df['MSRP'].isnull()]
#######################################################################

# REALLY SIMPLISTIC APPROACH TO RUN PREDICTIVE MODELS #################

#######################################################################



df = df.fillna(0)
df.head()



duplicate_columns = df.columns[df.columns.duplicated()]

duplicate_columns.to_list()



df = df.select_dtypes(exclude=['object'])



df.drop(duplicate_columns.to_list(), axis=1)
df.MSRP = df.MSRP.astype(int)
from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

from sklearn.metrics import mean_squared_error
# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(df.drop('MSRP', axis=1), df['MSRP'], test_size=0.33, random_state=42)
alphas = 10**np.linspace(10,-2,100)*0.5



lasso = Lasso(max_iter = 100, normalize = True)

coefs = []



for a in alphas:

    lasso.set_params(alpha=a)

    lasso.fit(scale(X_train), y_train)

    coefs.append(lasso.coef_)

    

ax = plt.gca()

ax.plot(alphas*2, coefs)

ax.set_xscale('log')

plt.axis('tight')

plt.xlabel('alpha')

plt.ylabel('weights')



#CV

lassocv = LassoCV(alphas = alphas , cv = 10, max_iter = 100, normalize = True)

lassocv.fit(X_train, y_train)



lasso.set_params(alpha=lassocv.alpha_)

lasso.fit(X_train, y_train)

np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))

# EXAMPLE OF PREDICTION : Get estimates

for i in range(94, 194, 10):

    #Left the real, right the predicted

    y_test[i],lasso.predict(X_test)[i]
# Some of the coefficients are now reduced to exactly zero.

res = pd.Series(lasso.coef_, index=X_train.columns).to_frame("coeff")

# Selected Features

print("POSITIVE impact Features")

print(res[(res.coeff!=0) & (res.coeff>0) ].count())

print(res[(res.coeff!=0) & (res.coeff>0)  ].sort_values('coeff', ascending=False))
print("NEGATIVE impact Features")

print(res[(res.coeff!=0) & (res.coeff<0) ].count())

print(res[(res.coeff!=0) & (res.coeff<0)  ].sort_values('coeff', ascending=True))
print("Penalised Features")

print(res[(res.coeff==0) ].count())

print(res[(res.coeff==0)])
import xgboost as xgb



xgbc = xgb.XGBRegressor()

model = xgbc.fit(X_train, y_train)



xgb.plot_importance(model, max_num_features=20)
np.sqrt(mean_squared_error(y_test, model.predict(X_test)))