input_country, input_city = 'India', 'Lucknow'

# find_topk_nearby_channels(input_country, input_city)
import numpy as np

import pandas as pd
WorldData = pd.read_csv(

                 '../input/world-cities-database/worldcitiespop.csv',

                 dtype={'Country':str,'City':str,'AccentCity':str,'Region':str,'Population':np.float64,'Latitude':np.float64,'Longitude':np.float64},

                 low_memory=False)

WorldData.drop(['Region','Population'], inplace=True, axis=1)

WorldData.dropna(axis=0, inplace=True)

CodeToCountry = pd.read_csv('../input/isoalpha2codes/data_csv.csv')

SlackData = pd.read_csv('../input/openminedslackchannellocationdata/SlackData.csv')
slackChannelData = [

    ['#bangalore', 'Bengaluru', 'India'],

    ['#bhubaneshwar', 'Bhubaneshwar', 'India'],

    ['#chennai', 'Chennai', 'India'],

    ['#coimbatore', 'Coimbatore','India'],

    ['#hyderabad', 'Hyderabad', 'India'],

    ['#kolkata', 'Kolkata', 'India'],

    ['#mumbai', 'Mumbai', 'India'],

    ['#delhi-ncr', 'Delhi', 'India'],

    ['#pune-india', 'Pune', 'India'],

    ['#atlanta', 'Atlanta', 'United States'],

    ['#boston', 'Boston', 'United States'],

    ['#chicago', 'Chicago', 'United States'],

    ['#dallas', 'Dallas', 'United States'],

    ['#florida', 'Florida', 'United States'],

    ['#los-angeles', 'Los Angeles', 'United States'],

    ['#montreal', 'Montreal', 'Canada'],

    ['#nashville', 'Nashville', 'United States'],

    ['#newyork', 'New York', 'United States'],

    ['#san-francisco', 'San Francisco', 'United States'],

    ['#seattle', 'Seattle', 'United States'],

    ['#toronto', 'Toronto', 'Canada'],

    ['#washington-dc', 'Washington', 'United States'],

    ['#amsterdam', 'Amsterdam', 'Netherlands'],

    ['#athens', 'Athens', 'Greece'],

    ['#barcelona', 'Barcelona', 'Spain'],

    ['#berlin', 'Berlin', 'Germany'],

    ['#dublin', 'Dublin', 'Ireland'],

    ['#lisbon', 'Lisbon', 'Portugal'],

    ['#london', 'London', 'United Kingdom'],

    ['#madrid', 'Madrid', 'Spain'],

    ['#munich', 'Munich', 'Germany'],

    ['#paris', 'Paris', 'France'],

    ['#prague', 'prague', 'Czech Republic'],

    ['#preveza', 'Preveza', 'Greece'],

    ['#zurich', 'Zurich', 'Switzerland'],

    ['#abuja', 'Abuja', 'Nigeria'],

    ['#auckland', 'Auckland', 'New Zealand'],

    ['#bangkok', 'Bangkok', 'Thailand'],

    ['#dhaka', 'Dhaka', 'Bangladesh'],

    ['#ghana', 'Accra', 'Ghana'],

    ['#istanbul', 'Istanbul', 'Turkey'],

    ['#italy', 'Rome', 'Italy'],

    ['#jakarta', 'Jakarta', 'Indonesia'],

    ['#kathmandu', 'Kathmandu', 'Nepal'],

    ['#lagos', 'Lagos', 'Nigeria'],

    ['#melbourne', 'Melbourne', 'Australia'],

    ['#mexico', 'Mexico', 'Mexico'],

    ['#moscow', 'Moscow', 'Russian Federation'],

    ['#philippines', 'Manila', 'Philippines'],

    ['#punjab', 'Lahore', 'Pakistan'],

    ['#sao-paulo', 'São Paulo', 'Brazil'],

    ['#singapore', 'Singapore', 'Singapore'],

    ['#tokyo', 'Tokyo', 'Japan'],

]
def country_to_code(country: str):

    code = CodeToCountry[CodeToCountry.Name.str.lower() == str.lower(country)].Code

    countryCode = next(iter(code), None)

    return countryCode
def find_coords(country: str, city: str):

    countryCode = country_to_code(country)

    if countryCode is not None:

        locationData = WorldData[(WorldData.Country == str.lower(countryCode)) & ((WorldData.AccentCity==city) | (WorldData.City == str.lower(city)))]

    else:

        print("Country not found. Searching cities worldwide")

        locationData = WorldData[(WorldData.AccentCity == city) | (WorldData.City == str.lower(city))]

    if not locationData.shape[0]:

        print("City not found. Try again with some nearby prominent city")

    lat, long = locationData.Latitude, locationData.Longitude

    lat = next(iter(lat), None)

    long = next(iter(long), None)

    return lat, long
def build_slack_channel_data(slackChannelData):

    latitudes = []

    longitudes = []

    for channel, city, country in slackChannelData:

        lat, long = find_coords(country, city)

        if lat is not None and long is not None:

            latitudes.append(lat)

            longitudes.append(long)

        else:

            print("Failed to find coordinates for channel: ", channel)

            return

    SlackData = pd.DataFrame(slackChannelData, columns=['Channel Name','City', 'Country'])

    SlackData['Latitude'] = latitudes

    SlackData['Longitude'] = longitudes

    return SlackData

#SlackData = build_slack_channel_data(slackChannelData)

#SlackData.to_csv('SlackData.csv', index=False)
def find_topk_nearby_channels(inputCountry, inputCity, slackdata=SlackData, k=5):

    lat, long = find_coords(inputCountry, inputCity)

    if lat is not None and long is not None:

        return slackdata.iloc[((slackdata['Longitude'] - long)**2+(slackdata['Latitude'] - lat)**2).argsort()[:k],:3]

    else:

        return