import pandas as pd
import matplotlib.pyplot as plt
road_weather_info = pd.read_csv("../input/road-weather-information-stations.csv")
road_weather_info.describe()
road_weather_info.boxplot(column = 'AirTemperature')
road_weather_info.hist(column = 'RoadSurfaceTemperature')