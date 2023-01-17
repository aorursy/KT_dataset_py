import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv", header=0, index_col=0, encoding='utf-8')



data.info()
data.isnull().sum()
hotel = data.reset_index().groupby("hotel").aggregate({"hotel": "count"}).rename(columns={'hotel': 'count', 'index': 'hotel'})

hotel = hotel.sort_values('count', ascending=False).reset_index()



plt.figure(figsize=(8, 8))

explode = [0.07,0]

labels = hotel["hotel"]

colors = ['lightsalmon', 'skyblue']

plt.pie(hotel["count"], autopct='%.1f%%', explode=explode, labels=labels, colors=colors)

plt.title(label=" Type of Hotel", loc="center", fontsize=16)



plt.show()
where = data["country"].fillna({"country": "unknown"}).reset_index()

where = where.groupby("country").aggregate({"country": "count"})

where = where.rename(columns={'country': 'count', 'index': 'country'}).reset_index().sort_values('count', ascending=False)



sns.set(style='darkgrid')

sns.catplot(x="country", y="count", kind="bar", data=where.head(10))

plt.title(label="Country Top 10", loc="center", fontsize=15)

plt.tight_layout()



plt.show()
price = data[~data['adr'].isin([5400])] 

price = price.sort_values('assigned_room_type', ascending=True)



plt.figure(figsize=(10, 7))

current_palette =sns.hls_palette(2, h=0.5)

sns.boxplot(x="assigned_room_type", y="adr",

            hue="hotel", palette=current_palette,

            data=price.reset_index())

plt.title(label="Price of room type", loc="center", fontsize=15)

plt.show()
month = data[["arrival_date_month", "adr"]].reset_index().sort_values("arrival_date_month")



ordered_months = ["January", "February", "March", "April", "May", "June",

          "July", "August", "September", "October", "November", "December"]

month["arrival_date_month"] = pd.Categorical(month["arrival_date_month"], categories=ordered_months, ordered=True)



plt.figure(figsize=(10, 7))

sns.lineplot(x = "arrival_date_month", y="adr", hue="hotel", data=month,

            ci='sd', sizes=(2.5, 2.5))

plt.title(label="Price changes over year", loc="center", fontsize=15)



plt.show()
data = data.fillna({"children": 0})

data["guest"] = data["adults"] + data ["children"] + data["babies"]

guest = data[["guest", "arrival_date_month"]].reset_index()

ordered_months = ["January", "February", "March", "April", "May", "June",

    "July", "August", "September", "October", "November", "December"]

guest["arrival_date_month"] = pd.Categorical(guest["arrival_date_month"], categories=ordered_months, ordered=True)



plt.figure(figsize=(12, 7))

sns.lineplot(x = "arrival_date_month", y="guest", hue="hotel", data=guest, ci=1, sizes=(2.5, 2.5))

plt.title(label="Guest changes over year", loc="center", fontsize=15)



plt.tight_layout()

plt.show()
data["guest"] = data["adults"] + data ["children"] + data["babies"]

data = data[data["is_canceled"]==0].reset_index()



long = data[["stays_in_weekend_nights", "stays_in_week_nights", "guest", "hotel"]].reset_index()

long["nights"] = long["stays_in_weekend_nights"] + long["stays_in_week_nights"]



plt.figure(figsize=(20, 7))

plt.subplot2grid((1,2), (0,0))

sns.barplot(x="nights", y="guest", data=long)

plt.title(label="Length of Nights", loc="center", fontsize=15)

plt.xlim(-1, 29)



plt.subplot2grid((1,2), (0,1))

long = long.groupby("nights").aggregate({"guest": "sum"}).reset_index()

sum = long["guest"].sum()

long["guest%"] = long["guest"]/sum

plt.bar(long["nights"], long["guest%"], width=0.5)

plt.title(label="Length of Nights in %", loc="center", fontsize=15)

plt.xlim(-1, 30)

plt.xticks(range(0,30,2))



plt.show()
data = data[data["is_canceled"]==0].reset_index()

segment = data[["hotel", "market_segment"]]



plt.figure(figsize=(12, 6))

segment = segment.groupby("market_segment").aggregate({"hotel": "count"}).reset_index()

segment = segment.sort_values(by=['hotel'], ascending=False)

plt.title(label="Market Segment", loc="center", fontsize=15)



sns.barplot(x="market_segment", y="hotel", data=segment)

plt.show()
data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv", header=0, index_col=0, encoding='utf-8')

data2 = data[data["is_canceled"]==0].reset_index()

segmentype = data2[["hotel", "market_segment", "reserved_room_type", "adr"]]

segmentype = segmentype.rename(columns={'reserved_room_type': 'room type'})



plt.figure(figsize=(15, 7))

plt.rc('axes', axisbelow=True)

plt.grid(axis="y", linestyle='-.')

sns.barplot(x="market_segment", y="adr",

            hue="room type",

            data=segmentype,

            ci="sd",

            capsize=0.1,

            errwidth=0.7)

plt.title(label="Adr in each segment & room_type", fontsize=15)

plt.legend(loc="upper right")



plt.show()


data3 = data.reset_index()

cancellations = data3[["hotel", "is_canceled", "arrival_date_year", "arrival_date_month"]]

sns.catplot(x="arrival_date_year", col="is_canceled",

            data=cancellations, hue="hotel", kind="count", height=4, aspect=.7, legend=False)

plt.legend(loc="upper right")



plt.show()
plt.figure(figsize=(13, 6))

data4 = data[data["is_canceled"]==1].reset_index()

cancellations = data4[["hotel", "is_canceled", "arrival_date_year", "arrival_date_month"]]

ordered_months = ["January", "February", "March", "April", "May", "June",

          "July", "August", "September", "October", "November", "December"]

cancellations["arrival_date_month"] = pd.Categorical(cancellations["arrival_date_month"], categories=ordered_months, ordered=True)



sns.countplot(x="arrival_date_month",  hue="hotel", data=cancellations)

plt.title(label="Cancellation each Month", fontsize=15)

plt.legend(loc="upper right")

plt.tight_layout()



plt.show()
corr = data.corr()["is_canceled"]



corr.sort_values()
sns.heatmap(data.corr())

plt.title(label="Correlation", fontsize=15)

plt.show()
from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression
data = data.fillna(({"agent": "0"}))

x=data[["lead_time", "previous_cancellations", "booking_changes", "required_car_parking_spaces",

        "total_of_special_requests", "is_repeated_guest", "agent", "adults",

        "previous_bookings_not_canceled", "babies"]]

y=data['is_canceled']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)



L = LogisticRegression()



L.fit(X_train, y_train)



y_pred = L.predict(X_test)

L.score(X_train, y_train)
from sklearn.metrics import confusion_matrix



matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(matrix, annot=True)



plt.show()

print(matrix)
from sklearn.metrics import accuracy_score



accuracy = accuracy_score(y_test, y_pred)



print(accuracy)