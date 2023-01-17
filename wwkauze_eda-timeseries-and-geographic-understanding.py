import os
import glob

import pandas as pd
import numpy as np
from itertools import chain

import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMapWithTime
from IPython.display import HTML

plt.rcParams['figure.figsize'] = [10, 10]

# dataset read
olist_orders_df = pd.read_csv("../input/olist_orders_dataset.csv")
olist_order_items_df = pd.read_csv("../input/olist_order_items_dataset.csv")
olist_customers_df = pd.read_csv("../input/olist_customers_dataset.csv")
olist_geolocation_df = pd.read_csv("../input/olist_geolocation_dataset.csv")

groupby_month_zipcode_latlng_df = pd.merge(
            olist_orders_df,
            olist_customers_df,
            on = ["customer_id"],
            how = "left"
        ).\
        pipe(
            lambda df:
            pd.merge(
                df,
                olist_order_items_df,
                on = ["order_id"],
                how = "left"
            )
        ).\
        rename(columns={
            "customer_zip_code_prefix":"zip_code_prefix",
            "customer_city":"city",
            "customer_state":"state"
        }).\
        pipe(lambda df:
            df.assign(
                year_month = df["order_purchase_timestamp"].str[0:7]               
        )).\
        groupby([
            "zip_code_prefix",
            "city",
            "state",
            "year_month",
            "order_id"
        ])[
            "price",
            "freight_value"
        ].sum().reset_index().\
        assign(
            order_count = 1
        ).\
        groupby([
            "zip_code_prefix",
            "city",
            "state",
            "year_month"
        ])[
            "price",
            "freight_value",
            "order_count"
        ].sum().reset_index().\
    pipe(lambda df:
        pd.merge(
            df,
            olist_geolocation_df.\
            rename(columns={
                "geolocation_zip_code_prefix":"zip_code_prefix",
                "geolocation_lat":"lat",
                "geolocation_lng":"lng",
                "geolocation_city":"city",
                "geolocation_state":"state"
            }).\
            groupby([
                    "zip_code_prefix",
                    "city",
                    "state"
                ])["lat","lng"].mean().reset_index(),
            on=["zip_code_prefix","city","state"]
    )
)

time_index,for_heat_map_data = groupby_month_zipcode_latlng_df.\
    pipe(lambda df:
        df.assign(order_count_for_hm = df["order_count"]/10)
    ).\
    groupby("year_month")["lat","lng","order_count_for_hm"].\
    apply(lambda row: row.values.tolist()).\
    pipe(lambda df:
        (
            df.reset_index()["year_month"].tolist(),
            df.tolist()
        )
    )

center_map = [
    groupby_month_zipcode_latlng_df["lat"].mean(),
    groupby_month_zipcode_latlng_df["lng"].mean()
]

map1 = folium.Map(location=center_map,tiles='stamentoner',zoom_start=4.0)

heat_map = HeatMapWithTime(
    for_heat_map_data,
    index = time_index,
    auto_play = True,
    radius = 10,
    max_opacity = 0.5
).add_to(map1)

map1
import os
import glob

import pandas as pd
import numpy as np
from itertools import chain

import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMapWithTime
from IPython.display import HTML

plt.rcParams['figure.figsize'] = [10, 10]

# dataset read
olist_orders_df = pd.read_csv("../input/olist_orders_dataset.csv")
olist_order_items_df = pd.read_csv("../input/olist_order_items_dataset.csv")
olist_customers_df = pd.read_csv("../input/olist_customers_dataset.csv")
olist_geolocation_df = pd.read_csv("../input/olist_geolocation_dataset.csv")

# group by city (groupby_city_df) : For Table 1-2
groupby_city_df = pd.merge(
            olist_orders_df,
            olist_customers_df,
            on = ["customer_id"],
            how = "left"
        ).\
        pipe(
            lambda df:
            pd.merge(
                df,
                olist_order_items_df,
                on = ["order_id"],
                how = "left"
            )
        ).\
        rename(columns={
                    "customer_zip_code_prefix":"zip_code_prefix",
                    "customer_city":"city",
                    "customer_state":"state"
                }).\
        groupby([
            "zip_code_prefix",
            "city",
            "state",
            "order_id"
        ])[
            "price",
            "freight_value",
        ].sum().reset_index().\
        assign(
            order_count = 1
        ).\
        groupby([
            "zip_code_prefix",
            "city",
            "state",
        ])[
            "price",
            "freight_value",
            "order_count"
        ].sum().reset_index().\
        pipe(lambda df:
            pd.merge(  
                df,
                df.groupby([
                        "city"
                    ])["order_count"].sum().reset_index().\
                    sort_values("order_count",ascending=False).\
                    pipe(lambda df:
                        df.assign(
                                order_count_cumsum = df["order_count"].cumsum()/sum(df["order_count"])
                            )
                    ).\
                    reset_index(drop=True).\
                    reset_index().\
                    pipe(lambda df:
                        df.assign(
                                city_other = df.apply(
                                    lambda row: 
                                    str(row["index"]).zfill(2)+ "_"+ row["city"] if row["index"]<10 else "99_other",
                                    axis=1
                                )
                            )
                    ).loc[:,["city","city_other"]],
                on=["city"]
            )
    )

# group by month and city(groupby_month_city_df) : For Chart 1-2
groupby_month_city_df = pd.merge(
            olist_orders_df,
            olist_customers_df,
            on = ["customer_id"],
            how = "left"
        ).\
        pipe(
            lambda df:
            pd.merge(
                df,
                olist_order_items_df,
                on = ["order_id"],
                how = "left"
            )
        ).\
        rename(columns={
                    "customer_zip_code_prefix":"zip_code_prefix",
                    "customer_city":"city",
                    "customer_state":"state"
        }).\
        pipe(lambda df:
            df.assign(
                year_month = df["order_purchase_timestamp"].str[0:7]              
        )).\
        groupby([
            "city",
            "state",
            "year_month",
            "order_id"
        ])[
            "price",
            "freight_value"
        ].sum().reset_index().\
        assign(
            order_count = 1
        ).\
        groupby([
            "city",
            "state",
            "year_month"
        ])[
            "price",
            "freight_value",
            "order_count"
        ].sum().reset_index().\
    pipe(lambda df:
        pd.merge(
            df,
            olist_geolocation_df.\
                rename(columns={
                    "geolocation_zip_code_prefix":"zip_code_prefix",
                    "geolocation_lat":"lat",
                    "geolocation_lng":"lng",
                    "geolocation_city":"city",
                    "geolocation_state":"state"
                }).\
                groupby([
                        "city",
                        "state"
                    ])["lat","lng"].mean().reset_index(),
            on=["city","state"]
        )
    ).\
    pipe(lambda df:
        pd.merge(
            df,
            df.groupby([
                    "city"
                ])["order_count"].sum().reset_index().\
                sort_values("order_count",ascending=False).\
                pipe(lambda df:
                    df.assign(
                            order_count_cumsum = df["order_count"].cumsum()/sum(df["order_count"])
                        )
                ).\
                reset_index(drop=True).\
                reset_index().\
                pipe(lambda df:
                    df.assign(
                            city_other = df.apply(
                                lambda row: 
                                str(row["index"]).zfill(2)+ "_"+ row["city"] if row["index"]<10 else "99_other",
                                axis=1
                            )
                        )
                ).loc[:,["city","city_other"]],
            on=["city"]
        )
    )

# group by month and zip_code with latlng(groupby_month_zipcode_latlng_df) : For Map 1
groupby_month_zipcode_latlng_df = pd.merge(
            olist_orders_df,
            olist_customers_df,
            on = ["customer_id"],
            how = "left"
        ).\
        pipe(
            lambda df:
            pd.merge(
                df,
                olist_order_items_df,
                on = ["order_id"],
                how = "left"
            )
        ).\
        rename(columns={
            "customer_zip_code_prefix":"zip_code_prefix",
            "customer_city":"city",
            "customer_state":"state"
        }).\
        pipe(lambda df:
            df.assign(
                year_month = df["order_purchase_timestamp"].str[0:7]               
        )).\
        groupby([
            "zip_code_prefix",
            "city",
            "state",
            "year_month",
            "order_id"
        ])[
            "price",
            "freight_value"
        ].sum().reset_index().\
        assign(
            order_count = 1
        ).\
        groupby([
            "zip_code_prefix",
            "city",
            "state",
            "year_month"
        ])[
            "price",
            "freight_value",
            "order_count"
        ].sum().reset_index().\
    pipe(lambda df:
        pd.merge(
            df,
            olist_geolocation_df.\
            rename(columns={
                "geolocation_zip_code_prefix":"zip_code_prefix",
                "geolocation_lat":"lat",
                "geolocation_lng":"lng",
                "geolocation_city":"city",
                "geolocation_state":"state"
            }).\
            groupby([
                    "zip_code_prefix",
                    "city",
                    "state"
                ])["lat","lng"].mean().reset_index(),
            on=["zip_code_prefix","city","state"]
    )
)
print("【Table 1】: Top10 City EDA index")
groupby_city_df.\
    groupby("city_other")[
        "order_count",
        "price",
        "freight_value"
    ].sum()
print("【Table 2】: Top10 City EDA index Percent (%)")
groupby_city_df.\
    groupby("city_other")[
        "order_count",
        "price",
        "freight_value"
    ].sum().\
    pipe(
        lambda df: df.divide(df.sum(axis=0), axis=1)*100
    )
print("【Chart 1】Top 10 City TimeSeries order_count value stacked area chart")
chart1 = pd.pivot_table(
    groupby_month_city_df,
    index="year_month",
    columns="city_other",
    values="order_count",
    aggfunc=np.sum
).plot(kind="area", stacked=True, title="Top 10 City TimeSeries order_count value stacked area chart")
chart1.legend(loc="upper left",bbox_to_anchor=(1, 1),borderaxespad=0, fontsize=10)
chart1.set_ylabel("order count")
plt.show()
print("【Chart 2】Top 10 City TimeSeries order_count 100 % stacked area chart")
chart2 = pd.pivot_table(
    groupby_month_city_df,
    index="year_month",
    columns="city_other",
    values="order_count",
    aggfunc=np.sum
).\
pipe(
    lambda df: df.divide(df.sum(axis=1), axis=0)*100
).plot(kind="area", stacked=True, title="Top 10 City TimeSeries order_count 100 % stacked area chart",ylim=[0,100])
chart2.legend(loc="upper left",bbox_to_anchor=(1, 1),borderaxespad=0, fontsize=10)
chart2.set_ylabel("Percent (%)")
plt.show()
time_index,for_heat_map_data = groupby_month_zipcode_latlng_df.\
    pipe(lambda df:
        df.assign(order_count_for_hm = df["order_count"]/10)
    ).\
    groupby("year_month")["lat","lng","order_count_for_hm"].\
    apply(lambda row: row.values.tolist()).\
    pipe(lambda df:
        (
            df.reset_index()["year_month"].tolist(),
            df.tolist()
        )
    )

center_map = [
    groupby_month_zipcode_latlng_df["lat"].mean(),
    groupby_month_zipcode_latlng_df["lng"].mean()
]

map1 = folium.Map(location=center_map,tiles='stamentoner',zoom_start=4.0)

heat_map = HeatMapWithTime(
    for_heat_map_data,
    index = time_index,
    auto_play = True,
    radius = 10,
    max_opacity = 0.5
).add_to(map1)

map1
print("【Chart 3】Top 10 City TimeSeries order_count value stacked area chart(excluding 99_other)")

chart3 = pd.pivot_table(
    groupby_month_city_df[groupby_month_city_df["city_other"] != "99_other"],
    index="year_month",
    columns="city_other",
    values="order_count",
    aggfunc=np.sum
).plot(kind = "area", stacked=True, title="Top 10 City TimeSeries order_count value stacked area chart(excluding 99_other)")
chart3.legend(loc="upper left",bbox_to_anchor=(1, 1),borderaxespad=0, fontsize=10)
chart3.set_ylabel("order count")
plt.show()
print("【Chart 4】Top 10 City TimeSeries order_count 100 % stacked area chart(excluding 99_other)")
chart_4 = pd.pivot_table(
        groupby_month_city_df[groupby_month_city_df["city_other"] != "99_other"],
        index="year_month",
        columns="city_other",
        values="order_count",
        aggfunc=np.sum
    ).\
    pipe(
        lambda df: df.divide(df.sum(axis=1), axis=0)*100
    ).plot(kind="area", stacked=True, title="Top 10 City TimeSeries order_count 100 % stacked area chart(excluding 99_other)",ylim=[0,100])
chart_4.legend(loc="upper left",bbox_to_anchor=(1, 1),borderaxespad=0, fontsize=10)
chart_4.set_ylabel("Percent (%)")
plt.show()