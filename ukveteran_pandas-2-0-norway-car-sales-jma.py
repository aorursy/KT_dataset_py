import pandas as pd

data=pd.read_csv("../input/norway-new-car-sales-by-model/norway_new_car_sales_by_model.csv",encoding="latin-1")
data.head(10)
# get data for Toyota Corolla

data.query("Model=='Toyota Corolla'")
# now find months where toyota corrola sold more than 100 cars

data.query("Model=='Toyota Corolla' and Quantity > 100")
data.query("Quantity > 500 and Make=='Toyota '")
data.head()
data.query("2010<Year<2015 & Make=='Toyota '")
data.query("2010<Year<2015 & Make=='Toyota '").Quantity.sum()
data.query("2010<Year<2015 & Make=='Toyota '").Quantity.sum()
data.head()
data.Quantity.sum()
data[["Quantity","Pct"]].sum()
data.head()
len(data.Model.unique())
result=data.Model
result.nunique()
data.head()
data.Model.isin(["Toyota Avensis","Audi A3","Volvo V70"])
data[data.Model.isin(["Toyota Avensis","Audi A3","Volvo V70"])]
data_toyota=data[data.Make=="Toyota "]
data_toyota.sort_values(by="Quantity",ascending=False)
data.head()
maketotal=data.pivot_table(values="Quantity",index=["Make","Model"],aggfunc="sum")
maketotal.head()
maketotal.loc["Audi ","Audi A3"]
data.Year.head()
data_yearly=data.pivot_table(values="Quantity",index=["Make","Year"],aggfunc="sum")
data_yearly.head()
data_yearly.loc["Toyota ",2015]
data_yearly=data.pivot_table(values="Quantity",index=["Make"],columns=["Year"],aggfunc="sum")
data_yearly.head()
data_yearly.head()
data.query("Make=='Citroen '")
data.head()
test=" text test "
test.strip()
data.Make.str.strip()
data.Make=data.Make.str.strip()
data.head()
data.query("Make=='Toyota'")
data.pivot_table(values="Quantity",index="Year",columns="Month",aggfunc="sum")