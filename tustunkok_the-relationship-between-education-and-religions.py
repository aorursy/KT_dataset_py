# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
WORLD_RELIGIONS_DIR = "../input/world-religions/"

EDUCATION_STATISTICS_DIR = "../input/education-statistics/edstats-csv-zip-32-mb-/"
es_stats_series_df = pd.read_csv(os.path.join(EDUCATION_STATISTICS_DIR, "EdStatsSeries.csv"))

es_stats_country_df = pd.read_csv(os.path.join(EDUCATION_STATISTICS_DIR, "EdStatsCountry.csv"))

es_stats_footnote_df = pd.read_csv(os.path.join(EDUCATION_STATISTICS_DIR, "EdStatsFootNote.csv"))

es_stats_country_series_df = pd.read_csv(os.path.join(EDUCATION_STATISTICS_DIR, "EdStatsCountry-Series.csv"))

es_stats_data_df = pd.read_csv(os.path.join(EDUCATION_STATISTICS_DIR, "EdStatsData.csv"))
es_stats_series_df.head()
es_stats_series_df.iloc[0]["Short definition"]
es_stats_country_df.head()
es_stats_footnote_df.head()
es_stats_country_series_df.head()
es_stats_data_df.head()
es_stats_data_prj_rmvd_df = es_stats_data_df.drop([str(year) for year in range(2020, 2101, 5)] + ["Unnamed: 69"], axis=1)

es_stats_data_prj_rmvd_df.head()
es_stats_series_df[["Series Code", "Short definition"]].values[:5]
wr_regional_df = pd.read_csv(os.path.join(WORLD_RELIGIONS_DIR, "regional.csv"))

wr_national_df = pd.read_csv(os.path.join(WORLD_RELIGIONS_DIR, "national.csv"))

wr_global_df = pd.read_csv(os.path.join(WORLD_RELIGIONS_DIR, "global.csv"))
print("Unique years:", wr_regional_df["year"].unique())

print("Unique regions:", wr_regional_df["region"].unique())
wr_regional_df.head()
print(f"Total of {len(wr_regional_df)} rows.")
wr_national_df.head()
print(f"Total of {len(wr_national_df)} rows.")
wr_global_df.head()
print(f"Total of {len(wr_global_df)} rows.")
wr_national_df[wr_national_df["year"] == 2010].nlargest(20, "religion_sumpercent")
es_stats_data_prj_rmvd_df[es_stats_data_prj_rmvd_df["Indicator Code"] == "BAR.NOED.15UP.ZS"].nlargest(20, "2010")
lar_relig_pop_countries_df = wr_national_df[wr_national_df["year"] == 2010].nlargest(20, "religion_sumpercent")

low_ed_levels_countries_df = es_stats_data_prj_rmvd_df[es_stats_data_prj_rmvd_df["Indicator Code"] == "BAR.NOED.15UP.ZS"].nlargest(20, "2010")



set(lar_relig_pop_countries_df["code"].values) & set(low_ed_levels_countries_df["Country Code"].values)
wr_national_df[wr_national_df["year"] == 2010].nsmallest(20, "religion_sumpercent")
es_stats_data_prj_rmvd_df[es_stats_data_prj_rmvd_df["Indicator Code"] == "BAR.NOED.15UP.ZS"].nsmallest(20, "2010")
smal_relig_pop_countries_df = wr_national_df[wr_national_df["year"] == 2010].nsmallest(20, "religion_sumpercent")

high_ed_levels_countries_df = es_stats_data_prj_rmvd_df[es_stats_data_prj_rmvd_df["Indicator Code"] == "BAR.NOED.15UP.ZS"].nsmallest(20, "2010")



set(smal_relig_pop_countries_df["code"].values) & set(high_ed_levels_countries_df["Country Code"].values)
smal_relig_pop_countries_df = wr_national_df[wr_national_df["year"] == 2010]

high_ed_levels_countries_df = es_stats_data_prj_rmvd_df[es_stats_data_prj_rmvd_df["Indicator Code"] == "BAR.NOED.15UP.ZS"].drop([str(year) for year in range(1970, 2010)] + [str(year) for year in range(2011, 2018)], axis=1)
high_ed_levels_countries_df.rename(columns={"Country Code": "code"}, inplace=True)
small_relig_high_ed_countries_df = pd.merge(smal_relig_pop_countries_df, high_ed_levels_countries_df, on="code")[["year", "state", "code", "Country Name", "2010"] + [x for x in smal_relig_pop_countries_df.columns if x.endswith("_percent")]]

small_relig_high_ed_countries_df.head()
ed_rel_corr_df = small_relig_high_ed_countries_df.drop("year", axis=1).corr()[["2010"]].dropna().drop("2010")

ed_rel_corr_df
np.argmax(np.absolute(ed_rel_corr_df))
ed_rel_corr_df.iloc[np.argmax(np.absolute(ed_rel_corr_df))]