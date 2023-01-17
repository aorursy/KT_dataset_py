import pandas as pd
import numpy as np
import sqlite3 
con = sqlite3.connect('../input/database.sqlite')
raw = pd.read_sql('select UNITID, INSTNM, cast(LO_INC_DEBT_MDN as int) LO_INC_DEBT_MDN,\
cast(NOPELL_DEBT_MDN as int) NOPELL_DEBT_MDN, cast(PELL_DEBT_MDN as int) PELL_DEBT_MDN,\
 PAR_ED_PCT_1STGEN, cast(DEBT_MDN as int) DEBT_MDN, cast(md_earn_wne_p6 as int) md_earn_wne_p6 from Scorecard', con)
print("Dropping all 'PrivacySuppressed' schools")
raw = raw.drop_duplicates('UNITID')
debt = raw[(raw.PELL_DEBT_MDN > 0) & (raw.DEBT_MDN > 0) & (raw.NOPELL_DEBT_MDN > 0)]
print(len(debt)/float(len(raw))*100, "% of raw data remaining")
print(len(debt), "schools remain")
print(sum(debt.PELL_DEBT_MDN > debt.DEBT_MDN)/float(len(debt)*100),
      "% of median Pell grant debt higher than median debt at any school")
print(sum(debt.NOPELL_DEBT_MDN > debt.DEBT_MDN)/float(len(debt)*100),\
      "% of median non-Pell grant debt higher than meidian debt at any school")
print(sum(debt.PELL_DEBT_MDN > debt.NOPELL_DEBT_MDN)/float(len(debt))*100, \
      "% of median Pell grant debt higher than median non-Pell debt at any school")
# next, onto long-term earnings.
print("\n Only keeping 'six yr earning' data schools")
earn = raw[(raw.md_earn_wne_p6 > 0) & (raw.DEBT_MDN > 0)]
print(len(earn)/float(len(raw))*100, "% of raw data remaning")
print(len(earn), "schools remain")
print(sum(earn.DEBT_MDN > earn.md_earn_wne_p6)/float(len(earn))*100,\
      "% of median debt higher than annual median earnings (6yrs out) by school.")
low_inc = earn[(earn.LO_INC_DEBT_MDN > 0)]
print("\n Only keeping 'median low income debt' data schools")
print(sum(low_inc.LO_INC_DEBT_MDN > low_inc.md_earn_wne_p6)/float(len(low_inc))*100,\
      "% of median low income debt higher than median annual earnings (6yrs out) by school.")
print(sum(low_inc.md_earn_wne_p6 > 30000)/float(len(low_inc))*100,\
      "% of low income students made more than $30k, 6 years out")
print(sum(low_inc.md_earn_wne_p6 > 40000)/float(len(low_inc))*100,\
      "% of low income students made >$40, 6 years out")
print(sum(low_inc.md_earn_wne_p6 > 50000)/float(len(low_inc))*100,\
      "% of low income students made >$50K, 6 years out")
print("\n Let's check low_inc data for the online for-profits:")
df = low_inc.sort_values(by='md_earn_wne_p6', ascending=False)
print(df[['INSTNM','md_earn_wne_p6']].head(50))