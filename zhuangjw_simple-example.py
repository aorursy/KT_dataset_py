import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr

import seaborn as sns
from ipywidgets import interact, SelectionSlider, IntSlider

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
spec_names = ['CH2I2', 'CH2ICl', 'CH2IBr', 'AERI', 'CO2', 'INDIOL', 'ISALA', 'ISALC', 'ISN1OA', 'ISN1OG', 'LBRO2H', 'LBRO2N', 'LISOPOH', 'LISOPNO3', 'LTRO2H', 'LTRO2N', 'LVOCOA', 'LVOC', 'LXRO2H', 'LXRO2N', 'MSA', 'PYAC', 'SO4H1', 'SO4H2', 'SOAGX', 'SOAIE', 'SOAME', 'IMAE', 'SOAMG', 'POx', 'LOx', 'PCO', 'LCO', 'PSO4', 'LCH4', 'PH2O2', 'I2O4', 'DHDN', 'DHDC', 'I2O2', 'MONITA', 'BENZ', 'CH3CCl3', 'CH3I', 'H1301', 'H2402', 'I2O3', 'PMNN', 'PPN', 'TOLU', 'BrNO2', 'CCl4', 'CFC11', 'CFC12', 'CFC113', 'CFC114', 'CFC115', 'H1211', 'IBr', 'IEPOXD', 'INO', 'N2O', 'TRO2', 'BRO2', 'IEPOXA', 'IEPOXB', 'IONITA', 'N', 'OCS', 'XRO2', 'HI', 'MAP', 'CHBr3', 'ICl', 'CH2Cl2', 'IMAO3', 'CHCl3', 'MPN', 'Cl2O2', 'CH2Br2', 'ETP', 'HCFC123', 'ClNO2', 'HCFC141b', 'HCFC142b', 'IONO', 'HCFC22', 'OIO', 'RA3P', 'RB3P', 'XYLE', 'DMS', 'CH3Cl', 'CH3Br', 'HNO4', 'ClOO', 'HNO2', 'OClO', 'PAN', 'RP', 'PP', 'PRPN', 'SO4', 'ALK4', 'PIP', 'R4P', 'HPALD', 'BrCl', 'C3H8', 'DHPCARP', 'HOI', 'IAP', 'HPC52O2', 'VRP', 'ATOOH', 'Br2', 'HC187', 'MOBA', 'HONIT', 'DHMOB', 'RIPB', 'BrSALC', 'ISNP', 'MP', 'BrSALA', 'MAOP', 'MRP', 'RIPA', 'RIPD', 'EOH', 'ETHLN', 'N2O5', 'INPN', 'MTPA', 'MTPO', 'NPMN', 'C2H6', 'IONO2', 'MOBAOO', 'DIBOO', 'IPMN', 'LIMO', 'H', 'BrNO3', 'MACRNO2', 'ROH', 'I2', 'MONITS', 'Cl2', 'ISOPNB', 'CH4', 'ISNOHOO', 'MVKOO', 'ISNOOB', 'GAOO', 'CH3CHOO', 'IEPOXOO', 'GLYX', 'MVKN', 'MGLYOO', 'PRN1', 'MONITU', 'MGLOO', 'A3O2', 'PROPNN', 'MAN2', 'ISNOOA', 'PO2', 'ISOPNDO2', 'HCOOH', 'B3O2', 'MACROO', 'R4N1', 'ISOP', 'MAOPO2', 'H2O2', 'ATO2', 'I', 'RCO3', 'LIMO2', 'MACRN', 'OLND', 'OLNN', 'IO', 'KO2', 'HOBr', 'ISOPNBO2', 'PIO2', 'HC5OO', 'HNO3', 'ISOPND', 'GLYC', 'NMAO3', 'ACTA', 'VRO2', 'HOCl', 'CH2OO', 'ISN1', 'ClNO3', 'MGLY', 'ACET', 'HC5', 'RIO2', 'ETO2', 'INO2', 'R4O2', 'R4N2', 'HAC', 'MRO2', 'BrO', 'PRPE', 'RCHO', 'MEK', 'CH2O', 'MACR', 'ALD2', 'MVK', 'MCO3', 'SO2', 'CO', 'MO2', 'Br', 'NO', 'HBr', 'HCl', 'O1D', 'Cl', 'O', 'NO3', 'NO2', 'O3', 'HO2', 'ClO', 'OH', 'H2O', 'H2', 'MOH', 'N2', 'O2', 'RCOOH']
redundant_specs = ['LBRO2H', 'LBRO2N', 'LISOPOH', 'LISOPNO3', 'LTRO2H', 'LTRO2N', 'LXRO2H', 'LXRO2N', 'CO2', 'H2O', 'H2', 'MOH', 'N2', 'O2', 'RCOOH', 'POx', 'LOx', 'PCO', 'LCO', 'PSO4', 'LCH4', 'PH2O2']
used_spec = [s for s in spec_names if s not in redundant_specs]
jval_names = ['1/O2/O2', '2/O3/O3', '3/O3/O3(1D)', '4/H2O/H2O', '5/HO2/HO2', '6/NO/NO', '7/CH2O/H2COa', '8/CH2O/H2COb', '9/H2O2/H2O2', '10/MP/CH3OOH', '11/NO2/NO2', '12/NO3/NO3', '13/NO3/NO3', '14/N2O5/N2O5', '15/HNO2/HNO2', '16/HNO3/HNO3', '17/HNO4/HNO4', '18/HNO4/HNO4', '19/ClNO3/ClNO3a', '20/ClNO3/ClNO3b', '21/ClNO2/ClNO2', '22/Cl2/Cl2', '23/Br2/Br2', '24/HOCl/HOCl', '25/OClO/OClO', '26/Cl2O2/Cl2O2', '27/ClO/ClO', '28/BrO/BrO', '29/BrNO3/BrNO3', '30/BrNO3/BrNO3', '31/BrNO2/BrNO2', '32/HOBr/HOBr', '33/BrCl/BrCl', '34/OCS/OCS', '35/SO2/SO2', '36/N2O/N2O', '37/CFC11/CFCl3', '38/CFC12/CF2Cl2', '39/CFC113/F113', '40/CFC114/F114', '41/CFC115/F115', '42/CCl4/CCl4', '43/CH3Cl/CH3Cl', '44/CH3CCl3/MeCCl3', '45/CH2Cl2/CH2Cl2', '46/HCFC22/CHF2Cl', '47/HCFC123/F123', '48/HCFC141b/F141b', '49/HCFC142b/F142b', '50/CH3Br/CH3Br', '51/H1211/H1211', '52/H12O2/H1211', '53/H1301/H1301', '54/H2402/H2402', '55/CH2Br2/CH2Br2', '56/CHBr3/CHBr3', None, '58/CF3I/CF3I', '59/PAN/PAN', '60/R4N2/CH3NO3', '61/ALD2/ActAld', '62/ALD2/ActAlx', '63/MVK/MeVK', '64/MVK/MeVK', '65/MVK/MeVK', '66/MACR/MeAcr', '67/MACR/MeAcr', '68/GLYC/GlyAld', '69/MEK/MEKeto', '70/RCHO/PrAld', '71/MGLY/MGlyxl', '72/GLYX/Glyxla', '73/GLYX/Glyxlb', '74/GLYX/Glyxlc', '75/HAC/HAC', '76/ACET/Acet-a', '77/ACET/Acet-b', '78/INPN/CH3OOH', '79/PRPN/CH3OOH', '80/ETP/CH3OOH', '81/RA3P/CH3OOH', '82/RB3P/CH3OOH', '83/R4P/CH3OOH', '84/PP/CH3OOH', '85/RP/CH3OOH', '86/RIP/CH3OOH', '87/IAP/CH3OOH', '88/ISNP/CH3OOH', '89/VRP/CH3OOH', '90/MRP/CH3OOH', '91/MAOP/CH3OOH', '92/MACRN/MACRN', '93/MVKN/MVKN', '94/ISOPNB/ONIT1', '95/ISOPND/ONIT1', '96/PROPNN/PROPNN', '97/ATOOH/CH3OOH', '98/R4N2/CH3NO3', '99/MAP/CH3OOH', '100/SO4/H2SO4', '101/ClNO2/ClNO2', '102/ClOO/ClOO', '103/O3/O3(1D)', '104/MPN/MPN', '105/MPN/MPN', '106/PIP/H2O2', '107/IPMN/PAN', '108/ETHLN/ETHLN', '109/DHDC/MeAcr', '110/HPALD/MeAcr', '111/ISN1/MeAcr', '112/MONITS/ONIT1', '113/MONITU/ONIT1', '114/HONIT/ONIT1', '115/I2/I2', '116/HOI/HOI', '117/IO/IO', '118/OIO/OIO', '119/INO/INO', '120/IONO/IONO', '121/IONO2/IONO2', '122/I2O2/I2O2', '123/CH3I/CH3I', '124/CH2I2/CH2I2', '125/CH2ICl/CH2ICl', '126/CH2IBr/CH2IBr', '127/I2O4/I2O2', '128/I2O3/I2O3', '129/IBr/IBr', '130/ICl/ICl']
used_jvals = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 58, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 85, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
phy_names = ['TEMP', 'PRESS', 'NUMDEN', 'H2O']
ds = xr.open_dataset('../input/KPP_fields_007.nc')
ds
ds.coords['nspec'] = xr.DataArray(spec_names, dims='nspec')
ds
ds['C_after'].sel(nspec='O3')[0].plot()
# only get lower troposphere
ds_sub = ds.isel(lev=slice(0, 20)).stack(sample=('lev', 'lat', 'lon')).transpose()
ds_sub
df_c_before = pd.DataFrame(ds_sub['C_before'].values, columns=spec_names)[used_spec]
df_c_before.head()
df_c_after = pd.DataFrame(ds_sub['C_after'].values, columns=spec_names)[used_spec]
df_c_after.head()
df_jval = pd.DataFrame(ds_sub['PHOTOL'].values, columns=jval_names).iloc[:,used_jvals]
df_jval.head()
df_phy = pd.DataFrame(ds_sub['PHY'].values, columns=phy_names)
df_phy.head()
df_c_before.describe(percentiles=[0.1, 0.9])
%%time
df_c_corr = df_c_before.corr()
df_c_corr.head()
sns.heatmap(df_c_corr, vmin=0, vmax=1, cmap='Reds')
df_c_corr['OH'].nlargest(10)
pd.plotting.scatter_matrix(df_c_before[['OH', 'Cl', 'H', 'O1D']], alpha=0.2, s=2);
df_c_corr['CFC115'].nlargest(10)
pd.plotting.scatter_matrix(df_c_before[['CFC115', 'N2O', 'OCS', 'CH4']], alpha=0.2, s=2);
df_jval.describe()
%%time
df_jval_corr = df_jval.corr()
sns.heatmap(df_jval_corr, vmin=0, vmax=1, cmap='Reds')
