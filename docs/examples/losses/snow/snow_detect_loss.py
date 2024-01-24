"""
Snow detection and loss quantification
======================================
Identifying periods of snow cover by comparing measured irradiance to measured
current and voltage. Quantifying snow losses as losses incurred under
conditions that are suggestive of snow cover.

"""
# %%
import pathlib
import os
import sys
import json
import pandas as pd
import numpy as np
import re
import pvlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches

base_dir = os.path.join(os.path.expanduser("~"), 'pvanalytics')
sys.path.append(base_dir)

import pvanalytics
from pvanalytics.features.clipping import geometric
from pvanalytics.features import snow

##%%
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
example_data_dir = pathlib.Path.joinpath(pvanalytics_dir, 'data', 'snow_demo_data')

##%% Read in utility-scale data - includes measurements of current, voltage,
# plane-of-array irradiance, and module temperature

data_dir = pathlib.Path.joinpath(example_data_dir, 'data.csv')
data = pd.read_csv(data_dir, index_col='Timestamp')
data.set_index(pd.DatetimeIndex(data.index,ambiguous='infer'), inplace=True)
data = data[~data.index.duplicated()]

## %%
# Explore the dataset
print('Utility-scale dataset')
print('Start: {}'.format(data.index[0]))
print('End: {}'.format(data.index[-1]))
print('Frequency: {}'.format(data.index.inferred_freq))
data.head()

## %% General housekeeping

# Identify current, voltage, and AC power columns
dc_voltage_cols = [c for c in data.columns if 'Voltage' in c]
dc_current_cols = [c for c in data.columns if 'Current' in c]
ac_power_cols = [c for c in data.columns if 'AC' in c]

# Set negative current, voltage values to zero
data.loc[:, dc_voltage_cols] = np.maximum(data[dc_voltage_cols], 0)
data.loc[:, dc_current_cols] = np.maximum(data[dc_current_cols], 0)
data.loc[:, ac_power_cols] = np.maximum(data[ac_power_cols], 0)

# Set NaN current, voltage values to zero
data.loc[:, dc_voltage_cols] = data[dc_voltage_cols].replace({np.nan: 0, None: 0})
data.loc[:, dc_current_cols] = data[dc_current_cols].replace({np.nan: 0, None: 0})
data.loc[:, ac_power_cols] = data[ac_power_cols].replace({np.nan: 0, None: 0})

## %% Read in system configuration parameters
config_dir = pathlib.Path.joinpath(example_data_dir, 'config.json')
with open(config_dir) as json_data:
    config = json.load(json_data)

##%% Plot DC voltage in comparison to inverter rating
fig, ax = plt.subplots(figsize=(10,10))                  
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
col='INV1 CB2 Voltage [V]'
ax.scatter(data.index, data[col], s=0.5, c='b')
ax.plot(data.index, data[col], alpha=0.2, c='b')
ax.axhline(float(config['max_dcv']), c='r', ls='--', label='Maximum allowed voltage: {}'.format(config['max_dcv']))
ax.axhline(float(config['min_dcv']), c='g', ls='--', label='Inverter turn-on voltage: {}'.format(config['min_dcv']))
ax.set_xlabel('Date', fontsize='large')
ax.set_ylabel(col, fontsize='large', c='b')
ax.legend()

## %% Identify periods with non-unity power factor (either outside of inverter rating or at Voc)
for v, i in zip(dc_voltage_cols, dc_current_cols):

    # Data where V > rating
    data.loc[(data[v] > float(config['max_dcv'])), v] = np.nan
    data.loc[(data[v] > float(config['max_dcv'])), i] = np.nan
    
    # Data where V < turn-on voltage
    data.loc[data[v] < float(config['min_dcv']), v] = 0
    data.loc[data[v] < float(config['min_dcv']), i] = 0
    
    # Data where system is at Voc
    data.loc[data[i] == 0, v] = 0

## %% Plot AC power in comparison to inverter rating
fig, ax = plt.subplots(figsize=(10,10))                         
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
col = 'INV1 AC Power [W]'
ax.scatter(data.index, data[col], s=0.5, c='g')
ax.plot(data.index, data[col], alpha=0.2, c='g')
ax.axhline(float(config['max_ac']), c='r', ls='--', label='Inverter rating: {}'.format(config['max_ac'] + ' kW'))
ax.set_ylabel(col, fontsize='large', c='g')
ax.legend()

## %% Identify and mask periods of clipping
for c in ac_power_cols:
    
    mask1 = data[c] > float(config['max_ac'])
    mask2 = geometric(ac_power=data[c], freq='15T')
    mask3 = np.logical_or(mask1.values, mask2.values)

    inv_num = re.match('INV(\d+)', c).group(1)
    v_cols = [c for c in dc_voltage_cols if inv_num in c]
    i_cols = [c for c in dc_current_cols if inv_num in c]

    data.loc[mask3, c] = np.nan
    data.loc[mask3, v_cols] = np.nan
    data.loc[mask3, i_cols] = np.nan

## %% Load in and apply horizon mask
    
mask_dir = os.path.join(example_data_dir, 'mask.csv')
horizon_mask = pd.read_csv(mask_dir, index_col='Unnamed: 0')

def apply_mask(mask, x ,y):
    if np.isnan(x) == False:
        if y > mask.at[int(np.floor(x)), '0']:
            return False
        else:
            return True
    else:
        return np.nan

data.loc[:, 'Horizon Mask'] = data.apply(lambda x: apply_mask(horizon_mask, x['azimuth'], x['elevation']), axis = 1)

data = data[data['Horizon Mask'] == False]
data = data[data['elevation'] > 0]

## %% Define SAPM coefficients for modeling transmission

coeffs = config['sapm_coeff']
imp0 = float(coeffs['Impo'])
vmp0 = float(coeffs['Vmpo'])
c0 = float(coeffs['C0'])
c1 = float(coeffs['C1'])
c2 = float(coeffs['C2'])
c3 = float(coeffs['C3'])
beta_vmp = float(coeffs['Bvmpo'])
alpha_imp = float(coeffs['Aimp'])
n = float(coeffs['N'])
ns = float(coeffs['Cells_in_Series'])

#%%

"""
Model cell temperature using procedure outlined in Eqn. 12 of [1]
and model effective irradiance using Eqn. 23 of [2]. An incidence
angle modifier is applied to effective irradiance.

[1] King, D.L., E.E. Boyson, and J.A. Kratochvil, Photovoltaic Array
Performance Model, SAND2004-3535, Sandia National Laboratories,
Albuquerque, NM, 2004.
[2] B. H. King, C. W. Hansen, D. Riley, C. D. Robinson and L. Pratt,
“Procedure to Determine Coefficients for the Sandia Array Performance
Model (SAPM)," SAND2016-5284, June 2016.
"""

e_0 = 1000
iam = pvlib.iam.ashrae(data['aoi'])
data['Effective irradiance [W/m²]'] = iam*data['POA [W/m²]']/e_0
data['Cell Temp [C]'] = data['Module Temp [C]'] + 3*(data['Effective irradiance [W/m²]'])

# %% Plot cell temperature
fig, ax = plt.subplots(figsize=(10,10))                        
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
ax.scatter(data.index, data['Cell Temp [C]'], s=0.5, c='b')
ax.set_ylabel('Cell Temp [C]', c='b', fontsize='xx-large')
ax.set_xlabel('Date', fontsize='xx-large')

# %% Demonstrate transmission calculation
j = 0
v = dc_voltage_cols[j]
i = dc_current_cols[j]
matched = re.match(r'INV(\d+) CB(\d+)', i)
inv_cb = matched.group(0)

i_scaling_factor = int(config['num_str_per_cb'][f'{inv_cb}'])

#  p = np.array([_a_func(cell_temp,imp0, c1,alpha_imp), _b_func(cell_temp,imp0, c0, alpha_imp),_c_func(current)])

T1, T2 = snow.get_transmission(data['Cell Temp [C]'].values, data[i].values,
                                       data['Effective irradiance [W/m²]'].values,
                                       i_scaling_factor, imp0, c0, c1, alpha_imp, 25)

name_T1 = inv_cb + ' Transmission'
data[name_T1] = T1

# %% Plot transmission
fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

ax.scatter(temp.index, temp[name_T1], s=0.5, c='b')
ax.set_ylabel(name_T1, c='b', fontsize='xx-large')
ax.set_xlabel('Date', fontsize='large')

# %% Model voltage using calculated transmission

v_scaling_factor = int(config['num_mods_per_str'][inv_cb])
modeled_vmp = pvlib.pvsystem.sapm(e_0*data['Effective irradiance [W/m²]']*T1,
                                  data['Cell Temp [C]'],
                                  coeffs)['v_mp']

name_modeled_vmp = inv_cb + ' Predicted Vmp from Imp'
data[name_modeled_vmp] = v_scaling_factor*modeled_vmp

# %% Plot modeled and measured voltage
fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

ax.scatter(temp.index, temp[name_modeled_vmp], s=1, c='b', label='Modeled')
ax.scatter(temp.index, temp[inv_cb + ' Voltage [V]'], s=1, c = 'g', label='Measured')
ax.legend(fontsize='xx-large')
ax.set_ylabel('Voltage [V]', fontsize='xx-large')
ax.set_xlabel('Date', fontsize='large')
# %% Calculate transmission and model voltage for all other columns

for v_col, i_col in zip(dc_voltage_cols, dc_current_cols):
    
    matched = re.match(r'INV(\d+) CB(\d+) Current', i_col)
    inv_num = matched.group(1)
    cb_num = matched.group(2)
    inv_cb = f'INV{inv_num} CB{cb_num}' 
    
    v_scaling_factor = int(config['num_mods_per_str'][inv_cb])
    i_scaling_factor = int(config['num_str_per_cb'][f'INV{inv_num} CB{cb_num}'])
    
    T1, T2 = snow.get_transmission(data['Cell Temp [C]'].values, data[i_col].values,
                                       data['Effective irradiance [W/m²]'].values,
                                       i_scaling_factor, imp0, c0, c1, alpha_imp, 25)
    name_T1 = inv_cb + ' Transmission'
    data[name_T1] = T1

    modeled_vmp = pvlib.pvsystem.sapm(e_0*data['Effective irradiance [W/m²]']*T1,
                                  data['Cell Temp [C]'],
                                  coeffs)['v_mp']

    # Scale modeled Vmp by the number of modules per string
    name_modeled_vmp = inv_cb + ' Predicted Vmp from Imp'
    data[name_modeled_vmp] = v_scaling_factor*modeled_vmp

    data.loc[data[name_T1] == 0, f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'] = 0
    data.loc[data[name_T1].isna(), f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'] = np.nan
    
    data.loc[(data[f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'] > float(config['max_dcv'])), f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'] = np.nan
    data.loc[data[f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'] < float(config['min_dcv']), f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'] = 0

    data[f'INV{inv_num} CB{cb_num} Vmp Ratio from Imp'] = np.maximum(data[v_col]/data[f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'], 0)
    data.loc[data[f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'] == 0, f'INV{inv_num} CB{cb_num} Vmp Ratio from Imp'] = 0
    data.loc[data[f'INV{inv_num} CB{cb_num} Predicted Vmp from Imp'].isna(), f'INV{inv_num} CB{cb_num} Vmp Ratio from Imp'] = np.nan

#%% Look at variance in transmission
transmission_cols = [c for c in data.columns if 'Transmission' in c]
fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

for c in transmission_cols:
    ax.scatter(temp.index, temp[c], s=0.5, label=c)
ax.set_xlabel('Date', fontsize='xx-large')
ax.legend()

#%% Look at variance in modeled voltage
modeled_v_cols = [c for c in data.columns if "Predicted Vmp from Imp" in c]
fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

for c in modeled_v_cols:
    ax.scatter(temp.index, temp[c], s=0.5, label=c)

ax.set_xlabel('Date', fontsize='xx-large')
ax.set_ylabel('Voltage [V]', fontsize='xx-large')
ax.legend()

#%% Look at variance in voltage ratio
vratio_cols = [c for c in data.columns if "Vmp Ratio from Imp" in c]
fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

for c in vratio_cols:
    ax.scatter(temp.index, temp[c], s=0.5, label=c)

ax.set_xlabel('Date', fontsize='xx-large')
ax.set_ylabel('Voltage Ratio (measured/modeled)', fontsize='xx-large')
ax.legend()

# %% categorize into modes

vratio_cols = [c for c in data.columns if 'Vmp Ratio' in c]
t_cols = [c for c in data.columns if 'Transmission' in c]
v_cols = [c for c in data.columns if re.match(r'INV(\d+) CB(\d)', c) and 'Voltage' in c and 'Ratio' not in c and 'Predicted' not in c]

# Empirically determined through a statistical analysis of a larger dataset
# for the same site
v_std, t_std = 1.1102230246251565e-16, 0.007244664857579446
threshold_vratio, threshold_t = 0.9331598025404861, 0.5976185185741869
trans_vals_mean = 0.5976185185741869

def get_mode(vratio, t, v):
    if np.isnan(vratio) or np.isnan(t):
        return np.nan
    
    elif v < float(config['min_dcv']):
        return 0
    elif vratio < threshold_vratio:
        if t < threshold_t:
            return 1
        elif t > threshold_t:
            return 2
    elif vratio > threshold_vratio:
        if t < threshold_t:
            return 3
        elif t > threshold_t:
            return 4
    return np.nan

for vratio, t, v in zip(vratio_cols, t_cols, v_cols):
    col_name = re.match('INV(\d+) CB(\d+)', vratio)[0]
    new_name = col_name + ' Mode'
    data[new_name] = data[[vratio, t, v]].apply(lambda x: get_mode(x[vratio], x[t], x[v]), axis=1)


#%% Model voltage, current, and power using the SAPM model,
# without accounting for transmission
    
modeled_df = pvlib.pvsystem.sapm(data['Effective irradiance [W/m²]']*e_0,
                                 data['Cell Temp [C]'],
                                 coeffs)

#%% Calculate all losses, and ratio between transmission-weighted voltage and measured voltage

for v_col, i_col in zip(dc_voltage_cols, dc_current_cols):
    matched = re.match(r'INV(\d+) CB(\d+) Current', i_col)
    inv_num = matched.group(1)
    cb_num = matched.group(2)
    inv_cb = f'INV{inv_num} CB{cb_num}'
    i_scaling_factor = int(config['num_str_per_cb'][f'INV{inv_num} CB{cb_num}'])
    v_scaling_factor = int(config['num_mods_per_str'][inv_cb])

    modeled_power = modeled_df['p_mp']*v_scaling_factor*i_scaling_factor
    name_modeled_power = inv_cb + ' Modeled Power [W]'
    data[name_modeled_power] = modeled_power

    name_loss = inv_cb + ' Loss [W]'
    loss = np.maximum(data[name_modeled_power] - data[i_col]*data[v_col], 0)
    data[name_loss] = loss

#%% Load in snow data
    
snow_dir = pathlib.Path.joinpath(example_data_dir, 'snow.csv')
snow = pd.read_csv(snow_dir)
snow.set_index(pd.DatetimeIndex(snow['DATE']), inplace=True)

# %% Plot losses, color points by mode

loss_cols = [c for c in data.columns if "Loss" in c]
mode_cols = [c for c in data.columns if "Mode" in c and "Modeled" not in c]
modeled_power_cols = [c for c in data.columns if "Modeled Power" in c]

i = 0
l = loss_cols[i]
m = mode_cols[i]
p = modeled_power_cols[i]

cmap = {0 : 'r',
        1: 'r',
        2: 'r',
        3: 'r',
        4: 'g'}

fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data[~data[m].isna()]

ax.plot(temp.index, temp[p], c='k', alpha=0.2)
ax.scatter(temp.index, temp[p] - temp[l], c=temp[m].map(cmap), s=1)
ax.fill_between(temp.index, temp[p] - temp[l], temp[p], color='y', alpha=0.4, label='Loss [W]')

handles, labels = ax.get_legend_handles_labels()
red_patch = mpatches.Patch(color='r', label='Snow conditions present')
green_patch = mpatches.Patch(color='g', label='No snow present')
handles.append(red_patch) 
handles.append(green_patch)

ax.set_xlabel('Date', fontsize='xx-large')
ax.set_ylabel('DC Power [W]', fontsize='xx-large')

ax2 = ax.twinx()
ax2.bar(snow.index, snow['SNOW'].values, alpha=0.1, ec='k')

light_blue_patch = mpatches.Patch(color='b', alpha=0.1, ec='k', label='Snowfall [mm]')
handles.append(light_blue_patch)
ax.legend(handles=handles, fontsize='xx-large')
ax2.set_ylabel('Snowfall [mm]', fontsize='xx-large')

