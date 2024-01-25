#%%

import os
import sys
import pathlib

base_dir = os.path.join(os.path.expanduser("~"), 'pvanalytics')
sys.path.append(base_dir)

import pandas as pd
import numpy as np
import pvanalytics
from pvanalytics.features import snow
import pvlib

def test_get_transmission():
    database = pvlib.pvsystem.retrieve_sam('SandiaMod')
    panel = 'LG_LG290N1C_G3__2013_'
    coeffs = database[panel]

    pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
    ghi_file = pvanalytics_dir / 'data' / 'rmis_weather_data.csv'
    data = pd.read_csv(ghi_file, index_col=0, parse_dates=True)

    data['Module temp [C]'] = data['Plane of array']*np.exp(-3.47 -0.0594*data['Wind Speed'])
    data['Cell temp [C]'] = data['Module temp [C]'] + 3*(data['Plane of array']/1000)
    T = np.random.random_sample(len(data))

    out_df = pvlib.pvsystem.sapm(data['Plane of array']*T,
                        data['Cell temp [C]'],
                        coeffs)


    T1, T2 = snow.get_transmission(data['Cell temp [C]'].values,
                        out_df['i_mp'].values,
                        data['Plane of array'].values,
                        1,
                        coeffs['Impo'],
                        coeffs['C0'],
                        coeffs['C1'],
                        coeffs['Aimp'])

    normalized_rmse = np.sqrt(np.nansum(np.square(T1 - T)))/np.nansum(~np.isnan(T1))

    assert(normalized_rmse < 0.02)
