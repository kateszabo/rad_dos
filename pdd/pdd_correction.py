# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

targ_depth = 2 # cm
field_size = 3 # cm
ssd = 75 # cm

# %% Import data from csv

data610 = pd.read_csv('6_10_Narayanasamy.csv', skiprows=1, header=0)
# data15 = pd.read_csv('15_Pogorski.csv', skiprows=1, header=0)
data15 = pd.read_csv('15 MV.csv', names=['X', 'Y'])

# %% define datasets
pdd_6mv = data610.filter(['X', 'Y'], axis=1).sort_values(by='X').dropna()
pdd_6mv['X'] = pdd_6mv['X'] / 10  # convert mm to cm
pdd_6mv['Y'] = pdd_6mv['Y'] / 100  # convert % to fraction

pdd_6fff = data610.filter(['X.1', 'Y.1'], axis=1).rename(columns={'X.1': 'X', 'Y.1':'Y'}).sort_values(by='X').dropna()
pdd_6fff['X'] = pdd_6fff['X'] / 10  # convert mm to cm
pdd_6fff['Y'] = pdd_6fff['Y'] / 100  # convert % to fraction

pdd_10mv = data610.filter(['X.2', 'Y.2'], axis=1).rename(columns={'X.2': 'X', 'Y.2':'Y'}).sort_values(by='X').dropna()
pdd_10mv['X'] = pdd_10mv['X'] / 10  # convert mm to cm
pdd_10mv['Y'] = pdd_10mv['Y'] / 100  # convert % to fraction

pdd_10fff = data610.filter(['X.3', 'Y.3'], axis=1).rename(columns={'X.3': 'X', 'Y.3':'Y'}).sort_values(by='X').dropna()
pdd_10fff['X'] = pdd_10fff['X'] / 10  # convert mm to cm
pdd_10fff['Y'] = pdd_10fff['Y'] / 100  # convert % to fraction

pdd_15mv = data15.filter(['X', 'Y'], axis=1).sort_values(by='X').dropna()
pdd_15mv['Y'] = pdd_15mv['Y'] / 100  # convert % to fraction

# %% Plot
plt.plot(pdd_6mv['X'], pdd_6mv['Y'] * 100, 'o-', label='6 MV')
plt.plot(pdd_6fff['X'], pdd_6fff['Y'] * 100, 'o-', label='6 FFF')
plt.plot(pdd_10mv['X'], pdd_10mv['Y'] * 100, 'o-', label='10 MV')
plt.plot(pdd_10fff['X'], pdd_10fff['Y'] * 100, 'o-', label='10 FFF')
plt.plot(pdd_15mv['X'], pdd_15mv['Y'] * 100, 'o-', label='15 MV')
plt.axvline(x=2, color='k', linestyle='--', label='detection depth')
plt.xlim(0, 4)

plt.xlabel('Depth (cm)')
plt.ylabel('PDD (%)')
plt.title('PDD curves')
plt.legend()

# %% Interpolate

percent_6mv = np.interp(targ_depth, pdd_6mv['X'], pdd_6mv['Y'])
print(f'6 MV at {targ_depth} cm is {percent_6mv * 100 :.0f} %')
percent_6fff = np.interp(targ_depth, pdd_6fff['X'], pdd_6fff['Y'])
print(f'6 FFF at {targ_depth} cm is {percent_6fff * 100:.0f} %')
percent_10mv = np.interp(targ_depth, pdd_10mv['X'], pdd_10mv['Y'])
print(f'10 MV at {targ_depth} cm is {percent_10mv * 100:.0f} %')
percent_10fff = np.interp(targ_depth, pdd_10fff['X'], pdd_10fff['Y'])
print(f'10 FFF at {targ_depth} cm is {percent_10fff * 100:.0f} %')

percent_15mv = np.interp(targ_depth, pdd_15mv['X'], pdd_15mv['Y'])
print(f'15 MV at {targ_depth} cm is {percent_15mv * 100:.0f} %')

# %% Field size correction
rdf_6mv = np.array([])

# %% SSD correction