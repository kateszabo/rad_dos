import numpy as np
import pandas as pd

# %% Define inputs
beam    = '6X'  # beam energy
depth   = 2     # depth of detection, cm
size    = 3     # equivalent square field size, cm
ssd     = 75    # cm

# %% Depth dose correction
pdd_data = pd.read_csv('dose_correction/pdd_data.csv', dtype=float)
pdd = np.interp(depth, pdd_data['Depth'], pdd_data[beam])
print(f'PDD for {beam} at {depth} cm is {pdd:.3f}')

# %% Field size correction
rdf_data = pd.read_csv('dose_correction/RDF_Linac.csv', sep='\t', dtype=float)
rdf = np.interp(size, rdf_data['Eq sq'], rdf_data[beam])
print(f'RDF for {beam} at {size} cm is {rdf:.3f}')

# %% SSD correction
ssd_x = np.linspace(0.1, 200, 1000)
ssd_data = (100 ** 2) / (ssd_x ** 2)  # 1/r^2, normalized to 1 at 100 cm
ssd_correction = np.interp(ssd, ssd_x, ssd_data)
print(f'SSD correction at ssd = {ssd} cm is {ssd_correction:.3f}')
# %% Total
total_correction = pdd * rdf * ssd_correction
print(f'Total correction is {total_correction:.3f}')
