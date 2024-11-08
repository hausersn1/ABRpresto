import ABRpresto.utils
import numpy as np

# This script shows how to convert psi to csv data.
# You can use it to convert the data in used to validate the ABRpresto algorithm to csv for other use.
# The data is filtered as it is converted using a 300-3000 Hz 1st order butterworth filter, forward and backwards
# using scipy.signal.filtfilt

# For example, if the data at https://zenodo.org/records/13987792 is extracted to C:/Data/ABRpresto data/, these
# 3 lines will convert one file to csv
Psi_data_path = 'C:/Data/ABRpresto data/Mouse0_timepoint0_left abr_io'
target_path = 'C:/Data/ABRpresto data csv'
load_options = {'reject_threshold': np.inf} # Rejection threshold in volts if None, uses the value stored in the data file. To not apply rejection, use `np.inf`.
ABRpresto.utils.Psi_to_csv(Psi_data_path, target_path, load_options=load_options)

# To recursively convert the whole dataset, uncomment below:
Psi_data_path_all = 'C:/Data/ABRpresto data/'
target_path = 'C:/Data/ABRpresto data csv'
# ABRpresto.utils.Psi_to_csv_all(Psi_data_path_all, target_path, load_options=load_options)

# To apply artifact rejection before converting, pass an artifact rejection criteria as below:
load_options = {'reject_threshold': 20e-6} # Rejection threshold in volts if None, uses the value stored in the data file. To not apply rejection, use `np.inf`.
# ABRpresto.utils.Psi_to_csv(Psi_data_path, target_path, load_options=load_options)

# You can also pass in any of these other options to change the filtering or detrend settings (the below settings are the defaults)
load_options = {'filter_lb': 300,
                'filter_ub': 3000,
                'filter_order': 1,
                'offset': -1e-3,# Starting point of epoch, in seconds re. trial start. Can be negative to capture prestimulus baseline.
                'duration': 10e-3, #Duration of epoch, in seconds, relative to offset.
                'detrend': 'constant', # One of {'constant', 'linear', None}. Method for detrending using scipy.signal.detrend
                'pad_duration': 10e-3, # Duration, in seconds, to pad epoch prior to filtering. The extra samples will be discarded after filtering.
                'reject_threshold': None, # Rejection threshold in volts if None, uses the value stored in the data file. To not apply rejection, use `np.inf`.
                }
# ABRpresto.utils.Psi_to_csv(Psi_data_path, target_path, load_options=load_options)
