import ABRpresto.utils

# This script shows how to convert psi to csv data.
# You can use it to convert the data in used to validate the ABRpresto algorithm to csv for other use.
# For example, if the data at https://zenodo.org/records/13987792 is extracted to C:/Data/ABRpresto data/, these
# 3 lines will convert one file to csv
Psi_data_path = 'C:/Data/ABRpresto data/Mouse0_timepoint0_left abr_io'
target_path = 'C:/Data/ABRpresto data csv'
ABRpresto.utils.Psi_to_csv(Psi_data_path, target_path)

# To recursively convert the whole dataset, uncomment below:
Psi_data_path_all = 'C:/Data/ABRpresto data/'
target_path = 'C:/Data/ABRpresto data csv'
# ABRpresto.utils.Psi_to_csv_all(Psi_data_path_all, target_path)
