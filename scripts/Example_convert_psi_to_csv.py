import ABRpresto.utils
import os

# This script loops through all 5 example datasets and fits them with ABRpresto.
# It does this for both formats (Psi and csv)

# A fit of all csv data can also be run from the command line. Do this, navigate to the ABRpresto directory, then run:
#    ABRpresto example_data -r --loader csv
# To fit all Psi data from the command lin, navigate to the ABRpresto directory, then run:
#    ABRpresto example_data_psi -r
Psi_data_path = 'C:/Data/ABRpresto data/Mouse0_timepoint0_left abr_io'
target_path = 'C:/Data/ABRpresto data csv'
ABRpresto.utils.Psi_to_csv(Psi_data_path, target_path)