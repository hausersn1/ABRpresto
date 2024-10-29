import ABRpresto.main
import os

# This script loops through all 5 example datasets and fits them with ABRpresto.
# It does this for both formats (Psi and csv)

# A fit of all csv data can also be run from the command line. Do this, navigate to the ABRpresto directory, then run:
#    ABRpresto example_data -r --loader csv
# To fit all Psi data from the command lin, navigate to the ABRpresto directory, then run:
#    ABRpresto example_data_psi -r

example_freqs = [8000, 32000, 4000, 22627, 16000]  # These are the frequencies used for each example (1-5).
for i, freq in enumerate(example_freqs):
    #Fit examples in psi data format
    pth = os.path.realpath(f'../example_data_psi/Example_{i+1} abr_io')
    ABRpresto.main.run_fit(pth, 'psi', frequencies=[freq])

    #Fit examples in csv format
    pth = os.path.realpath(f'../example_data/Example_{i}.csv')
    ABRpresto.main.run_fit(pth, 'csv')
