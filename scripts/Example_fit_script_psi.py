import os
from pathlib import Path

from cftsdata import abr

import ABRpresto


XCsubargs = {
    'seed': 0,
    'pst_range': [0.0005, 0.006],
    'N_shuffles': 500,
    'avmode': 'median',
    'peak_lag_threshold': 0.5,
    'XC0m_threshold': 0.3,
    'XC0m_sigbounds': 'increasing, midpoint within one step of x range',  # sets bounds to make slope positive,
    # and midpoint within [min(level) - step, max(level) + step] step is usually 5 dB
    'XC0m_plbounds': 'increasing',  # sets bounds to make slope positive
    'second_filter': 'pre-average',
    'calc_XC0m_only': True,
    'save_data_resamples': False  # use this to save intermediate data (from each resample)
}

print(os.path.basename(__file__))

filename = Path('../example_data_psi/Example_1 abr_io').absolute()
print(f'Loading {filename}')

# Load ABRdata from psiexperiment data file format
fh = abr.load(filename)
epochs = fh.get_epochs_filtered()

# Loop through each of the frequencies in the file
for freq, freq_epochs in epochs.groupby('frequency'):
	print(f'Fitting {freq} Hz with ABRpresto algorithm')

	fit_results, fig_handle = ABRpresto.XCsub.estimate_threshold(freq_epochs, **XCsubargs)
	fit_results['ABRpresto version'] = ABRpresto.get_version()

	print(f"Threshold is {fit_results['threshold']:.1f}, fit with: {fit_results['status_message']}")

	# Save figure as png
	figname = filename / f'{freq}Hz_ABRpresto_fit.png'
	fig_handle.savefig(figname)
	print(f'Figure saved to {figname}')

	# Save fit results as json
	jsonname = filename / f'{freq}Hz_ABRpresto_fit.json'
	print(jsonname)
	ABRpresto.utils.write_json(fit_results, jsonname)
	print(f'Fit results saved to {jsonname}')

# In the left column the figures show mean +/- SE of all trials in black, and median (or mean, depending on AVmode) for
# the two subsets. Waveforms are normalized (for each level all 3 lines are scaled by the peak-to-peak of the mean
# of all trials). The right hand side shows mean correlation coefficient vs stimulus level. Sigmoid and power law fits
# to this data are shown in green and purple. The threshold is shown by the pink dashed line.
