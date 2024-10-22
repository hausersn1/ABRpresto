import ABRpresto
import os
import pandas as pd

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

filename = os.path.realpath('../example_data/Example_1.csv')
print(f'Loading {filename}')
abr_single_trial_data = pd.read_csv(filename, index_col=[0, 1, 2])
abr_single_trial_data.columns.name = 'time'
abr_single_trial_data.columns = abr_single_trial_data.columns.astype('float')

print('Fitting with XCsub algorithm')
fit_results, fig_handle = ABRpresto.XCsub.estimate_threshold(abr_single_trial_data, **XCsubargs)
fit_results['ABRpresto version'] = ABRpresto.get_version()

print(f"Threshold is {fit_results['threshold']:.1f}, fit with: {fit_results['status_message']}")

# Save figure as png
figname = filename.replace('.csv', '_XCsub_fit.png')
fig_handle.savefig(figname)
print(f'Figure saved to {figname}')

# Save fit results as json
jsonname = filename.replace('.csv', '_XCsub_fit.json')
ABRpresto.utils.write_json(fit_results, jsonname)
print(f'Fit results saved to {jsonname}')

# In the left column the figures show mean +/- SE of all trials in black, and median (or mean, depending on AVmode) for
# the two subsets. Waveforms are normalized (for each level all 3 lines are scaled by the peak-to-peak of the mean
# of all trials). The right hand side shows mean correlation coefficient vs stimulus level. Sigmoid and power law fits
# to this data are shown in green and purple. The threshold is shown by the pink dashed line.