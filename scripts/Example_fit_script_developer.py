import ABRpresto
import os
import pandas as pd

# XCsubargs = {
#     'seed': 0,
#     'pst_range': [0.0005, 0.006],
#     'N_shuffles': 500,
#     'avmode': 'median',
#     'peak_lag_threshold': 0.5,
#     'XC0m_threshold': 0.3,
#     'XC0m_sigbounds': 'increasing, midpoint within one step of x range',  # sets bounds to make slope positive,
#     # and midpoint within [min(level) - step, max(level) + step] step is usually 5 dB
#     'XC0m_plbounds': 'increasing',  # sets bounds to make slope positive
#     'second_filter': 'pre-average',
#     'calc_XC0m_only': True,
# }

#These options also fit the correlation distributions in other ways, which empirically didn't work as well.
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
    'KSs_sigbounds': 'increasing, midpoint within one step of x range',
    'KSs_plbounds': 'increasing',
    'XCp_near_0_sigbounds': 'increasing, midpoint within one step of x range',
    'XCp_near_0_plbounds': 'increasing',
    'second_filter': 'pre-average',
    'calc_XC0m_only': False,
    'save_data_resamples': True  # use this to save intermediate data (from each resample)
}

print(os.path.basename(__file__))

filename = os.path.realpath('../example_data/Example_1.csv')
print(f'Loading {filename}')
abr_single_trial_data = pd.read_csv(filename, index_col=[0, 1, 2])
abr_single_trial_data.columns.name = 'time'
abr_single_trial_data.columns = abr_single_trial_data.columns.astype('float')

print('Fitting with XCsub algorithm')
fit_results, fig_handle = ABRpresto.XCsub.estimate_threshold(abr_single_trial_data, **XCsubargs)

print(f"Threshold is {fit_results['threshold']:.1f}, fit with: {fit_results['status_message']}")

# Save figure as png
figname = filename.replace('.csv', '_XCsub_fit_full.png')
fig_handle.savefig(figname)
print(f'Figure saved to {figname}')

# Save fit results as json
jsonname = filename.replace('.csv', '_XCsub_fit_full.json')
ABRpresto.utils.write_json(fit_results, jsonname)
print(f'Fit results saved to {jsonname}')
