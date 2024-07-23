from ABRpresto import XCsub
from ABRpresto import utils
import ABRpresto
import os
import pandas as pd


def run_fit(file_path, XCsubargs=None):
    if XCsubargs is None:
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

    print(f'Loading {file_path}')
    abr_single_trial_data = pd.read_csv(file_path, index_col=[0, 1, 2])
    abr_single_trial_data.columns.name = 'time'
    abr_single_trial_data.columns = abr_single_trial_data.columns.astype('float')

    print('Fitting with XCsub algorithm')
    fit_results, figure = XCsub.estimate_threshold(abr_single_trial_data, **XCsubargs)
    fit_results['ABRpresto version'] = ABRpresto.get_version()

    print(f"Threshold is {fit_results['threshold']:.1f}, fit with: {fit_results['status_message']}")

    # Save figure as png
    figname = file_path.replace('.csv', '_XCsub_fit.png')
    figure.savefig(figname)
    print(f'Figure saved to {figname}')

    # Save fit results as json
    jsonname = file_path.replace('.csv', '_XCsub_fit.json')
    utils.write_json(fit_results, jsonname)
    print(f'Fit results saved to {jsonname}')



def main_process():
    import argparse
    parser = argparse.ArgumentParser('auto-th')
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-r', '--recursive', action='store_true')
    args = parser.parse_args()

    if args.recursive:
        for pth in args.paths:
            filenames = [filename for filename in os.listdir(pth) if filename.endswith('.csv')]
            print(f'Found {len(filenames)} .csv files in {pth}, running XCsub on each:')
            for filename in filenames:
                run_fit(os.path.join(pth, filename))
    else:
        for pth in args.paths:
            if os.path.isdir(pth):
                raise RuntimeError(f'{pth} is a directory, pass only full paths to csv files.')
            run_fit(pth)

if __name__ == '__main__':
    main_process()
