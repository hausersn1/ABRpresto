# ABRpresto
ABR thresholds: fast, accurate, no human needed.

An algorithm for generating ABR thresholds by cross-correlation of resampled subaverages

This code algorithmically thresholds ABR data as described in Shaheen et al. 2024. Instead of averaged waveforms, the algorithm uses the single-trial data (i.e., the response to each presentation of the stimuli). Single-trial data can be collected using [psiexperiment's CFTS module](https://github.com/psiexperiment/cfts).

A full dataset of single-trial ABR waveforms used to test this algorithm is available on [Zenodo](https://zenodo.org/records/13987792).

Thresholds are generated by:
1. Randomly splitting the trials into two groups, each containing an equal number of responses to positive and negative polarity stimuli.
2. Calculating the median waveform for each group.
3. Calculating the normalized cross correlation between these median waveforms. 
4. This process (steps 1-3) is repeated 500 times to obtain a reshuffled cross-correlation distribution. 
5. The mean values of these distributions are computed for each level and fit with a sigmoid and a power law. The fit that provides the best mean squared error is then used, and threshold is defined as where that fit crossed a criterion (default 0.3).

## Installing

This code must be installed using `pip install`. To install a version for
in-place modification (e.g., development), use `pip install -e`.

Full installation steps: (if you are using conda environments, you can
create a new environment to install this code, otherwise you can just install
directly into your main Python environment):

	(if using conda environments) conda create -n ABRpresto python git
	(if using conda environments) conda activate ABRpresto
	(navigate to the folder containing ABRpresto)
     python -m pip install -e ./ABRpresto


## Usage

Example data is provided in both `csv` format ([example_data](example_data)) and `psi` format ([example_data_psi](example_data_psi)) and two example scripts [Example_fit_script.py](scripts%2FExample_fit_script.py) and [Example_fit_script_psi.py](scripts%2FExample_fit_script_psi.py) demonstrate how each of these datasets can be fit using ABRpresto.

The algorithm requires data to be passed as a pandas dataframe with a multiindex containing `polarity` and `level`. Extra levels in the index will be ignored (e.g., `t0` in the example datasets). Each row contains a single trial. Columns are the time relative to stimulus onsset in seconds. A dictionary of results and a figure are returned, which can be saved as json and png.

### Figure output

In the left column the figures show mean +/- SE of all trials in black, and median (or mean, depending on AVmode) for the two subsets. Waveforms are normalized (for each level all 3 lines are scaled by the peak-to-peak of the mean of all trials). The right hand side shows mean correlation coefficient vs stimulus level. Sigmoid and power law fits to this data are shown in green and purple. The threshold is shown by the pink dashed line.

### Command line usage

The algorithm can be run on the command line to process one or more datasets.

To process a single psi dataset:

	ABRpresto <path to dataset>

To process several psi datasets:

	ABRpresto <path to dataset 1> <path to dataset2> <etc>

To recursively scan and process all datasets in a folder:

	ABRpresto <path> -r

For example, to fit all example psi data, navigate to the ABRpresto directory, then run:
    
    ABRpresto example_data_psi -r

To fit all example csv data, you just need to add the loader option to use the csv data loader.
Navigate to the ABRpresto directory, then run: 
 
    ABRpresto example_data -r --loader csv


## Citation

If you use this algorithm in you research, please cite:
XXX

The curve fitting decision tree used in this algorithm was inspired by Suthakar et al. If you use this algorithm please 
 also cite their paper:
Suthakar, K. & Liberman, M. C. A simple algorithm for objective threshold determination of auditory brainstem responses.
  Hear. Res. 381, 107782 (2019).
 

## License

Please see [LICENSE](LICENSE)

