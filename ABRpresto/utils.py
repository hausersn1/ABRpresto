import numpy as np
from scipy import signal
import json
import scipy.optimize as optimize
from pathlib import Path
import pandas as pd


def crossCorr(x1, x2, norm=False):
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    if norm:
        x1 = x1 / (np.std(x1) * len(x1))
        x2 = x2 / np.std(x2)
    return np.correlate(x1, x2, 'same')


def fit_sigmoid_power_law(levels, y, thresholdCriterion, y_err=None, sigbounds=None, plbounds=None):
    """
        Fits sigmoid and power low functions to y, determines which to use based on mse, then computes threshold as
         where that fit crosses a criterion (default 0.3).

Parameters
    ----------
    levels: stimulus levels
    y: data to fit
    thresholdCriterion: The criterion value used to find threshold
    y_err: error on y. Sigmoid fit is weithed based on the sqrt(y_err).
       square root is used to de-emphasize errors. Otherwise the sloping part of the sigmoid isn't well-fit
    sigbounds:
        If 'increasing, midpoint within one step of x range' sets bounds to make slope positive,
                # and midpoint within [min(level) - step, max(level) + step]. step is usually 5 dB
    plbounds: string, default None
            If 'increasing' sets bounds to make slope positive

Outputs
    ----------
    fit_results: dictionary of fitting results, see XCsub.estimate_threshold for details
    threshold: the threshold estimated by the algorithm.

    """

    adjR2Criterion = 0.7

    thdEstimationFailed = True
    threshold = np.nan
    sigmoid_fit = None
    power_law_fit = None
    bestFitType = None

    # 1 check if cc passes criterion
    if min(y) > thresholdCriterion * 1.1:  # Used a factor of 1.1 to allow a little extrapolation
        threshold = -np.inf
        bestFitType = 'all above criterion, threshold is -inf'
        thdEstimationFailed = False
    elif max(y) < thresholdCriterion:
        threshold = np.inf
        bestFitType = 'all below criterion, threshold is inf'
        thdEstimationFailed = False
    else:
        power_law_fit = fit_power_law(levels, y, y_err, bounds=plbounds)
        sigmoid_fit = fit_sigmoid(levels, y, y_err, bounds=sigbounds)

        if (sigmoid_fit['yfit'][0] < thresholdCriterion) & (sigmoid_fit['yfit'][-1] > thresholdCriterion) & \
                (sigmoid_fit['sse'] < power_law_fit['sse']):
                threshold = sigmoid_get_threshold(thresholdCriterion, *sigmoid_fit['params'])
                bestFitType = 'sigmoid'
                thdEstimationFailed = False
        elif not((power_law_fit['yfit'][0] < thresholdCriterion) & (power_law_fit['yfit'][-1] > thresholdCriterion)):
            if (y > thresholdCriterion).sum() <= 2:
                # Cases where noise pushed some levels above criterion, but couldn't fit. Can safely call inf
                threshold = np.inf
                bestFitType = "most below criterion, but couldn't fit, threshold is inf"
                thdEstimationFailed = False
            elif (y < thresholdCriterion).sum() <= 2:
                # Cases where noise pushed some levels below criterion, but couldn't fit. Can safely call -inf
                threshold = -np.inf
                bestFitType = 'all above criterion, threshold is -inf'
                thdEstimationFailed = False
            # Otherwise default remain: threshold=nan and thdEstimationFailed=False.
            # We want to take a closer look at these cases
        else:
            threshold = power_law_get_threshold(thresholdCriterion, *power_law_fit['params'])
            if power_law_fit['adj_r2'] > adjR2Criterion:
                bestFitType = 'power law'
            else:
                bestFitType = 'power law (noisy)'
            thdEstimationFailed = False

    algpars = {'thresholdCriterion': thresholdCriterion, 'adjR2Criterion': adjR2Criterion}
    fit_results = {'threshold': threshold, 'sigmoid_fit': sigmoid_fit, 'power_law_fit': power_law_fit,
                   'bestFitType': bestFitType, 'thdEstimationFailed': thdEstimationFailed, 'algpars': algpars}
    return fit_results, threshold


def fit_sigmoid(x, y, y_err=None, bounds=None):
    if bounds is None:
        bounds_ = (-np.inf, np.inf)
    elif bounds == 'increasing, midpoint within one step of x range':
        bounds_ = ((.1, 0, x.min() - np.diff(x[:2])[0], -0.5), (1, np.inf, x.max() + np.diff(x[:2])[0], 0.8))
    else:
        raise RuntimeError(f'bounds was set to "{bounds}" but this is not an option.')

    Pinit = sigmoid_find_initial_params(x, y, bounds=bounds, bounds_=bounds_)

    try:
        span = x.max() - x.min()
        # Grid 5 steps over range
        # P0_grid = (slice(Pinit[0], Pinit[0]+1, 1),slice(x.min(), x.max()+.00001, (span)/4))
        # Grid 3 steps from 1/6 to 5/6 range.
        P0_grid = (slice(Pinit[0], Pinit[0] + 1, 1),
                   slice(Pinit[1], Pinit[1] + 1, 1),
                   slice(x.min() + span / 8, x.max() - span / 8 + .0001, (span * 3 / 4) / 4),
                   slice(Pinit[3], Pinit[3] + 1, 1))
        # Quick grid search on x0 to get close to global minima
        Pinit = optimize.brute(sigmoid_obj_fn, P0_grid, args=(x, y), finish=None)
        # Unused code, was used when finish was fmin to ensure Pinit was in bounds
        # if (resbrute[0] > bounds_[0][0]) & (resbrute[0] < bounds_[1][0]) & \
        #         (resbrute[1] > bounds_[0][1]) & (resbrute[1] < bounds_[1][1]):
        #     Pinit = resbrute
        #     print(f'{Pinit}')
        # else:
        #     print(f'Grid search on {P0_grid} returned {resbrute}, which is outside the bounds {bounds_}. '
        #           f'Seeding curve_fit with default, {Pinit})')
        # least-squares on best grid fit
        if y_err is not None:
            # Take sqrt to de-emphasize errors. Otherwise the sloping part of the sigmoid isn't well-fit
            y_err = np.sqrt(y_err)
        P, pcov = optimize.curve_fit(
            sigmoid, x, y, p0=Pinit, method="trf", max_nfev=50000, sigma=y_err, bounds=bounds_
        )
        # print(f'[{Pinit[0]:.4f}, {Pinit[1]:.1f}] -> [{P[0]:.4f}, {P[1]:.1f}]')
    except RuntimeError:
        print(f'Fit failed, using default {Pinit}')
        P = Pinit

    yfit = sigmoid(x, *P)
    sse = np.sum((yfit - y)**2)
    return {'params': P, 'yfit': yfit, 'sse': sse}


def sigmoid_find_initial_params(x, y, bounds=None, bounds_=None):
    amplitude = np.max(y)
    x0 = np.mean(x)
    baseline = np.min(y)
    slope = 8 / amplitude * (amplitude - baseline) / (np.max(x) - np.min(x))
    if bounds == 'increasing, midpoint within one step of x range':
        if slope <= 0:
            raise RuntimeError("Negative initial slope. This shouldn't happen?")
        if baseline < bounds_[0][3]:
            baseline = bounds_[0][3]
        if baseline > bounds_[1][3]:
            baseline = bounds_[1][3]
    return np.array([amplitude, slope, x0, baseline])


def sigmoid(x, amplitude, slope, x0, baseline):
    y = amplitude / (1 + np.exp(-slope * (x - x0))) + baseline
    return y


def sigmoid_obj_fn(params, *data):
    yfit = sigmoid(data[0], *params)
    return np.sum((yfit - data[1]) ** 2)


def sigmoid_get_threshold(criterion, amplitude, slope, x0, baseline):
    if amplitude / (criterion - baseline) <= 1:
        return np.nan
    return x0 - np.log(amplitude / (criterion - baseline) - 1)/slope


def fit_power_law(x, y, y_err=None, bounds=None):
    x = np.array(x)
    if np.min(x) == 0:
        offset = 1
    elif np.min(x) < 0:
        offset = -np.min(x)
    else:
        offset = 0

    xp = x + offset
    yp = y

    if bounds is None:
        bounds_ = (-np.inf, np.inf)
    elif bounds == 'increasing':
        bounds_ = ((-np.inf, 0, -np.inf), (np.inf, np.inf, np.inf))
    else:
        raise RuntimeError(f'bounds was set to "{bounds}" but this is not an option.')

    Pinit = power_law_find_initial_params(xp, yp, bounds=bounds)

    try:
        P, pcov = optimize.curve_fit(
            power_law, xp, yp, p0=Pinit, method="trf", max_nfev=50000, bounds=bounds_)

    except RuntimeError as RTE:
        print(f'Fit failed for power law with sigma, using default {Pinit}')
        print(RTE)
        P = Pinit
    except Exception as E:
        pass

    yfit = power_law(x, *P)
    sse = np.sum((yfit - y) ** 2)
    sstot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sstot

    n = len(y)
    free_params = len(P)
    adj_r2 = 1 - (((1 - r2) * (n - 1)) / (n - free_params - 1))

    return {'params': P, 'yfit': yfit, 'sse': sse, 'adj_r2': adj_r2}


def power_law_find_initial_params(x, y, bounds=None):
    idx = (x > 0) & (y > 0)
    if idx.sum() <= 2:
        power = 0.5
        if idx.sum() > 0:
            amplitude = (y[idx] / (x[idx] ** power)).mean()
        else:
            amplitude = 1
    else:
        x = x[idx]
        y = y[idx]
        if bounds == 'increasing':
            powers = np.log(y[0] / y[1:]) / np.log(x[0] / x[1:])
            if np.all(powers <= 0):
                power = 0.5
            else:
                power = np.mean(powers[powers > 0])
        else:
            power = np.mean(np.log(y[0] / y[1:]) / np.log(x[0] / x[1:]))
        amplitude = (y/(x**power)).mean()
    baseline = 0
    return np.array([amplitude, power, baseline])


def power_law(x, amp, power, baseline):
    return amp * x**power + baseline


def power_law_get_threshold(y,  amp, power, baseline):
    return ((y - baseline) / amp) ** (1 / power)


def filter(sig, fs, highpass, lowpass, order=1):
    if highpass >= lowpass:
        raise ValueError(f'Highpass must be < lowpass')
    Wn = highpass / (0.5 * fs), lowpass / (0.5 * fs)
    b, a = signal.iirfilter(order, Wn)
    return signal.filtfilt(b, a, sig, axis=-1)


def write_json(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f, cls=AutoThJsonEncoder, indent=4)


class AutoThJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif callable(obj):
            return f'{obj.__module__}.{obj.__name__}'
        else:
            return super().default(obj)

def load_fits(pth, save=True, algorithm = 'ABRpresto'):
    mouse_num = []
    timepoint = []
    ear = []
    frequency = []
    threshold = []
    status = []
    status_msg = []
    for filename in Path(pth).glob(f'*_{algorithm}_fit*.json'):
        if 'Copy' in str(filename):
            continue
        D = json.load(open(filename))
        threshold.append(D['threshold'])
        mouse_num.append(int(str(filename).split('\\')[-1].split('_')[0][5:]))
        timepoint.append(int(str(filename).split('\\')[-1].split('_')[1][9:]))
        ear.append(str(filename).split('\\')[-1].split('_')[2].split(' ')[0])
        frequency.append(int(str(filename).split(' ')[-1].split('.')[0]))
        status.append(D['status'])
        status_msg.append(D['status_message'])

    df = pd.DataFrame(
        {'threshold': threshold, 'id': mouse_num, 'timepoint': timepoint, 'ear': ear, 'frequency': frequency,
         'status': status, 'status_message': status_msg})
    if save:
        save_pth = pth + f'{algorithm} thresholds.csv'
        df.to_csv(save_pth, index=False)
        print(f'Saved {len(df)} thresholds to {save_pth}')
    return df

def compare_thresholds(df, thresholders, impute_infs=True):
    def set_ephys_id(DF):
        DF['ephys_id_no_freq'] = DF['study'] + '_ID' + DF['id'].astype('str') + \
                                 '_' + 'W' + \
                                 DF.timepoint.astype('str') + '_' + DF.ear
        DF['ephys_id'] = DF['ephys_id_no_freq'] + \
                         '_' + DF['frequency'].astype('float').round(-2).astype('int').astype('str')
        DF['ephys_dir'] = DF['study'] + '_' + DF.timepoint_unit.str.slice(0, 1).str.upper() + DF.timepoint.astype(
            'str') + \
                          '_ID' + DF['id'].astype('str')
        DF['ephys_dir2'] = DF['ephys_dir'] + '_' + DF.ear + '_' + DF.assay + ' abr_io ' + DF.group
        # DF['ephys_id'] = DF['ephys_id_dir']                     '_' + DF['frequency'].astype('float').round(-2).astype('int').astype('str')

        return DF

    # for i in range(len(thresholders)):
    #     df[i] = set_ephys_id(df[i].reset_index())

    if impute_infs:
        for v in ['threshold', 'XCsubpublic threshold', 'manual threshold']:
            ii = np.isinf(df[v]) & (df[v] < 0);
            df.loc[ii, v] = df.loc[ii, 'min_level'] - 5
            ii = np.isinf(df[v]) & (df[v] > 0);
            df.loc[ii, v] = df.loc[ii, 'max_level'] + 5

    if 'status_message' in df[1].keys():
        C = df[1].set_index(DF_indicies)['status_message']
        C.name = ('', 'status_message')
        dfj = pd.concat((dfj, C.to_frame()), axis=1)

    # dfj.loc[dfj[('', 'status_message')].isnull(), ('', 'status_message')] = ''
    # dfj = dfj[dfj[('', 'status_message')].str.contains('power law')]

    analyzers = thresholders[1:]
    dfs_all = summarize_thd_diffs(dfj, analyzers, crits=(0, 5, 10))
    dfj['threshold', 'diff'] = dfj['threshold', analyzers[0]] - dfj['threshold', 'THD']

    binsize = 5

    def pli(x_, y_, binsize=binsize):
        print(f'Got {x_},{y_}')
        if pli.x == x_ and pli.y == y_:
            pli.ind += 1
            print(f'Loaded, incrementing ind to {pli.ind}')
        else:
            pli.ind = 0
            pli.x = x_
            pli.y = y_
            if (x_ == 5) and (y_ == 0):
                pli.sel = dfj[('threshold', 'THD')].isnull() & dfj[('threshold', thresholders[1])].isnull()
            elif (x_ == 5) and (y_ == 5):
                pli.sel = dfj[('threshold', 'THD')].isnull() & ~dfj[('threshold', thresholders[1])].isnull()
            elif (x_ == 5) and (y_ == -5):
                pli.sel = ~dfj[('threshold', 'THD')].isnull() & dfj[('threshold', thresholders[1])].isnull()
            else:
                pli.sel = (dfj['threshold', 'diff'] >= (pli.y - binsize / 2)) & (
                        dfj['threshold', 'diff'] < (pli.y + binsize / 2)) & \
                          (dfj['threshold', 'THD'] >= (pli.x - binsize / 2)) & (
                                  dfj['threshold', 'THD'] < (pli.x + binsize / 2))
            pli.ephys_ids = dfj[pli.sel].index.get_level_values(level='ephys_id').values
            print(f'There are {pli.sel.sum()} points in this cell: {pli.ephys_ids}')
        print(f'**** {pli.ephys_ids[pli.ind]}')
        print(dfj[pli.sel].iloc[pli.ind])

        freq_str = f"{dfj[pli.sel].index.get_level_values(level='frequency')[pli.ind] / 1000:1.3f}"
        # freq_str = f"{dfj[pli.sel].iloc[pli.ind]['frequency'] / 1000:1.3f}"
        if freq_str[-2:] == '00':
            freq_str = freq_str[:-2]
        if freq_str == '5.656':
            freq_str = '5.657'
        if freq_str == '11.313':
            freq_str = '11.314'
        # fn = f"{dfj[pli.sel].iloc[pli.ind]['ephys_dir2']} {freq_str}kHz {thresholders[0]} threshold.svg"
        study = dfj[pli.sel].reset_index().iloc[pli.ind]['study'].values[0]
        path = study.split('-')[0] + '/' + study

        if thresholders[0] == 'manual':
            fn = f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0].split(' ')[0]} abr_io-group-combined-{freq_str}kHz-"
            pth = f"X:/{path}/{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir'].values[0]}/{fn}"
            fns = os.listdir(f"X:/{path}/{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir'].values[0]}/")
            fns = [fn_ for fn_ in fns if (fn_.count(fn) > 0) and (fn_.count('.png') > 0)]
            if len(fns) > 1:
                print(f"Multilple files {fns}")
                fns = fns[:1]
            pth = f"X:/{path}/{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir'].values[0]}/{fns[0]}"
            img = Image.open(pth)
            print(pth)
            pli.axes[0].cla()
            pli.axes[0].imshow(img)
            pli.axes[0].axis('off')

        else:
            fn = f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0]} {freq_str}kHz AUTO-{thresholders[0]} threshold.svg"
            pth = f"X:/{path}/{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir'].values[0]}/" \
                  f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0]}/{fn}"
            try:
                svg_file = open(pth, mode="rb")
                print(pth)
            except FileNotFoundError:
                if freq_str == '5.657':
                    freq_str = '5.656'
                if freq_str == '11.314':
                    freq_str = '11.313'
                fn = f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0]} {freq_str}kHz AUTO-{thresholders[0]} threshold.svg"
                pth = f"X:/{path}/{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir'].values[0]}/" \
                      f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0]}/{fn}"
                svg_file = open(pth, mode="rb")
                print('Changed freq from 5.657 to 5.656')

            img_png = cairosvg.svg2png(bytestring=svg_file.read())
            img = Image.open(BytesIO(img_png))
            pli.axes[0].cla()
            pli.axes[0].imshow(img)
            pli.axes[0].axis('off')

        if thresholders[1] == 'manual':
            print('fix me')
        else:
            fn2 = f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0]} {freq_str}kHz AUTO-{thresholders[1]} threshold.svg"
            pth2 = f"X:/{path}/{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir'].values[0]}/" \
                   f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0]}/{fn2}"
            try:
                svg_file = open(pth2, mode="rb")
                print(pth2)
            except FileNotFoundError:
                if freq_str == '5.657':
                    freq_str = '5.656'
                if freq_str == '11.314':
                    freq_str = '11.313'
                fn = f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0]} {freq_str}kHz AUTO-{thresholders[1]} threshold.svg"
                pth2 = f"X:/{path}/{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir'].values[0]}/" \
                       f"{dfj[pli.sel].reset_index().iloc[pli.ind]['ephys_dir2'].values[0]}/{fn}"
                svg_file = open(pth2, mode="rb")
                print('Changed freq from 5.657 to 5.656')

            img_png = cairosvg.svg2png(bytestring=svg_file.read())
            img = Image.open(BytesIO(img_png))
            pli.axes[1].cla()
            pli.axes[1].imshow(img)
            pli.axes[1].axis('off')

        plt.pause(.1)
        plt.draw()

    pli.ind = 0
    pli.sel = None
    pli.x = None
    pli.y = None
    pli.ephys_ids = None
    if len(thresholders) <= 2:
        pli.fig, pli.axes = plt.subplots(1, figsize=(16, 10),
                                         gridspec_kw={'top': 1, 'bottom': 0, 'left': 0, 'right': 1})
        _, axes2 = plt.subplots(1, figsize=(16, 10), gridspec_kw={'top': 1, 'bottom': 0, 'left': 0, 'right': 1})
        pli.axes = np.array((pli.axes, axes2))
    else:
        pli = None
    ax = plot_threshold_diffs(dfj, dfs_all, analyzers, as_diff=True, norm='by_column', fn=pli)
    ax[-1, 0].set_xlabel(f'Median Threshold (dB SPL; {thresholders[0]})', fontsize=20)

    # dfj.loc[dfj[('', 'status_message')].isnull(),('', 'status_message')] = ''
    # ax = plot_threshold_diffs(dfj[dfj[('','status_message')].str.contains('power law')], dfs_all, analyzers, as_diff=True, norm='by_column', fn=pli)

    subs = ut.get_subs(len(analyzers))
    gs_kw = dict(hspace=0.1, wspace=0, left=0.08, right=.99, bottom=.1, top=.93)
    fig, ax = plt.subplots(subs[1], subs[0], figsize=(5 * subs[0], 12), sharey='all', sharex='all',
                           gridspec_kw=gs_kw)
    if len(analyzers) == 1: ax = np.array(ax, ndmin=2)
    C = dfj.corr(method='spearman')
    for i in range(len(analyzers)):
        ax.flat[i].plot([10, 115], [10, 115], '--k')
        ax.flat[i].scatter(dfj[('threshold', 'THD')], dfj[('threshold', analyzers[i])], 3, 'k')
        ax.flat[i].text(0, 0.99, f"spearman p = {C.loc[('threshold', 'THD'), ('threshold', analyzers[i])]:.2f}",
                        transform=ax.flat[i].transAxes, va='top')
        ax.flat[i].set_title(analyzers[i])
        dfs_all.loc[analyzers[i], 'spearman'] = C.loc[('threshold', 'THD'), ('threshold', analyzers[i])]

    [ax_.set_xlim([10, 115]) for ax_ in ax.flat]
    [ax_.set_ylim([10, 115]) for ax_ in ax.flat]
    [ax_.set_aspect('equal', 'box') for ax_ in ax.flat]
    return df, dfs_all, dfj, thresholders, pli

