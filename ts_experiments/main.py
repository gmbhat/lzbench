#!/usr/bin/env python

# here's what we want:
#   option to do everything 8b or 16b
#   option to do rowmajor or colmajor (although only colmajor for now)
#   option to pick dataset[s] (defaults to all of them)
#   option to pick algorithm[s] (defaults to all of them)
#       -and note that all these names are some canonical form
#   option to pick memory limit in KB (defaults to no limit)
#   option to pick how many seconds of compression/decompression (-t arg)
#   for selected combo of (nbits, order, algos, dsets)
#       figure out which dset to actually use (eg, faspfor needs u32 versions)
#           -and other stuff needs deltas
#       figure out which algo to actually use
#           -eg, 'bitshuf' needs to get turned into 'blosc_bitshuf8b' or 16b
#       figure out cmd line name for selected algo
#       figure out cmd line params for selected algo based on orig name
#       figure out path for selected dset
#       figure out path for file in which to dump the df
#           -maybe just one giant df we can query later (so timestamp the versions)
#       figure out path for file in which to dump the fig(s)
#
#   code to generate scatterplots for speed vs ratio
#   code to read in our stored data and generate real plots via this func

import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import sys
from . import files
from . import pyience

# import gflags   # google's command line lib; pip install python-gflags
# FLAGS = gflags.FLAGS

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

# gflags.DEFINE_


DATASETS_DIR = '~/Desktop/datasets/compress'  # change this if you aren't me
FIG_SAVE_DIR = 'figs'
RESULTS_SAVE_DIR = 'results'
ALL_RESULTS_PATH = os.path.join(RESULTS_SAVE_DIR, 'all_results.csv')

DEFAULT_LEVELS = [1, 5, 9]  # many compressors have levels 1-9

NEEDS_NBITS = '<nbits>'

# np.random.seed(123)


class AlgoInfo(object):

    def __init__(self, lzbench_name, levels=None, preprocs=['delta'],
                 needs_32b=False, group=None):
        self.lzbench_name = lzbench_name
        self.levels = levels
        self.preprocs = preprocs  # try with each of these variants of data
        self.needs_32b = needs_32b
        # self.group = group if group is not None else np.random.randint(99999)
        self.group = group


ALGO_INFO = {
    'Memcpy':       AlgoInfo('memcpy'),
    # general-purpose compressors
    'Zlib':         AlgoInfo('zlib', levels=DEFAULT_LEVELS),
    'Zstd':         AlgoInfo('zstd', levels=DEFAULT_LEVELS),
    'LZ4':          AlgoInfo('lz4', levels=DEFAULT_LEVELS),
    'Gipfeli':      AlgoInfo('gipfeli'),
    'Snappy':       AlgoInfo('snappy'),
    # just entropy coding
    'FSE':          AlgoInfo('fse'),
    'Huffman':      AlgoInfo('huff0'),
    # integer compressors
    'Delta':        AlgoInfo('sprintzDelta', preprocs=None, group='Sprintz'),
    'DoubleDelta':  AlgoInfo('sprintzDoubleDelta', preprocs=None, group='Sprintz'), # noqa
    'FastPFOR':     AlgoInfo('fastpfor', needs_32b=True),
    'OptPFOR':      AlgoInfo('optpfor', needs_32b=True),
    'SIMDBP128':    AlgoInfo('binarypacking', needs_32b=True),
    'SIMDGroupSimple': AlgoInfo('simdgroupsimple', needs_32b=True),
    'BitShuffle':   AlgoInfo('blosc_bitshuf{}b'.format(NEEDS_NBITS), levels=DEFAULT_LEVELS), # noqa
    'ByteShuffle':  AlgoInfo('blosc_byteshuf{}b'.format(NEEDS_NBITS), levels=DEFAULT_LEVELS), # noqa
}

# associate each algorithm with a color
cmap = plt.get_cmap('tab20')
for i, (name, info) in enumerate(sorted(ALGO_INFO.items())):
    if info.group == 'Sprintz':
        info.color = 'r'
        # info.marker = '.'

    if i >= 6:
        i += 1  # don't let anything be red (which is color6 in tab20)
    frac = i * (20. / 256.)
    # frac = float(i) / len(ALGO_INFO)
    info.color = cmap(frac)


BENCH_NAME_TO_PRETTY_NAME = dict([(info.lzbench_name, key)
                                 for (key, info) in ALGO_INFO.items()])


PRETTY_DSET_NAMES = {
    'ucr':          'UCR',
    'ampd_gas':     'AMPD Gas',
    'ampd_water':   'AMPD Water',
    'ampd_power':   'AMPD Power',
    'ampd_weather': 'AMPD Weather',
    'uci_gas':      'UCI Gas',
    'pamap':        'PAMAP',
    'msrc':         'MSRC-12',
}
ALL_DSET_NAMES = PRETTY_DSET_NAMES.keys()


# ================================================================ experiments

def _clean_algorithm_name(algo_name):
    tokens = algo_name.split()
    algo_name = BENCH_NAME_TO_PRETTY_NAME[tokens[0]]
    if tokens[-1][0] == '-':  # if compression level given; eg, '-5'
        algo_name += ' ' + tokens[-1]
    return algo_name


def _df_from_string(s, **kwargs):
    return pd.read_csv(StringIO(s), **kwargs)


def _dset_path(nbits, dset, algos, order):
    algos = pyience.ensure_list_or_tuple(algos)

    join = os.path.join
    path = DATASETS_DIR

    # storage order
    assert order in ('c', 'f')  # rowmajor, colmajor order (C, Fortran order)
    path = join(path, 'colmajor') if order == 'f' else join(path, 'rowmajor')

    # storage format (number of bits, whether delta-encoded)
    assert nbits in (8, 16)
    want_deltas = np.array([(name.endswith('-Delta')) for name in algos])
    if np.sum(want_deltas) not in (0, len(algos)):
        raise ValueError('Some algorithms want delta-encoded input, while '
                         'others do not; requires separate commands. Requested'
                         ' algorithms:\n'.format(', '.join(algos)))
    want_deltas = np.sum(want_deltas) > 0

    want_32b = np.array([ALGO_INFO[algo].needs_32b for algo in algos])
    if np.sum(want_32b) not in (0, len(algos)):
        raise ValueError('Some algorithms require 32-bit input, while '
                         'others do not; requires separate commands. Requested'
                         ' algorithms:\n'.format(', '.join(algos)))
    want_32b = np.sum(want_32b) > 0

    if want_deltas and want_32b:
        # also zigzag encode since fastpfor and company assume nonnegative ints
        subdir = '{}-as-uint32_deltas_zigzag'.format(nbits)
    elif want_deltas:
        subdir = 'int{}_deltas'.format(nbits)
    elif want_32b:
        subdir = 'uint{}-as-uint32'.format(nbits)
    else:
        subdir = 'uint{}'.format(nbits)
    path = join(path, subdir)

    # dataset
    return join(path, dset)


def _generate_cmd(nbits, algos, dset_path, memlimit=None, minsecs=1):
    algos = pyience.ensure_list_or_tuple(algos)

    cmd = './lzbench -r -j -o4 -e'  # o4 is csv
    # cmd = './lzbench -r -j -e'
    algo_strs = []
    for algo in algos:
        info = ALGO_INFO[algo]
        s = info.lzbench_name
        s = s.replace(NEEDS_NBITS, str(nbits))
        if info.levels is not None:
            s += ',' + ','.join([str(lvl) for lvl in info.levels])
        algo_strs.append(s)

    cmd += '/'.join(algo_strs)
    if memlimit is not None and int(memlimit) > 0:
        cmd += ' -b{}'.format(int(memlimit))
    cmd += ' -t{},{}'.format(int(minsecs), int(minsecs))
    cmd += ' {}'.format(dset_path)
    return cmd


def _run_experiment(nbits, dsets, algos, memlimit=-1, minsecs=0, order='f',
                    create_fig=False, verbose=1):
    dsets = pyience.ensure_list_or_tuple(dsets)

    for dset in dsets:
        dset_path = _dset_path(nbits=nbits, dset=dset, algos=algos, order=order)
        cmd = _generate_cmd(nbits=nbits, dset_path=dset_path, algos=algos,
                            memlimit=memlimit, minsecs=minsecs)

        if verbose > 0:
            print '------------------------'
            print cmd
        output = os.popen(cmd).read()
        # trimmed = output[output.find('\n') + 1:output.find('\ndone...')]
        trimmed = output[:output.find('\ndone...')]
        # trimmed = trimmed[:]

        if verbose > 1:
            print "raw output:\n" + output
            print "trimmed output:\n", trimmed

        results = _df_from_string(trimmed[:])
        # print "==== results df:\n", results
        # print results_dicts
        results_dicts = results.to_dict('records')
        for d in results_dicts:
            d['Dataset'] = dset
            d['Memlimit'] = memlimit
            d['Minsecs'] = minsecs
            d['Nbits'] = nbits
            d['Order'] = order
            d['Algorithm'] = _clean_algorithm_name(d['Compressor name'])
            d.pop('Compressor name')
            # d.pop('Filename')  # not useful because of -j
        results = pd.DataFrame.from_records(results_dicts)

        if verbose > 0:
            print "==== Results"
            print results

        # dump raw results with a timestamp for archival purposes
        pyience.save_data_frame(results, RESULTS_SAVE_DIR,
                                name='results', timestamp=True)
        # add these results to master set of results, overwriting previous
        # results where relevant
        if os.path.exists(ALL_RESULTS_PATH):
            existing_results = pd.read_csv(ALL_RESULTS_PATH)
            all_results = pd.concat([results, existing_results], axis=0)
            relevant_cols = 'Algorithm Dataset Memlimit Nbits Order'.split()
            all_results.drop_duplicates(subset=relevant_cols, inplace=True)
        else:
            all_results = results

        all_results.to_csv(ALL_RESULTS_PATH, index=False)
        # print "all results ever:\n", all_results

    if create_fig:
        for dset in dsets:
            fig_for_dset(dset, save=True, df=all_results)
            # fig_for_dset(dset, algos=algos, save=True, df=all_results)


# ================================================================ plotting


def _pretty_scatterplot(x, y):
    sb.set_context('talk')
    _, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y)
    ax.set_title('Compression Speed vs Ratio')
    ax.set_xlabel('Compression Speed (MB/s)')
    ax.set_ylabel('Compression Ratio')

    plt.show()


def fig_for_dset(dset, algos=None, save=True, df=None, **sink):

    fig, axes = plt.subplots(2)
    dset_name = PRETTY_DSET_NAMES[dset] if dset in PRETTY_DSET_NAMES else dset
    fig.suptitle(dset_name)

    axes[0].set_title('Compression Speed vs Ratio')
    axes[0].set_xlabel('Compression Speed (MB/s)')
    axes[0].set_ylabel('Compression Ratio')
    axes[1].set_title('Decompression Speed vs Compression Ratio')
    axes[1].set_xlabel('Decompression Speed (MB/s)')
    axes[1].set_ylabel('Compression Ratio')

    if df is None:
        df = pd.read_csv(ALL_RESULTS_PATH)
    # print "read back df"
    # print df

    df = df[df['Dataset'] == dset]
    df = df[df['Algorithm'] != 'Memcpy']

    if algos is None:
        algos = list(df['Algorithm'])
    else:
        df = df[df['Algorithm'].isin(algos)]

    # munge algorithm names
    # raw_algos = df['Algorithm']
    # algos = []
    # for algo in raw_algos:
    #     algos.append(algo_name)

    # print df

    # df['Algorithm'] = raw_algos
    compress_speeds = df['Compression speed'].as_matrix()
    decompress_speeds = df['Decompression speed'].as_matrix()
    ratios = (100. / df['Ratio']).as_matrix()

    for i, algo in enumerate(algos):  # undo artificial boost from 0 padding
        name = algo.split()[0]
        if ALGO_INFO[name].needs_32b:
            nbits = df['Nbits'].iloc[i]
            ratios[i] *= nbits / 32.

    # ratios = 100. / ratios
    # df['Ratio'] = 100. / df['Ratio']

    # print "algos: ", algos
    # print "compress_speeds: ", compress_speeds
    # print "ratios: ", ratios

    # option 1: annotate each point with the algorithm name
    def scatter_plot(ax, x, y):
        # ignore level (eg, '-3') and deltas (eg, Zstd-Delta)
        base_algos = [algo.split()[0].split('-')[0] for algo in algos]

        # scatterplot color-coded by algorithm
        infos = [ALGO_INFO[algo] for algo in base_algos]
        colors = [info.color for info in infos]
        ax.scatter(x, y, c=colors)

        # annotations
        xscale = ax.get_xlim()[1] - ax.get_xlim()[0]
        yscale = ax.get_ylim()[1] - ax.get_ylim()[0]
        perturb_x = .01 * xscale
        perturb_y = .01 * yscale
        for i, algo in enumerate(algos):
            ax.annotate(algo, (x[i] + perturb_x, y[i] + perturb_y))
        ax.margins(0.2)

    scatter_plot(axes[0], compress_speeds, ratios)
    scatter_plot(axes[1], decompress_speeds, ratios)

    # # option 2: color the points by algorithm and have a legend
    # # EDIT: I can't get this to plot into a particular ax; dealbreaker
    # def scatter_plot(ax, x, y, df):
    #     groups = df.groupby('Algorithm')
    #     sb.pairplot(x_vars=[x], y_vars=[y], data=df, hue='Algorithm',
    #                 diag_kws=dict(ax=ax))

    # scatter_plot(axes[0], 'Compression speed', 'Ratio', df)
    # scatter_plot(axes[1], 'Decompression speed', 'Ratio', df)

    plt.tight_layout()
    plt.subplots_adjust(top=.88)
    if save:
        files.ensure_dir_exists(FIG_SAVE_DIR)
        plt.savefig(os.path.join(FIG_SAVE_DIR, dset))
    else:
        plt.show()


def fig_for_dsets(dsets, **kwargs):
    for dset in pyience.ensure_list_or_tuple(dsets):
        fig_for_dset(dset, **kwargs)


# ================================================================ main

def main():
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['Zstd', 'FSE'])
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['FSE'])
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['Huffman'])

    kwargs = pyience.parse_cmd_line()

    if kwargs.get('dsets', None) == 'all':
        kwargs['dsets'] = ALL_DSET_NAMES

    # print kwargs; return

    if kwargs and 'fig' not in kwargs:
        _run_experiment(**kwargs)
    elif 'fig' in kwargs:
        fig_for_dsets(**kwargs)

    # fig_for_dset('ampd_gas')

    # gradient = np.linspace(0, 1, 256)
    # gradient = np.vstack((gradient, gradient))
    # # plt.imshow(gradient, aspect='auto', cmap=plt.get_cmap('Dark2'))
    # # plt.imshow(gradient, aspect='auto', cmap=plt.get_cmap('tab10'))
    # plt.imshow(gradient, aspect='auto', cmap=plt.get_cmap('tab20'))
    # plt.show()

    # # this is how you get the colors out of a cmap; vals in (0, 1)
    # cmap = plt.get_cmap('tab20')
    # print cmap(0)
    # print cmap(.1)
    # print cmap(.11)
    # print cmap(.2)
    # print cmap(.3)
    # # print cmap(26)
    # # print cmap(27)
    # # print cmap(255)


if __name__ == '__main__':
    main()
