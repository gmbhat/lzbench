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

import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sb

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

# DEFAULT_LEVELS = [1, 5, 9]  # many compressors have levels 1-9
DEFAULT_LEVELS = [1, 9]  # many compressors have levels 1-9

# NEEDS_NBITS = '<nbits>'

# np.random.seed(123)


class AlgoInfo(object):

    def __init__(self, lzbench_name, levels=None, allow_delta=True,
                 allowed_nbits=[8, 16], needs_32b=False, group=None):
        self.lzbench_name = lzbench_name
        self.levels = levels
        self.allow_delta = allow_delta
        self.allowed_nbits = allowed_nbits
        self.needs_32b = needs_32b
        self.group = group


ALGO_INFO = {
    'Memcpy':           AlgoInfo('memcpy'),
    # general-purpose compressors
    'Zlib':             AlgoInfo('zlib', levels=DEFAULT_LEVELS),
    'Zstd':             AlgoInfo('zstd', levels=DEFAULT_LEVELS),
    'LZ4':              AlgoInfo('lz4'),
    'LZ4HC':            AlgoInfo('lz4hc', levels=DEFAULT_LEVELS),
    'Gipfeli':          AlgoInfo('gipfeli'),
    'Snappy':           AlgoInfo('snappy'),
    # just entropy coding
    'FSE':              AlgoInfo('fse'),
    'Huffman':          AlgoInfo('huff0'),
    # integer compressors
    'DeltaRLE':         AlgoInfo('sprDeltaRLE', allow_delta=False,
                                 allowed_nbits=[8], group='Sprintz'),
    'Delta':            AlgoInfo('sprintzDelta', allow_delta=False,
                                 allowed_nbits=[8], group='Sprintz'),
    'DoubleDelta':      AlgoInfo('sprintzDblDelta', allow_delta=False,
                                 allowed_nbits=[8], group='Sprintz'),
    'DynDelta':         AlgoInfo('sprintzDynDelta', allow_delta=False,
                                 allowed_nbits=[8], group='Sprintz'),
    'FastPFOR':         AlgoInfo('fastpfor', needs_32b=True),
    'OptPFOR':          AlgoInfo('optpfor', needs_32b=True),
    'SIMDBP128':        AlgoInfo('binarypacking', needs_32b=True),
    'SIMDGroupSimple':  AlgoInfo('simdgroupsimple', needs_32b=True),
    'BitShuffle8b':     AlgoInfo('blosc_bitshuf8b', allowed_nbits=[8],
                                 allow_delta=False, levels=DEFAULT_LEVELS),
    'ByteShuffle8b':    AlgoInfo('blosc_byteshuf8b', allowed_nbits=[8],
                                 allow_delta=False, levels=DEFAULT_LEVELS),
    'BitShuffle16b':    AlgoInfo('blosc_bitshuf16b', allowed_nbits=[16],
                                 allow_delta=False, levels=DEFAULT_LEVELS),
    'ByteShuffle16b':   AlgoInfo('blosc_byteshuf16b', allowed_nbits=[16],
                                 allow_delta=False, levels=DEFAULT_LEVELS),
}

# associate each algorithm with a color
# cmap = plt.get_cmap('tab20')
cmap = plt.get_cmap('tab10')
for i, (name, info) in enumerate(sorted(ALGO_INFO.items())):
    if info.group == 'Sprintz':
        # info.color = 'r'
        info.color = plt.get_cmap('tab20')(4 * 20. / 256)  # red
        continue
        # print "set info color to red for algorithm {} (group {})".format(name, info.group)

    if i >= 6:
        i += 1  # don't let anything be red (which is color6 in tab20)
    frac = i * (13 / 256.)
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

PREPROC_TO_INT = {
    'delta':  1,
    'delta2': 2,
    'delta3': 3,
    'delta4': 4,
}

INDEPENDENT_VARS = 'Algorithm Dataset Memlimit Nbits Order Deltas'.split()
DEPENDENT_VARS = ['Ratio', 'Compression speed', 'Decompression speed']


# ================================================================ experiments

def _clean_algorithm_name(algo_name):
    algo_name = algo_name.strip()
    # print "cleaning algo name: '{}'".format(algo_name)
    tokens = algo_name.split(' ')
    algo_name = BENCH_NAME_TO_PRETTY_NAME[tokens[0]]
    if tokens[-1][0] == '-':  # if compression level given; eg, '-5'
        algo_name += ' ' + tokens[-1]
    return algo_name


def _df_from_string(s, **kwargs):
    return pd.read_csv(StringIO(s), **kwargs)


def _dset_path(nbits, dset, algos, order, deltas):
    algos = pyience.ensure_list_or_tuple(algos)

    join = os.path.join
    path = DATASETS_DIR

    # storage order
    assert order in ('c', 'f')  # rowmajor, colmajor order (C, Fortran order)
    path = join(path, 'colmajor') if order == 'f' else join(path, 'rowmajor')

    # storage format (number of bits, whether delta-encoded)
    assert nbits in (8, 16)

    # algos = _canonical_algo_names(algos)
    want_32b = np.array([ALGO_INFO[algo].needs_32b for algo in algos])
    if np.sum(want_32b) not in (0, len(algos)):
        raise ValueError('Some algorithms require 32-bit input, while '
                         'others do not; requires separate commands. Requested'
                         ' algorithms:\n'.format(', '.join(algos)))
    want_32b = np.sum(want_32b) > 0

    if deltas and want_32b:
        # also zigzag encode since fastpfor and company assume nonnegative ints
        subdir = 'uint{}-as-uint32_deltas_zigzag'.format(nbits)
    # elif deltas:
        # subdir = 'int{}_deltas'.format(nbits)
    elif want_32b:
        subdir = 'uint{}-as-uint32'.format(nbits)
    else:
        subdir = 'uint{}'.format(nbits)
    path = join(path, subdir)

    # dataset
    return join(path, dset)


def _generate_cmd(nbits, algos, dset_path, preprocs=None, memlimit=None, miniters=1):
    algos = pyience.ensure_list_or_tuple(algos)

    # cmd = './lzbench -r -j -o4 -e'  # o4 is csv
    cmd = './lzbench -r -o4 -t0,0 -a'  # o4 is csv
    algo_strs = []
    for algo in algos:
        algo = algo.split('-')[0]  # rm possible '-Delta' suffix
        info = ALGO_INFO[algo]
        s = info.lzbench_name
        # s = s.replace(NEEDS_NBITS, str(nbits))
        if info.levels is not None:
            s += ',' + ','.join([str(lvl) for lvl in info.levels])
        algo_strs.append(s)

    cmd += '/'.join(algo_strs)
    if memlimit is not None and int(memlimit) > 0:
        cmd += ' -b{}'.format(int(memlimit))
    cmd += ' -i{},{}'.format(int(miniters), int(miniters))
    if preprocs is not None:
        preprocs = pyience.ensure_list_or_tuple(preprocs)
        for preproc in preprocs:
            cmd += ' -d{}'.format(PREPROC_TO_INT[preproc.lower()])

    cmd += ' {}'.format(dset_path)
    return cmd


def _run_experiment(nbits, algos, dsets=None, memlimit=-1, miniters=0, order='f',
                    deltas=False, create_fig=False, verbose=1):
    dsets = ALL_DSET_NAMES if dsets is None else dsets
    dsets = pyience.ensure_list_or_tuple(dsets)
    algos = pyience.ensure_list_or_tuple(algos)

    for dset in dsets:
        # don't tell dset_path about delta encoding; we'll use the benchmark's
        # preprocessing abilities for that, so that the time gets taken into
        # account
        dset_path = _dset_path(nbits=nbits, dset=dset, algos=algos,
                               order=order, deltas=deltas)
        preprocs = 'delta' if deltas else None
        cmd = _generate_cmd(nbits=nbits, dset_path=dset_path, algos=algos,
                            preprocs=preprocs, memlimit=memlimit,
                            miniters=miniters)

        if verbose > 0:
            print '------------------------'
            print cmd
        output = os.popen(cmd).read()
        # trimmed = output[output.find('\n') + 1:output.find('\ndone...')]
        trimmed = output[:output.find('\ndone...')]
        # trimmed = trimmed[:]

        if not os.path.exists('./lzbench'):
            os.path.system('make')

        if verbose > 1:
            print "raw output:\n" + output
            print "trimmed output:\n", trimmed

        results = _df_from_string(trimmed[:])
        # print "==== results df:\n", results
        # print results_dicts
        results_dicts = results.to_dict('records')
        for i, d in enumerate(results_dicts):
            d['Dataset'] = dset
            d['Memlimit'] = memlimit
            d['MinIters'] = miniters
            d['Nbits'] = nbits
            d['Order'] = order
            d['Deltas'] = deltas
            d['Algorithm'] = _clean_algorithm_name(d['Compressor name'])
            # if deltas and algo != 'Memcpy':
            #     d['Algorithm'] = d['Algorithm'] + '-Delta'
            d.pop('Compressor name')
            # d['Filename'] = d['Filename'].replace(os.path.expanduser('~'), '~')
            d['Filename'] = d['Filename'].replace(os.path.expanduser(DATASETS_DIR), '')
            # d.pop('Filename')  # not useful because of -j
        results = pd.DataFrame.from_records(results_dicts)

        if verbose > 0:
            print "==== Results"
            print results

        # print "returning prematurely"; return  # TODO rm

        # dump raw results with a timestamp for archival purposes
        pyience.save_data_frame(results, RESULTS_SAVE_DIR,
                                name='results', timestamp=True)
        # add these results to master set of results, overwriting previous
        # results where relevant
        if os.path.exists(ALL_RESULTS_PATH):
            existing_results = pd.read_csv(ALL_RESULTS_PATH)
            all_results = pd.concat([results, existing_results], axis=0)
            all_results.drop_duplicates(  # add filename since not doing '-j'
                subset=(INDEPENDENT_VARS + ['Filename']), inplace=True)
        else:
            all_results = results

        all_results.to_csv(ALL_RESULTS_PATH, index=False)
        # print "all results ever:\n", all_results

    if create_fig:
        for dset in dsets:
            fig_for_dset(dset, save=True, df=all_results, nbits=nbits)
            # fig_for_dset(dset, algos=algos, save=True, df=all_results)


# ================================================================ plotting


# def _pretty_scatterplot(x, y):
#     sb.set_context('talk')
#     _, ax = plt.subplots(figsize=(7, 7))
#     ax.scatter(x, y)
#     ax.set_title('Compression Speed vs Ratio')
#     ax.set_xlabel('Compression Speed (MB/s)')
#     ax.set_ylabel('Compression Ratio')

#     plt.show()


def fig_for_dset(dset, algos=None, save=True, df=None, nbits=None,
                 exclude_algos=None, exclude_deltas=False,
                 avg_across_files=True, **sink):

    fig, axes = plt.subplots(2, figsize=(9, 9))
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

    if exclude_deltas:
        df = df[~df['Deltas']]

    if avg_across_files:
        df = df.groupby(INDEPENDENT_VARS, as_index=False)[DEPENDENT_VARS].mean()
        # print "means: "
        # print df
        # return

    if nbits is not None:
        df = df[df['Nbits'] == nbits]

    # if algos is None:
        # algos = list(df['Algorithm'])
    # else:
    if algos is not None:
        algos_set = set(pyience.ensure_list_or_tuple(algos))
        mask = [algo.split()[0] in algos_set for algo in df['Algorithm']]
        df = df[mask]

    if exclude_algos is not None:
        exclude_set = set(pyience.ensure_list_or_tuple(exclude_algos))
        # print "exclude algos set:", exclude_set
        mask = [algo.split()[0] not in exclude_set for algo in df['Algorithm']]
        df = df[mask]

    algos = list(df['Algorithm'])
    used_delta = list(df['Deltas'])

    # print "pruned df to:"
    # print df; return

    # # munge algorithm names
    # new_algos = []
    # for algo, delta in zip(algos, df['Deltas']):
    #     new_algos.append(algo + '-Delta' if delta else algo)
    # algos = new_algos

    # print df

    # df['Algorithm'] = raw_algos
    compress_speeds = df['Compression speed'].as_matrix()
    decompress_speeds = df['Decompression speed'].as_matrix()
    ratios = (100. / df['Ratio']).as_matrix()

    for i, algo in enumerate(algos):  # undo artificial boost from 0 padding
        name = algo.split()[0]  # ignore level
        if ALGO_INFO[name].needs_32b:
            nbits = df['Nbits'].iloc[i]
            ratios[i] *= nbits / 32.

    # compute colors for each algorithm in scatterplot
    # ignore level (eg, '-3') and deltas (eg, Zstd-Delta)
    # base_algos = [algo.split()[0].split('-')[0] for algo in algos]
    base_algos = [algo.split()[0] for algo in algos]
    infos = [ALGO_INFO[algo] for algo in base_algos]
    colors = [info.color for info in infos]

    # ratios = 100. / ratios
    # df['Ratio'] = 100. / df['Ratio']
    # print "algos: ", algos
    # print "compress_speeds: ", compress_speeds
    # print "ratios: ", ratios

    # option 1: annotate each point with the algorithm name
    def scatter_plot(ax, x, y, colors=None):
        ax.scatter(x, y, c=colors)

        # annotations
        xscale = ax.get_xlim()[1] - ax.get_xlim()[0]
        yscale = ax.get_ylim()[1] - ax.get_ylim()[0]
        perturb_x = .01 * xscale
        perturb_y = .01 * yscale
        for i, algo in enumerate(algos):
            algo = algo + '-Delta' if used_delta[i] else algo
            ax.annotate(algo, (x[i] + perturb_x, y[i] + perturb_y))
        ax.margins(0.2)

    scatter_plot(axes[0], compress_speeds, ratios, colors=colors)
    scatter_plot(axes[1], decompress_speeds, ratios, colors=colors)

    for ax in axes:
        # ax.set_xscale('log')
        ax.set_ylim([.95, ax.get_ylim()[1]])

    plt.tight_layout()
    plt.subplots_adjust(top=.88)
    save_dir = FIG_SAVE_DIR
    if nbits is not None:
        save_dir = os.path.join(save_dir, '{}b'.format(nbits))
    if save:
        files.ensure_dir_exists(save_dir)
        plt.savefig(os.path.join(save_dir, dset))
    else:
        plt.show()


def fig_for_dsets(dsets=None, **kwargs):
    if dsets is None:
        dsets = ALL_DSET_NAMES
    for dset in pyience.ensure_list_or_tuple(dsets):
        fig_for_dset(dset, **kwargs)


# ================================================================ main

def run_it_all(create_fig=False, all_nbits=None, all_use_u32=None,
               all_use_deltas=None, **sink):
    if all_nbits is None:
        # all_nbits = [16]
        all_nbits = [8, 16]
    if all_use_u32 is None:
        # all_use_u32 = [True]
        all_use_u32 = [True, False]
    if all_use_deltas is None:
        # all_use_deltas = [True]
        all_use_deltas = [True, False]

    all_algorithms = ('Zstd LZ4 LZ4HC Snappy FSE Huffman FastPFOR Delta ' +
                      'DoubleDelta DeltaRLE BitShuffle8b ByteShuffle8b').split()
    # all_algorithms = ('BitShuffle8b ByteShuffle8b').split()
    all_dsets = ALL_DSET_NAMES
    # all_dsets = ['PAMAP']

    memlimit = -1
    miniters = 0  # TODO larger num for real experiments
    order = 'f'  # might need to iterate over 'c' and 'f' order at some point

    # delta_algos = [algo for algo in algos if ALGO_INFO[algo].allow_delta]
    # delta_u32_algos = [algo for algo in delta_algos if ALGO_INFO[algo].needs_32b]

    all_combos = itertools.product(all_nbits, all_use_u32, all_use_deltas)
    for (use_nbits, use_u32, use_deltas) in all_combos:
        # filter algorithms with incompatible requirements
        algos = []
        for algo in all_algorithms:
            info = ALGO_INFO[algo]
            if use_nbits not in info.allowed_nbits:
                continue
            if use_u32 != info.needs_32b:
                continue
            if use_deltas and not info.allow_delta:
                continue
            algos.append(algo)

        # print "nbits, use_u32, use_deltas = {}, {}, {}:\n\t{}".format(
        #     use_nbits, use_u32, use_deltas, ', '.join(algos))
        # continue  # TODO rm

        if len(algos) == 0:
            continue

        _run_experiment(algos=algos, dsets=all_dsets, nbits=use_nbits,
                        deltas=use_deltas, memlimit=memlimit, miniters=miniters,
                        order=order, create_fig=create_fig)


def main():
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['Zstd', 'FSE'])
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['FSE'])
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['Huffman'])

    kwargs = pyience.parse_cmd_line()

    if kwargs is not None and kwargs.get('everything', False):
        run_it_all(**kwargs)
        return

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
