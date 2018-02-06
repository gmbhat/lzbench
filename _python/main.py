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

# some queries I like: TODO put these somewhere sensible
#
# profile raw bitpacking speed
#   ./lzbench -r -asprJustBitpack/sprFixedBitpack -t0,0 -i25,25 -j synthetic/100M_randint_0_1.dat
#
# run everything and create figure for one dataset (here, uci_gas):
#   python -m _python.main --sweep algos=SprintzDelta,SprintzDelta_16b,
#       SprintzDelta_HUF,SprintzDelta_HUF_16b,Zstd,Brotli,LZ4,Huffman,
#       FastPFOR,SIMDBP128 --dsets=uci_gas --miniters=10 --create_fig


import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sb

import sys
from . import files
from . import pyience
from . import config as cfg

# import gflags   # google's command line lib; pip install python-gflags
# FLAGS = gflags.FLAGS

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

# gflags.DEFINE_

DEBUG = False


# ================================================================ experiments

def _clean_algorithm_name(algo_name):
    algo_name = algo_name.strip()
    # print "cleaning algo name: '{}'".format(algo_name)
    tokens = algo_name.split(' ')
    algo_name = cfg.BENCH_NAME_TO_PRETTY_NAME[tokens[0]]
    if tokens[-1][0] == '-':  # if compression level given; eg, '-5'
        algo_name += ' ' + tokens[-1]
    # print "cleaned algo name: '{}'".format(algo_name)
    return algo_name


def _df_from_string(s, **kwargs):
    return pd.read_csv(StringIO(s), **kwargs)


def _dset_path(nbits, dset, algos, order, deltas):
    algos = pyience.ensure_list_or_tuple(algos)

    join = os.path.join
    path = cfg.DATASETS_DIR

    # storage order
    assert order in ('c', 'f')  # rowmajor, colmajor order (C, Fortran order)
    path = join(path, 'colmajor') if order == 'f' else join(path, 'rowmajor')

    # storage format (number of bits, whether delta-encoded)
    assert nbits in (8, 16)

    # algos = _canonical_algo_names(algos)
    want_32b = np.array([cfg.ALGO_INFO[algo].needs_32b for algo in algos])
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

    return join(path, subdir, dset)


def _generate_cmd(nbits, algos, dset_path, preprocs=None, memlimit=None,
                  miniters=1, use_u32=False, ndims=None, dset_name=None):
    algos = pyience.ensure_list_or_tuple(algos)

    # cmd = './lzbench -r -j -o4 -e'  # o4 is csv
    cmd = './lzbench -r -o4 -t0,0'  # o4 is csv
    # cmd += ' -i{},{}'.format(int(miniters), int(miniters))
    cmd += ' -i{},{}'.format(2, int(miniters))  # XXX compress full number of trials
    cmd += ' -a'
    algo_strs = []
    for algo in algos:
        algo = algo.split('-')[0]  # rm possible '-Delta' suffix
        info = cfg.ALGO_INFO[algo]
        s = info.lzbench_name
        # s = s.replace(NEEDS_NBITS, str(nbits))
        if info.levels is not None:
            s += ',' + ','.join([str(lvl) for lvl in info.levels])
        if info.needs_ndims:
            ndims = cfg.NAME_2_DSET[dset_name].ndims
            s += ',{}'.format(int(ndims))
        algo_strs.append(s)

    cmd += '/'.join(algo_strs)
    if memlimit is not None and int(memlimit) > 0:
        cmd += ' -b{}'.format(int(memlimit))
    if preprocs is not None:
        preprocs = pyience.ensure_list_or_tuple(preprocs)
        for preproc in preprocs:
            # EDIT: this is handled by feeding it delta + zigzag
            # version of the data, which is really optimistic but lets
            # us get meaningful results since relevant algos can't
            # bitpack negative numbers at all (in provided impls)
            #
            # preproc_const = PREPROC_TO_INT[preproc.lower()]
            # if preproc_const == 1 and use_u32:
            #     # use FastPFOR vectorized delta impl instead of regular
            #     # scalar delta for FastPFOR 32b funcs
            #     preproc_const = 4
            # cmd += ' -d{}'.format(preproc_const)
            cmd += ' -d{}'.format(cfg.PREPROC_TO_INT[preproc.lower()])

        if not use_u32:
            cmd += ' -e{}'.format(int(nbits / 8))

    cmd += ' {}'.format(dset_path)
    return cmd


def _run_experiment(nbits, algos, dsets=None, memlimit=-1, miniters=0, order='f',
                    deltas=False, create_fig=False, verbose=1, dry_run=DEBUG,
                    save_path=None, **sink):
    dsets = cfg.ALL_DSET_NAMES if dsets is None else dsets
    dsets = pyience.ensure_list_or_tuple(dsets)
    algos = pyience.ensure_list_or_tuple(algos)

    for dset in dsets:
        # if verbose > 0:
        #     print "================================ {}".format(dset)

        # don't tell dset_path about delta encoding; we'll use the benchmark's
        # preprocessing abilities for that, so that the time gets taken into
        # account
        dset_path = _dset_path(nbits=nbits, dset=dset, algos=algos,
                               order=order, deltas=deltas)
        use_u32 = 'zigzag' in dset_path  # TODO this is hacky
        preprocs = 'delta' if (deltas and not use_u32) else None
        cmd = _generate_cmd(nbits=nbits, dset_path=dset_path, algos=algos,
                            preprocs=preprocs, memlimit=memlimit,
                            miniters=miniters, use_u32=use_u32, dset_name=dset)

        if verbose > 0 or dry_run:
            print '------------------------'
            print cmd

            if dry_run:
                # print "Warning: abandoning early for debugging!"
                continue

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
            d['Filename'] = d['Filename'].replace(
                os.path.expanduser(cfg.DATASETS_DIR), '')
            # d.pop('Filename')  # not useful because of -j
        results = pd.DataFrame.from_records(results_dicts)

        if verbose > 0:
            print "==== Results"
            print results

        # print "returning prematurely"; return  # TODO rm

        # dump raw results with a timestamp for archival purposes
        pyience.save_data_frame(results, cfg.RESULTS_SAVE_DIR,
                                name='results', timestamp=True)

        # pyience.save_data_frame(results, save_dir,
        #                         name='results', timestamp=True)
        # if save_dir != cfg.RESULTS_SAVE_DIR:

        # add these results to master set of results, overwriting previous
        # results where relevant
        if save_path is None:
            save_path = cfg.ALL_RESULTS_PATH

        if os.path.exists(save_path):
            existing_results = pd.read_csv(save_path)
            all_results = pd.concat([results, existing_results], axis=0)
            all_results.drop_duplicates(  # add filename since not doing '-j'
                subset=(cfg.INDEPENDENT_VARS + ['Filename']), inplace=True)
        else:
            all_results = results

        all_results.to_csv(save_path, index=False)
        # print "all results ever:\n", all_results

    if create_fig and not dry_run:
        for dset in dsets:
            fig_for_dset(dset, save=True, df=all_results, nbits=nbits)
            # fig_for_dset(dset, algos=algos, save=True, df=all_results)

    return all_results


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
                 memlimit=-1, avg_across_files=True, order=None, **sink):

    fig, axes = plt.subplots(2, figsize=(9, 9))
    dset_name = cfg.PRETTY_DSET_NAMES[dset] if dset in cfg.PRETTY_DSET_NAMES else dset
    fig.suptitle(dset_name)

    axes[0].set_title('Compression Speed vs Ratio')
    axes[0].set_xlabel('Compression Speed (MB/s)')
    axes[0].set_ylabel('Compression Ratio')
    axes[1].set_title('Decompression Speed vs Compression Ratio')
    axes[1].set_xlabel('Decompression Speed (MB/s)')
    axes[1].set_ylabel('Compression Ratio')

    if df is None:
        df = pd.read_csv(cfg.ALL_RESULTS_PATH)
    # print "read back df"
    # print df

    df = df[df['Dataset'] == dset]

    if order is not None:
        df = df[df['Order'] == order]
    # df = df[df['Algorithm'] != 'Memcpy']

    # print "using algos before exlude deltas: ", sorted(list(df['Algorithm']))
    # return

    if exclude_deltas:
        df = df[~df['Deltas']]

    # print "using algos before memlimit: ", sorted(list(df['Algorithm']))

    if memlimit is not None:  # can use -1 for runs without a mem limit
        df = df[df['Memlimit'] == memlimit]

    if avg_across_files:
        df = df.groupby(
            cfg.INDEPENDENT_VARS, as_index=False)[cfg.DEPENDENT_VARS].mean()
        # print "means: "
        # print df
        # return

    if nbits is not None:
        df = df[df['Nbits'] == nbits]
    else:
        raise ValueError("must specify nbits!")

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

    print "fig_for_dset: using algos: ", sorted(list(df['Algorithm']))
    # return

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
        if cfg.ALGO_INFO[name].needs_32b:
            nbits = df['Nbits'].iloc[i]
            ratios[i] *= nbits / 32.

    # compute colors for each algorithm in scatterplot
    # ignore level (eg, '-3') and deltas (eg, Zstd-Delta)
    # base_algos = [algo.split()[0].split('-')[0] for algo in algos]
    base_algos = [algo.split()[0] for algo in algos]
    infos = [cfg.ALGO_INFO[algo] for algo in base_algos]
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
    save_dir = cfg.FIG_SAVE_DIR
    if nbits is not None:
        save_dir = os.path.join(save_dir, '{}b'.format(nbits))
    if exclude_deltas:
        save_dir += '_nodeltas'
    if memlimit is not None and memlimit > 0:
        save_dir += '_{}KB'.format(memlimit)
    if order is not None:
        save_dir += '_{}'.format(order)
    if save:
        files.ensure_dir_exists(save_dir)
        plt.savefig(os.path.join(save_dir, dset))
    else:
        plt.show()


def fig_for_dsets(dsets=None, **kwargs):
    if dsets is None:
        dsets = cfg.ALL_DSET_NAMES
    for dset in pyience.ensure_list_or_tuple(dsets):
        fig_for_dset(dset, **kwargs)


# ================================================================ main

def run_sweep(algos=None, create_fig=False, nbits=None, all_use_u32=None,
              deltas=None, miniters=0, memlimit=-1,
              orders=['c', 'f'], dsets=None, **kwargs):
    # TODO I should just rename these vars
    all_nbits = nbits
    all_use_deltas = deltas  # 'deltas' was a more intuitive kwarg
    all_dsets = dsets  # rename kwarg for consistency
    all_algos = algos
    all_orders = orders

    if all_nbits is None:
        # all_nbits = [16]
        all_nbits = [8, 16]
    if all_use_u32 is None:
        # all_use_u32 = [True]
        all_use_u32 = [True, False]
    if all_use_deltas is None:
        # all_use_deltas = [True]
        all_use_deltas = [True, False]
    if all_orders is None:
        all_orders = ['c', 'f']
    if all_dsets is None:
        all_dsets = cfg.ALL_DSET_NAMES
    if all_algos is None:
        all_algos = ('Zstd LZ4 LZ4HC Snappy FSE Huffman FastPFOR Delta ' +
                     'DoubleDelta DeltaRLE_HUF DeltaRLE BitShuffle8b ' +
                     'ByteShuffle8b').split()

    all_nbits = pyience.ensure_list_or_tuple(all_nbits)
    all_use_u32 = pyience.ensure_list_or_tuple(all_use_u32)
    all_use_deltas = pyience.ensure_list_or_tuple(all_use_deltas)
    all_orders = pyience.ensure_list_or_tuple(all_orders)
    all_dsets = pyience.ensure_list_or_tuple(all_dsets)
    all_algos = pyience.ensure_list_or_tuple(all_algos)

    # all_algorithms = ('BitShuffle8b ByteShuffle8b').split()
    # all_dsets = ['PAMAP']

    # delta_algos = [algo for algo in algos if ALGO_INFO[algo].allow_delta]
    # delta_u32_algos = [algo for algo in delta_algos if ALGO_INFO[algo].needs_32b]

    all_combos = itertools.product(
        all_nbits, all_use_u32, all_use_deltas, all_orders)
    for (use_nbits, use_u32, use_deltas, use_order) in all_combos:
        # filter algorithms with incompatible requirements
        algos = []
        for algo in all_algos:
            info = cfg.ALGO_INFO[algo]
            if use_nbits not in info.allowed_nbits:
                continue
            if use_u32 != info.needs_32b:
                continue
            if use_deltas and not info.allow_delta:
                continue
            if use_order not in info.allowed_orders:
                continue
            algos.append(algo)

        if len(algos) == 0:
            continue

        _run_experiment(algos=algos, dsets=all_dsets, nbits=use_nbits,
                        deltas=use_deltas, memlimit=memlimit, miniters=miniters,
                        order=use_order, create_fig=create_fig, **kwargs)


def run_ucr():
    save_path = UCR_RESULTS_PATH
    run_sweep(algos=cfg.USE_WHICH_ALGOS, miniters=5, save_path=save_path,
              dsets='ucr')


def main():
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['Zstd', 'FSE'])
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['FSE'])
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['Huffman'])

    kwargs = pyience.parse_cmd_line()

    if kwargs.get('run_ucr', False):
        run_ucr()
        return


    if kwargs is not None and kwargs.get('sweep', False):
        run_sweep(**kwargs)
        return

    if kwargs.get('dsets', None) == 'all':
        kwargs['dsets'] = cfg.ALL_DSET_NAMES

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
