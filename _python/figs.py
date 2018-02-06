#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from _python import config as cfg
from _python import pyience
from _python import files

DSETS_POSITIONS = {'uci_gas': (0, 0), 'pamap': (0, 1),
                   'msrc':    (1, 0), 'ampd_power': (1, 1),
                   'ampd_gas': (2, 0), 'ampd_weather': (2, 1)}
DSETS_PLOT_SHAPE = (len(DSETS_POSITIONS) // 2, 2)

USE_WHICH_DSETS = DSETS_POSITIONS.keys()
USE_WHICH_DSETS = [cfg.NAME_2_DSET[name] for name in USE_WHICH_DSETS]

FIG_SAVE_DIR = os.path.join(cfg.FIG_SAVE_DIR, 'paper')


def save_fig(name, save_dir=FIG_SAVE_DIR):
    plt.savefig(os.path.join(save_dir, name + '.pdf'), bbox_inches='tight')


def save_fig_png(name, save_dir=FIG_SAVE_DIR):
    plt.savefig(os.path.join(save_dir, name + '.png'), bbox_inches='tight',
                dpi=300)


def _apply_filters(df, nbits=None, order=None, deltas=None,
                   memlimit=None, avg_across_files=True, algos=None,
                   exclude_algos=None):

    if nbits is not None:
        df = df[df['Nbits'] == nbits]
    else:
        raise ValueError("must specify nbits!")

    if order is not None:
        df = df[df['Order'] == order]

    if deltas is not None and deltas:
        df = df[df['Order'] == deltas]

    if memlimit is not None:  # can use -1 for runs without a mem limit
        df = df[df['Memlimit'] == memlimit]

    if avg_across_files:
        df = df.groupby(
            cfg.INDEPENDENT_VARS, as_index=False)[cfg.DEPENDENT_VARS].mean()

    if algos is not None:
        algos_set = set(pyience.ensure_list_or_tuple(algos))
        mask = [algo.split()[0] in algos_set for algo in df['Algorithm']]
        df = df[mask]

    if exclude_algos is not None:
        exclude_set = set(pyience.ensure_list_or_tuple(exclude_algos))
        # print "exclude algos set:", exclude_set
        mask = [algo.split()[0] not in exclude_set for algo in df['Algorithm']]
        df = df[mask]

    return df


def _scatter_plot(ax, x, y, algos, labels, scales=None, colors=None):

    print("labels: ", labels)
    print("labels type: ", type(labels))

    print("labels dtype: ", np.array(labels, dtype=np.object).dtype)
    ax.scatter(x, y, s=scales, c=colors, label=np.array(labels))

    # annotations
    xscale = ax.get_xlim()[1] - ax.get_xlim()[0]
    yscale = ax.get_ylim()[1] - ax.get_ylim()[0]
    perturb_x = .01 * xscale
    perturb_y = .01 * yscale
    for i, algo in enumerate(algos):
        # algo = algo + '-Delta' if used_delta[i] else algo
        ax.annotate(algo, (x[i] + perturb_x, y[i] + perturb_y),
                    rotation=30, rotation_mode='anchor')
    ax.margins(0.2)


def _decomp_vs_ratio_plot(dset, ax, df):
    # dset_name = dset.pretty_name
    # dset_name = cfg.PRETTY_DSET_NAMES[dset] if dset in cfg.PRETTY_DSET_NAMES else dset

    print("using dset ", dset)
    # print "df"
    # print df

    df = df[df['Dataset'] == dset.bench_name]

    algos = list(df['Algorithm'])
    # used_delta = list(df['Deltas'])

    # compress_speeds = df['Compression speed'].as_matrix()
    decompress_speeds = df['Decompression speed'].as_matrix()
    ratios = (100. / df['Ratio']).as_matrix()

    for i, algo in enumerate(algos):  # undo artificial boost from 0 padding
        name = algo.split()[0]  # ignore level
        if cfg.ALGO_INFO[name].needs_32b:
            nbits = df['Nbits'].iloc[i]
            ratios[i] *= nbits / 32.

    # compute colors for each algorithm in scatterplot
    # ignore level (eg, '-3') and deltas (eg, Zstd-Delta)
    base_algos = [algo.split()[0] for algo in algos]
    infos = [cfg.ALGO_INFO[algo] for algo in base_algos]
    colors = [info.color for info in infos]
    scales = [36 if info.group != 'Sprintz' else 64 for info in infos]

    # print "algos: ", algos
    # print "decompress_speeds", decompress_speeds
    # print "ratios", ratios
    _scatter_plot(ax, decompress_speeds, ratios, algos,
                  labels=algos, colors=colors, scales=scales)


def decomp_vs_ratio_fig(nbits=None, deltas=None, memlimit=None, order=None,
                        save=False):
    fig, axes = plt.subplots(*DSETS_PLOT_SHAPE, figsize=(6, 7))
    axes = axes.reshape(DSETS_PLOT_SHAPE)

    for ax in axes[:, 0]:
        ax.set_xlabel('Decompression Speed (MB/s)')
    for ax in axes[0, :]:
        ax.set_ylabel('Compression Ratio')

    # print "axes.shape", axes.shape

    dsets = USE_WHICH_DSETS
    df = pd.read_csv(cfg.ALL_RESULTS_PATH)
    df = _apply_filters(df, nbits=nbits, memlimit=memlimit, order=order)
    for dset in dsets:
        position = DSETS_POSITIONS[dset.bench_name]
        ax = axes[position[0], position[1]]
        ax.set_title(dset.pretty_name)
        _decomp_vs_ratio_plot(dset=dset, ax=ax, df=df)

    leg_lines, leg_labels = axes.ravel()[0].get_legend_handles_labels()
    plt.figlegend(leg_lines, leg_labels, loc='lower center',
                  ncol=3, labelspacing=0)

    plt.suptitle('Decompression Speed vs Compression Ratio')
    plt.subplots_adjust(top=.90, bottom=.2)
    plt.tight_layout()

    save_dir = cfg.FIG_SAVE_DIR
    if nbits is not None:
        save_dir = os.path.join(save_dir, '{}b'.format(nbits))
    if deltas is not None:
        save_dir += '_deltas' if deltas else ''
    if memlimit is not None and memlimit > 0:
        save_dir += '_{}KB'.format(memlimit)
    if order is not None:
        save_dir += '_{}'.format(order)
    if save:
        files.ensure_dir_exists(save_dir)
        plt.savefig(os.path.join(save_dir, dset.bench_name))
    else:
        plt.show()


def _compute_ranks(df, lower_better=True):
    """assumes each row of X is a dataset and each col is an algorithm"""
    # return df.rank(axis=1, numeric_only=True, ascending=lower_better)
    return df.rank(axis=1, numeric_only=True, ascending=lower_better, method='min')


def cd_diagram(df, lower_better=True):
    import Orange as orn  # requires py3.4 or greater environment

    ranks = _compute_ranks(df, lower_better=lower_better)
    names = [s.strip() for s in ranks.columns]
    mean_ranks = ranks.mean(axis=0)
    ndatasets = df.shape[0]

    print("--- raw ranks:")
    print(ranks)

    print("--- mean ranks:")
    print("\n".join(["{}: {}".format(name, rank)
                     for (name, rank) in zip(names, mean_ranks)]))

    # alpha must be one of {'0.1', '0.05', '0.01'}
    cd = orn.evaluation.compute_CD(mean_ranks, ndatasets, alpha='0.1')
    # cd = orn.evaluation.compute_CD(mean_ranks, ndatasets, alpha='0.05')
    orn.evaluation.graph_ranks(mean_ranks, names, cd=cd, reverse=True)
    print("\nNemenyi critical difference: ", cd)


def cd_diagram_ours_vs_others(others_deltas=False, save=True):
    df = pd.read_csv(cfg.UCR_RESULTS_PATH)

    # df = df[df['Order'] == 'c']
    df['Filename'] = df['Filename'].apply(lambda s: os.path.basename(s).split('.')[0])
    df = df.sort_values(['Filename', 'Algorithm'])
    if others_deltas:
        df = df[df['Deltas'] | df['Algorithm'].str.startswith('Sprintz')]
    else:
        df = df[~df['Deltas']]
    df = df[df['Algorithm'] != 'Memcpy']
    df = df[['Nbits', 'Algorithm', 'Filename', 'Ratio', 'Order']]
    df['Ratio'] = 100. / df['Ratio']  # bench reports % of original size
    full_df = df
    # for nbits in [8]:
    for nbits in (8, 16):
        df = full_df[full_df['Nbits'] == nbits]
        df = df.drop('Nbits', axis=1)
        # df = df.sort_values('Filename')

        # # counts = df.groupby(['Filename', 'Algorithm']).count()
        # counts = df.groupby(['Algorithm']).count()
        # print("fname, algo counts: ")
        # print(counts)
        # print(counts.min())
        # print(counts.max())
        # # print(df[:560].groupby('Filename').count())
        # # print(df[560:1120].groupby('Filename').count())
        # # print(df[-560:].groupby('Filename').count())

        # # return

        # df.to_csv('df_ucr.csv')

        # print("df8 algos", df8['Algorithm'])

        # if if complains about duplicate entries here, might because
        # memcpy ran on some stuff (but not everything for some reason)
        df = df.pivot(index='Filename', columns='Algorithm', values='Ratio')

        # print("df8", df8[:20])
        # print("df8", df8)
        # print("df8 cols", df8.columns.names)

        for col in df:
            info = cfg.ALGO_INFO.get(col)
            if info is not None and info.needs_32b:
                print("scaling comp ratio for col: {}".format(col))
                # print("old df col:")
                # print(df[col][:10])

                # df[col] = df[col] * (32 / nbits)  # if using % of orig
                df[col] = df[col] * (nbits / 32)    # if using orig/compressed

                # print("new df col:")
                # print(df[col][:10])

        df.to_csv('ucr_df_{}.csv'.format(nbits))

        cd_diagram(df)  # if using % of orig
        cd_diagram(df, lower_better=False)
        if save:
            save_fig_png('cd_diagram_{}b_deltas={}'.format(
                nbits, int(others_deltas)))
        else:
            plt.show()

    # print df8.groupby()

    # print(df[:2])


def cd_diagram_ours_vs_others_no_delta():
    cd_diagram_ours_vs_others(others_deltas=False)


def cd_diagram_ours_vs_others_delta():
    cd_diagram_ours_vs_others(others_deltas=True)


def decomp_vs_ndims_results():
    # df = pd.read_csv(cfg.NDIMS_SPEED_RESULTS_PATH)
    df = pd.read_csv(cfg.NDIMS_SPEED_RESULTS_PATH.replace('ndims_speed_results', '_ndims_speed_results'))

    # df = df[df['Order'] == 'c']
    # df['Filename'] = df['Filename'].apply(lambda s: os.path.basename(s).split('.')[0])
    # df = df.sort_values(['Filename', 'Algorithm'])
    df = df[df['Algorithm'] != 'Memcpy']
    df['Ratio'] = 100. / df['Ratio']  # bench reports % of original size

    # extract ndims from algo name; eg: "SprintzDelta -9" -> 9
    df['Ndims'] = df['Algorithm'].apply(lambda algo: -int(algo.split()[1]))
    df['TrueRatio'] = df['Filename'].apply(
        lambda path: int(path.split("ratio=")[1].split('.')[0]))
    df = df[['Nbits', 'Algorithm', 'Ratio', 'TrueRatio']]

    full_df = df
    # for nbits in (8, 16):
    for nbits in [8]:
        df = full_df[full_df['Nbits'] == nbits]
        df = df.drop('Nbits', axis=1)

        print("df")
        print(df)


def comp_vs_ndims_results():
    pass


def quantize_err_results():
    pass


def main():
    # camera-ready can't have Type 3 fonts, which are what matplotlib
    # uses by default; 42 is apparently TrueType fonts
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.family'] = cfg.CAMERA_READY_FONT

    # decomp_vs_ratio_fig(nbits=8)

    # cd_diagram_ours_vs_others()
    decomp_vs_ndims_results()


if __name__ == '__main__':
    main()
