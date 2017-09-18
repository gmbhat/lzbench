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

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

DATASETS_DIR = '~/Desktop/datasets/compress'  # change this if you aren't me
DEFAULT_LEVELS = [1, 5, 9]  # many compressors have levels 1-9

NEEDS_NBITS = '<nbits>'


class AlgoInfo(object):

    def __init__(self, lzbench_name, levels=None, preprocs=['delta'], needs_32b=False):
        self.lzbench_name = lzbench_name
        self.levels = levels
        self.preprocs = preprocs  # try with each of these variants of data
        self.needs_32b = needs_32b


ALGO_INFO = {
    # general-purpose compressors
    'Zlib':         AlgoInfo('zlib', levels=DEFAULT_LEVELS),
    'Zstd':         AlgoInfo('zstd', levels=DEFAULT_LEVELS),
    'LZ4':          AlgoInfo('lz4', levels=DEFAULT_LEVELS),
    'Gipfeli':      AlgoInfo('gipfeli'),
    'Snappy':       AlgoInfo('snappy'),
    'FSE':       AlgoInfo('fse'),
    'Huffman':       AlgoInfo('huff0'),
    # integer compressors
    'Delta':        AlgoInfo('sprintzDelta', preprocs=None),
    'DoubleDelta':  AlgoInfo('sprintzDoubleDelta', preprocs=None),
    'FastPFOR':     AlgoInfo('fastpfor', needs_32b=True),
    'OptPFOR':     AlgoInfo('optpfor', needs_32b=True),
    'SIMDBP128':    AlgoInfo('binarypacking', needs_32b=True),
    'SIMDGroupSimple': AlgoInfo('simdgroupsimple', needs_32b=True),
    'BitShuffle':   AlgoInfo('blosc_bitshuf{}b'.format(NEEDS_NBITS), levels=DEFAULT_LEVELS), # noqa
    'ByteShuffle':  AlgoInfo('blosc_byteshuf{}b'.format(NEEDS_NBITS), levels=DEFAULT_LEVELS), # noqa
}


def _pretty_scatterplot(x, y):
    sb.set_context('talk')
    _, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y)
    ax.set_title('Compression Speed vs Ratio')
    ax.set_xlabel('Compression Speed (MB/s)')
    ax.set_ylabel('Compression Ratio')

    plt.show()


def df_from_string(s, **kwargs):
    return pd.read_csv(StringIO(s), **kwargs)


def _dset_path(nbits, dset, algos, order):
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


def _run_experiment(nbits, dsets, algos, memlimit=None, minsecs=1, order='f',
                    verbose=1):

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

        # print "\n" + output
        if verbose > 0:
            # print "trimmed output:"
            print trimmed

        results = df_from_string(trimmed[:])
        print "==== results df:\n", results



        # SELF: pick up here by adding params to the df, then saving it



        # kvs = {'Algorithm: ': }


# ================================================================ main

def main():
    # _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['Zstd', 'FSE'])
    _run_experiment(nbits=8, dsets=['ampd_gas'], algos=['FSE'])


if __name__ == '__main__':
    main()
