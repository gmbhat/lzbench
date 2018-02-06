#!/usr/bin/env python

import os
import matplotlib.pyplot as plt

DATASETS_DIR = '~/Desktop/datasets/compress'  # change this if you aren't me
FIG_SAVE_DIR = 'figs'
RESULTS_SAVE_DIR = 'results'
ALL_RESULTS_PATH = os.path.join(RESULTS_SAVE_DIR, 'all_results.csv')
UCR_RESULTS_PATH = os.path.join(RESULTS_SAVE_DIR, 'ucr', 'ucr_results.csv')


# DEFAULT_LEVELS = [1, 5, 9]  # many compressors have levels 1-9
DEFAULT_LEVELS = [1, 9]  # many compressors have levels 1-9

CAMERA_READY_FONT = 'DejaVu Sans'


class AlgoInfo(object):

    def __init__(self, lzbench_name, levels=None, allow_delta=True,
                 allowed_nbits=[8, 16], needs_32b=False, group=None,
                 allowed_orders=['f'], needs_ndims=False):
        self.lzbench_name = lzbench_name
        self.levels = levels
        self.allow_delta = allow_delta
        self.allowed_nbits = allowed_nbits
        self.needs_32b = needs_32b
        self.group = group
        self.allowed_orders = allowed_orders
        self.needs_ndims = needs_ndims


class DsetInfo(object):

    def __init__(self, pretty_name, bench_name, ndims):
        self.pretty_name = pretty_name
        self.bench_name = bench_name
        self.ndims = ndims


ALL_DSETS = [
    DsetInfo('AMPD Gas', 'ampd_gas', 3),
    DsetInfo('AMPD Water', 'ampd_water', 2),
    DsetInfo('AMPD Power', 'ampd_power', 23),
    DsetInfo('AMPD Weather', 'ampd_weather', 7),
    DsetInfo('MSRC-12', 'msrc', 80),
    DsetInfo('PAMAP', 'pamap', 31),
    DsetInfo('UCI Gas', 'uci_gas', 18),
    DsetInfo('UCR', 'ucr', 1)
]
NAME_2_DSET = {ds.bench_name: ds for ds in ALL_DSETS}
PRETTY_DSET_NAMES = {ds.bench_name: ds.pretty_name for ds in ALL_DSETS}


# PRETTY_DSET_NAMES = {
#     'ucr':          'UCR',
#     'ampd_gas':     'AMPD Gas',
#     'ampd_water':   'AMPD Water',
#     'ampd_power':   'AMPD Power',
#     'ampd_weather': 'AMPD Weather',
#     'uci_gas':      'UCI Gas',
#     'pamap':        'PAMAP',
#     'msrc':         'MSRC-12',
# }

def _sprintz_algo_info(name, nbits=8):
    return AlgoInfo(name, allow_delta=False, allowed_nbits=[nbits],
                    allowed_orders=['c'], group='Sprintz', needs_ndims=True)


ALGO_INFO = {
    'Memcpy':           AlgoInfo('memcpy'),
    # general-purpose compressors
    'Zlib':             AlgoInfo('zlib', levels=DEFAULT_LEVELS),
    'Zstd':             AlgoInfo('zstd', levels=DEFAULT_LEVELS),
    'LZ4':              AlgoInfo('lz4'),
    'LZ4HC':            AlgoInfo('lz4hc', levels=DEFAULT_LEVELS),
    'Gipfeli':          AlgoInfo('gipfeli'),
    'Snappy':           AlgoInfo('snappy'),
    'Brotli':            AlgoInfo('brotli', levels=DEFAULT_LEVELS),
    # just entropy coding
    'FSE':              AlgoInfo('fse'),
    'Huffman':          AlgoInfo('huff0'),
    # integer compressors
    # 'DeltaRLE_HUF':     AlgoInfo('sprDeltaRLE_HUF', allow_delta=False,
    #                              allowed_nbits=[8], group='Sprintz'),
    # 'DeltaRLE':         AlgoInfo('sprDeltaRLE', allow_delta=False,
    #                              allowed_nbits=[8], group='Sprintz'),
    # 'SprDelta':         AlgoInfo('sprintzDelta1d', allow_delta=False,
    #                              allowed_nbits=[8], group='Sprintz'),
    # 'SprDoubleDelta':   AlgoInfo('sprintzDblDelta1d', allow_delta=False,
    #                              allowed_nbits=[8], group='Sprintz'),
    # 'SprDynDelta':      AlgoInfo('sprintzDynDelta1d', allow_delta=False,
    #                              allowed_nbits=[8], group='Sprintz'),
    'SprintzDelta':     _sprintz_algo_info('sprintzDelta'),
    'SprintzXff':       _sprintz_algo_info('sprintzXff'),
    'SprintzDelta_Huf': _sprintz_algo_info('sprintzDelta_HUF'),
    'SprintzXff_Huf':   _sprintz_algo_info('sprintzXff_HUF'),
    'SprintzDelta_16b':     _sprintz_algo_info('sprintzDelta_16b', nbits=16),
    'SprintzXff_16b':       _sprintz_algo_info('sprintzXff_16b', nbits=16),
    'SprintzDelta_Huf_16b': _sprintz_algo_info('sprintzDelta_HUF_16b', nbits=16),
    'SprintzXff_Huf_16b':   _sprintz_algo_info('sprintzXff_HUF_16b', nbits=16),
    'FastPFOR':         AlgoInfo('fastpfor', needs_32b=True),
    'OptPFOR':          AlgoInfo('optpfor', needs_32b=True),
    'SIMDBP128':        AlgoInfo('binarypacking', needs_32b=True),
    'SIMDGroupSimple':  AlgoInfo('simdgroupsimple', needs_32b=True),
    'Simple8B':         AlgoInfo('simple8b', needs_32b=True),
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

USE_WHICH_ALGOS = 'SprintzDelta SprintzXff SprintzDelta_Huf SprintzXff_Huf ' \
    'SprintzDelta_16b SprintzXff_16b SprintzDelta_Huf_16b SprintzXff_Huf_16b '\
    'SIMDBP128 FastPFOR Simple8B ' \
    'Zstd Snappy LZ4 Zlib Huffman'.split()

# PRETTY_DSET_NAMES = {
#     'ucr':          'UCR',
#     'ampd_gas':     'AMPD Gas',
#     'ampd_water':   'AMPD Water',
#     'ampd_power':   'AMPD Power',
#     'ampd_weather': 'AMPD Weather',
#     'uci_gas':      'UCI Gas',
#     'pamap':        'PAMAP',
#     'msrc':         'MSRC-12',
# }
ALL_DSET_NAMES = PRETTY_DSET_NAMES.keys()

PREPROC_TO_INT = {
    'delta':  1,
    'delta2': 2,
    'delta3': 3,
    'delta4': 4,
}

# XXX might actually want to vary Order as an independent var, but for
# now, this is a hack to not have two different memcpy results
# INDEPENDENT_VARS = 'Algorithm Dataset Memlimit Nbits Order Deltas'.split()
INDEPENDENT_VARS = 'Algorithm Dataset Memlimit Nbits Deltas'.split()
DEPENDENT_VARS = ['Ratio', 'Compression speed', 'Decompression speed']
