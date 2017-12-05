#BUILD_ARCH = 32-bit

ifeq ($(BUILD_ARCH),32-bit)
	CODE_FLAGS += -m32
	LDFLAGS += -m32
	DONT_BUILD_LZSSE ?= 1
endif

CC?=gcc
COMPILER = $(shell $(CC) -v 2>&1 | grep -q "clang version" && echo clang || echo gcc)
GCC_VERSION = $(shell $(CC) -dumpversion | sed -e 's:\([0-9.]*\).*:\1:' -e 's:\.\([0-9][0-9]\):\1:g' -e 's:\.\([0-9]\):0\1:g')
CLANG_VERSION = $(shell $(CC) -v 2>&1 | grep "clang version" | sed -e 's:.*version \([0-9.]*\).*:\1:' -e 's:\.\([0-9][0-9]\):\1:g' -e 's:\.\([0-9]\):0\1:g')

# glza doesn't work with gcc < 4.9 and clang < 3.6 (missing stdatomic.h)
ifeq (1,$(filter 1,$(shell [ "$(COMPILER)" = "gcc" ] && expr $(GCC_VERSION) \< 40900) $(shell [ "$(COMPILER)" = "clang" ] && expr $(CLANG_VERSION) \< 30600)))
    DONT_BUILD_GLZA ?= 1
endif

# LZSSE requires gcc with support of __SSE4_1__
ifeq ($(shell echo|$(CC) -dM -E - -march=native|grep -c SSE4_1), 0)
	DONT_BUILD_LZSSE ?= 1
endif


# detect Windows
ifneq (,$(filter Windows%,$(OS)))
	ifeq ($(COMPILER),clang)
		DONT_BUILD_GLZA ?= 1
	endif
	LDFLAGS += -lshell32 -lole32 -loleaut32 -static
else
	ifeq ($(shell uname -p),powerpc)
		# density and yappy don't work with big-endian PowerPC
		DONT_BUILD_DENSITY ?= 1
		DONT_BUILD_YAPPY ?= 1
		DONT_BUILD_ZLING ?= 1
	endif

	# MacOS doesn't support -lrt -static
	ifeq ($(shell uname -s),Darwin)
		DONT_BUILD_LZHAM ?= 1
		DONT_BUILD_CSC ?= 1
	else
		LDFLAGS	+= -lrt -static
	endif
	LDFLAGS	+= -lpthread
endif


DEFINES     += -I. -Izstd/lib -Izstd/lib/common -Ixpack/common -Ilibcsc
DEFINES     += -DHAVE_CONFIG_H
CODE_FLAGS  += -Wno-unknown-pragmas -Wno-sign-compare -Wno-conversion
# OPT_FLAGS   ?= -fomit-frame-pointer -fstrict-aliasing -ffast-math
OPT_FLAGS   ?= -fomit-frame-pointer -fstrict-aliasing -ffast-math


ifeq ($(BUILD_TYPE),debug)
	OPT_FLAGS_O2 = $(OPT_FLAGS) -g
	OPT_FLAGS_O3 = $(OPT_FLAGS) -g
else
	OPT_FLAGS_O2 = $(OPT_FLAGS) -O2 -DNDEBUG
	OPT_FLAGS_O3 = $(OPT_FLAGS) -O3 -DNDEBUG -march=native
endif

CFLAGS = $(MOREFLAGS) $(CODE_FLAGS) $(OPT_FLAGS_O3) $(DEFINES)
CFLAGS_O2 = $(MOREFLAGS) $(CODE_FLAGS) $(OPT_FLAGS_O2) $(DEFINES)
LDFLAGS += $(MOREFLAGS)
# enable cpp11, but disable this warning as a hacky workaround for Tornado
CXX_ONLY_FLAGS = -std=c++0x -Wno-c++11-narrowing

LZBENCH_FILES =  _lzbench/lzbench.o _lzbench/compressors.o _lzbench/output.o
LZBENCH_FILES += _lzbench/preprocessing.o _lzbench/main.o

LZO_FILES = lzo/lzo1.o lzo/lzo1a.o lzo/lzo1a_99.o lzo/lzo1b_1.o lzo/lzo1b_2.o lzo/lzo1b_3.o lzo/lzo1b_4.o lzo/lzo1b_5.o
LZO_FILES += lzo/lzo1b_6.o lzo/lzo1b_7.o lzo/lzo1b_8.o lzo/lzo1b_9.o lzo/lzo1b_99.o lzo/lzo1b_9x.o lzo/lzo1b_cc.o
LZO_FILES += lzo/lzo1b_d1.o lzo/lzo1b_d2.o lzo/lzo1b_rr.o lzo/lzo1b_xx.o lzo/lzo1c_1.o lzo/lzo1c_2.o lzo/lzo1c_3.o
LZO_FILES += lzo/lzo1c_4.o lzo/lzo1c_5.o lzo/lzo1c_6.o lzo/lzo1c_7.o lzo/lzo1c_8.o lzo/lzo1c_9.o lzo/lzo1c_99.o
LZO_FILES += lzo/lzo1c_9x.o lzo/lzo1c_cc.o lzo/lzo1c_d1.o lzo/lzo1c_d2.o lzo/lzo1c_rr.o lzo/lzo1c_xx.o lzo/lzo1f_1.o
LZO_FILES += lzo/lzo1f_9x.o lzo/lzo1f_d1.o lzo/lzo1f_d2.o lzo/lzo1x_1.o lzo/lzo1x_1k.o lzo/lzo1x_1l.o lzo/lzo1x_1o.o
LZO_FILES += lzo/lzo1x_9x.o lzo/lzo1x_d1.o lzo/lzo1x_d2.o lzo/lzo1x_d3.o lzo/lzo1x_o.o lzo/lzo1y_1.o lzo/lzo1y_9x.o
LZO_FILES += lzo/lzo1y_d1.o lzo/lzo1y_d2.o lzo/lzo1y_d3.o lzo/lzo1y_o.o lzo/lzo1z_9x.o lzo/lzo1z_d1.o lzo/lzo1z_d2.o
LZO_FILES += lzo/lzo1z_d3.o lzo/lzo1_99.o lzo/lzo2a_9x.o lzo/lzo2a_d1.o lzo/lzo2a_d2.o lzo/lzo_crc.o lzo/lzo_init.o
LZO_FILES += lzo/lzo_ptr.o lzo/lzo_str.o lzo/lzo_util.o

UCL_FILES = ucl/alloc.o ucl/n2b_99.o ucl/n2b_d.o ucl/n2b_ds.o ucl/n2b_to.o ucl/n2d_99.o ucl/n2d_d.o ucl/n2d_ds.o
UCL_FILES += ucl/n2d_to.o ucl/n2e_99.o ucl/n2e_d.o ucl/n2e_ds.o ucl/n2e_to.o ucl/ucl_crc.o ucl/ucl_init.o
UCL_FILES += ucl/ucl_ptr.o ucl/ucl_str.o ucl/ucl_util.o

ZLIB_FILES = zlib/adler32.o zlib/compress.o zlib/crc32.o zlib/deflate.o zlib/gzclose.o zlib/gzlib.o zlib/gzread.o
ZLIB_FILES += zlib/gzwrite.o zlib/infback.o zlib/inffast.o zlib/inflate.o zlib/inftrees.o zlib/trees.o
ZLIB_FILES += zlib/uncompr.o zlib/zutil.o

LZMAT_FILES = lzmat/lzmat_dec.o lzmat/lzmat_enc.o

LZRW_FILES = lzrw/lzrw1-a.o lzrw/lzrw1.o lzrw/lzrw2.o lzrw/lzrw3.o lzrw/lzrw3-a.o

LZMA_FILES = lzma/LzFind.o lzma/LzmaDec.o lzma/LzmaEnc.o

LZ4_FILES = lizard/lizard_compress.o lizard/lizard_decompress.o lz4/lz4.o lz4/lz4hc.o

LZF_FILES = lzf/lzf_c_ultra.o lzf/lzf_c_very.o lzf/lzf_d.o

LZFSE_FILES = lzfse/lzfse_decode.o lzfse/lzfse_decode_base.o lzfse/lzfse_encode.o lzfse/lzfse_encode_base.o lzfse/lzfse_fse.o lzfse/lzvn_decode.o lzfse/lzvn_decode_base.o lzfse/lzvn_encode_base.o

QUICKLZ_FILES = quicklz/quicklz151b7.o quicklz/quicklz1.o quicklz/quicklz2.o quicklz/quicklz3.o

SNAPPY_FILES = snappy/snappy-sinksource.o snappy/snappy-stubs-internal.o snappy/snappy.o

BROTLI_FILES = brotli/common/dictionary.o brotli/dec/bit_reader.o brotli/dec/decode.o brotli/dec/huffman.o brotli/dec/state.o
BROTLI_FILES += brotli/enc/backward_references.o brotli/enc/block_splitter.o brotli/enc/brotli_bit_stream.o brotli/enc/encode.o
BROTLI_FILES += brotli/enc/entropy_encode.o brotli/enc/histogram.o brotli/enc/literal_cost.o brotli/enc/memory.o
BROTLI_FILES += brotli/enc/metablock.o brotli/enc/static_dict.o brotli/enc/utf8_util.o brotli/enc/compress_fragment.o brotli/enc/compress_fragment_two_pass.o
BROTLI_FILES += brotli/enc/cluster.o brotli/enc/bit_cost.o brotli/enc/backward_references_hq.o brotli/enc/dictionary_hash.o

ZSTD_FILES = zstd/lib/decompress/zstd_decompress.o zstd/lib/decompress/huf_decompress.o zstd/lib/common/zstd_common.o zstd/lib/common/fse_decompress.o
ZSTD_FILES += zstd/lib/common/xxhash.o zstd/lib/common/error_private.o zstd/lib/common/entropy_common.o
ZSTD_FILES += zstd/lib/compress/zstd_compress.o zstd/lib/compress/fse_compress.o zstd/lib/compress/huf_compress.o

BRIEFLZ_FILES = brieflz/brieflz.o brieflz/depacks.o

LIBLZG_FILES = liblzg/decode.o liblzg/encode.o liblzg/checksum.o

XZ_FILES = xz/lzma_decoder.o xz/lzma_encoder.o xz/common.o xz/price_table.o xz/fastpos_table.o xz/lzma_encoder_optimum_fast.o xz/lzma_encoder_optimum_normal.o
XZ_FILES += xz/lz_decoder.o xz/lz_encoder.o xz/lz_encoder_mf.o xz/alone.o xz/alone_encoder.o xz/alone_decoder.o xz/lzma_encoder_presets.o xz/crc32_table.o

XPACK_FILES = xpack/lib/x86_cpu_features.o xpack/lib/xpack_common.o xpack/lib/xpack_compress.o xpack/lib/xpack_decompress.o

GIPFELI_FILES = gipfeli/decompress.o gipfeli/entropy.o gipfeli/entropy_code_builder.o gipfeli/gipfeli-internal.o gipfeli/lz77.o

LIBDEFLATE_FILES = libdeflate/adler32.o libdeflate/aligned_malloc.o libdeflate/crc32.o libdeflate/deflate_compress.o
LIBDEFLATE_FILES += libdeflate/deflate_decompress.o libdeflate/gzip_compress.o libdeflate/gzip_decompress.o
LIBDEFLATE_FILES += libdeflate/x86_cpu_features.o libdeflate/zlib_compress.o libdeflate/zlib_decompress.o

EXAMPLE_FILES = example_compressor/example.o

MISC_FILES = crush/crush.o shrinker/shrinker.o fastlz/fastlz.o pithy/pithy.o lzjb/lzjb2010.o wflz/wfLZ.o
MISC_FILES += lzlib/lzlib.o slz/slz.o

# SIMDCOMP_FILES = simdcomp/avx512bitpacking.o simdcomp/avxbitpacking.o
# SIMDCOMP_FILES += simdcomp/simdbitpacking.o simdcomp/simdpackedsearch.o
# SIMDCOMP_FILES += simdcomp/simdintegratedbitpacking.o simdcomp/simdcomputil.o
# SIMDCOMP_FILES += simdcomp/simdpackedselect.o

FASTPFOR_FILES  = fastpfor/bitpacking.o fastpfor/varintdecode.o
FASTPFOR_FILES += fastpfor/bitpackingaligned.o fastpfor/bitpackingunaligned.o
FASTPFOR_FILES += fastpfor/horizontalbitpacking.o fastpfor/simdbitpacking.o
FASTPFOR_FILES += fastpfor/simdunalignedbitpacking.o fastpfor/streamvbyte.o

BLOSC_FILES  = blosc/bitshuffle-avx2.o blosc/bitshuffle-generic.o
BLOSC_FILES += blosc/shuffle-avx2.o blosc/shuffle-generic.o blosc/bitshuffle-sse2.o
BLOSC_FILES += blosc/shuffle-sse2.o blosc/shuffle.o blosc/blosc.o blosc/blosclz.o

BBP_FILES  = bbp/bbp.o bbp/bitpacking.o bbp/bitstream.o bbp/coding.o
BBP_FILES += bbp/coding_helpers.o bbp/common.o

SPRINTZ_FILES = sprintz/univariate_8b.o sprintz/delta.o sprintz/predict.o
SPRINTZ_FILES += sprintz/sprintz_delta.o sprintz/sprintz_delta_lowdim.o
SPRINTZ_FILES += sprintz/sprintz_xff.o sprintz/sprintz_xff_lowdim.o
SPRINTZ_FILES += sprintz/sprintz_8b.o sprintz/format.o

ifeq "$(DONT_BUILD_CSC)" "1"
    DEFINES += -DBENCH_REMOVE_CSC
else
	CSC_FILES = libcsc/csc_analyzer.o libcsc/csc_coder.o libcsc/csc_dec.o libcsc/csc_enc.o libcsc/csc_encoder_main.o
	CSC_FILES += libcsc/csc_filters.o libcsc/csc_lz.o libcsc/csc_memio.o libcsc/csc_mf.o libcsc/csc_model.o libcsc/csc_profiler.o libcsc/csc_default_alloc.o
endif

ifeq "$(DONT_BUILD_DENSITY)" "1"
    DEFINES += -DBENCH_REMOVE_DENSITY
else
	DENSITY_FILES = density/block_decode.o density/block_encode.o density/block_footer.o density/block_header.o density/block_mode_marker.o
	DENSITY_FILES += density/buffer.o density/globals.o density/kernel_chameleon_decode.o density/kernel_chameleon_dictionary.o
	DENSITY_FILES += density/kernel_chameleon_encode.o density/kernel_cheetah_decode.o density/kernel_cheetah_dictionary.o
	DENSITY_FILES += density/kernel_cheetah_encode.o density/kernel_lion_decode.o density/kernel_lion_dictionary.o
	DENSITY_FILES += density/kernel_lion_encode.o density/kernel_lion_form_model.o density/main_decode.o density/main_encode.o
	DENSITY_FILES += density/main_footer.o density/main_header.o density/memory_location.o density/memory_teleport.o density/stream.o
	DENSITY_FILES += density/spookyhash/spookyhash.o density/spookyhash/context.o
endif

ifeq "$(DONT_BUILD_GLZA)" "1"
    DEFINES += -DBENCH_REMOVE_GLZA
else
    GLZA_FILES = glza/GLZAcomp.o glza/GLZAformat.o glza/GLZAcompress.o glza/GLZAencode.o glza/GLZAdecode.o glza/GLZAmodel.o
endif

ifeq "$(DONT_BUILD_LZHAM)" "1"
    DEFINES += -DBENCH_REMOVE_LZHAM
else
    LZHAM_FILES = lzham/lzham_assert.o lzham/lzham_checksum.o lzham/lzham_huffman_codes.o lzham/lzham_lzbase.o
    LZHAM_FILES += lzham/lzham_lzcomp.o lzham/lzham_lzcomp_internal.o lzham/lzham_lzdecomp.o lzham/lzham_lzdecompbase.o
    LZHAM_FILES += lzham/lzham_match_accel.o lzham/lzham_mem.o lzham/lzham_platform.o lzham/lzham_lzcomp_state.o
    LZHAM_FILES += lzham/lzham_prefix_coding.o lzham/lzham_symbol_codec.o lzham/lzham_timer.o lzham/lzham_vector.o lzham/lzham_lib.o
endif

ifeq "$(DONT_BUILD_LZSSE)" "1"
    DEFINES += -DBENCH_REMOVE_LZSSE
else
    LZSSE_FILES = lzsse/lzsse2/lzsse2.o lzsse/lzsse4/lzsse4.o lzsse/lzsse8/lzsse8.o
endif

ifeq "$(DONT_BUILD_TORNADO)" "1"
    DEFINES += "-DBENCH_REMOVE_TORNADO"
    LZMA_FILES += lzma/Alloc.o
else
    MISC_FILES += tornado/tor_test.o
endif

ifeq "$(DONT_BUILD_YAPPY)" "1"
    DEFINES += -DBENCH_REMOVE_YAPPY
else
    MISC_FILES += yappy/yappy.o
endif

ifeq "$(DONT_BUILD_ZLING)" "1"
    DEFINES += -DBENCH_REMOVE_ZLING
else
	ZLING_FILES = libzling/libzling.o libzling/libzling_huffman.o libzling/libzling_lz.o libzling/libzling_utils.o
endif

ifeq "$(BENCH_HAS_NAKAMICHI)" "1"
    DEFINES += -DBENCH_HAS_NAKAMICHI
	MISC_FILES += nakamichi/Nakamichi_Okamigan.o
endif

all: lzbench

# FIX for SEGFAULT on GCC 4.9+
wflz/wfLZ.o: wflz/wfLZ.c wflz/wfLZ.h
	$(CC) $(CFLAGS_O2) $< -c -o $@

shrinker/shrinker.o: shrinker/shrinker.c
	$(CC) $(CFLAGS_O2) $< -c -o $@

lzmat/lzmat_dec.o: lzmat/lzmat_dec.c
	$(CC) $(CFLAGS_O2) $< -c -o $@

lzmat/lzmat_enc.o: lzmat/lzmat_enc.c
	$(CC) $(CFLAGS_O2) $< -c -o $@

lzrw/lzrw1-a.o: lzrw/lzrw1-a.c
	$(CC) $(CFLAGS_O2) $< -c -o $@

lzrw/lzrw1.o: lzrw/lzrw1.c
	$(CC) $(CFLAGS_O2) $< -c -o $@

pithy/pithy.o: pithy/pithy.cpp
	$(CXX) $(CFLAGS_O2) $< -c -o $@

lzsse/lzsse2/lzsse2.o: lzsse/lzsse2/lzsse2.cpp
	$(CXX) $(CFLAGS) $(CXX_ONLY_FLAGS) -msse4.1 $< -c -o $@

lzsse/lzsse4/lzsse4.o: lzsse/lzsse4/lzsse4.cpp
	$(CXX) $(CFLAGS) $(CXX_ONLY_FLAGS) -msse4.1 $< -c -o $@

lzsse/lzsse8/lzsse8.o: lzsse/lzsse8/lzsse8.cpp
	$(CXX) $(CFLAGS) $(CXX_ONLY_FLAGS) -msse4.1 $< -c -o $@

nakamichi/Nakamichi_Okamigan.o: nakamichi/Nakamichi_Okamigan.c
	$(CC) $(CFLAGS) -mavx $< -c -o $@

_lzbench/lzbench.o: _lzbench/lzbench.cpp _lzbench/lzbench.h

lzbench: $(ZSTD_FILES) $(GLZA_FILES) $(LZSSE_FILES) $(LZFSE_FILES) 			\
		$(XPACK_FILES) $(GIPFELI_FILES) $(XZ_FILES) $(LIBLZG_FILES) 		\
		$(BRIEFLZ_FILES) $(LZF_FILES) $(LZRW_FILES) $(BROTLI_FILES) 		\
		$(CSC_FILES) $(LZMA_FILES) $(DENSITY_FILES) $(ZLING_FILES) 			\
		$(QUICKLZ_FILES) $(SNAPPY_FILES) $(ZLIB_FILES) $(LZHAM_FILES) 		\
		$(LZO_FILES) $(UCL_FILES) $(LZMAT_FILES) $(LZ4_FILES) 				\
		$(LIBDEFLATE_FILES) $(EXAMPLE_FILES) $(FASTPFOR_FILES) 				\
		$(BLOSC_FILES) $(BBP_FILES)	$(SPRINTZ_FILES)						\
		$(MISC_FILES) $(LZBENCH_FILES)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo Linked GCC_VERSION=$(GCC_VERSION) CLANG_VERSION=$(CLANG_VERSION) COMPILER=$(COMPILER)

.c.o:
	$(CC) $(CFLAGS) $< -std=c99 -c -o $@

.cc.o:
	$(CXX) $(CFLAGS) $(CXX_ONLY_FLAGS) $< -c -o $@

.cpp.o:
	$(CXX) $(CFLAGS) $(CXX_ONLY_FLAGS) $< -c -o $@

clean:
	rm -rf lzbench lzbench.exe *.o */*.o

	# rm -rf lzbench lzbench.exe *.o _lzbench/*.o slz/*.o zstd/lib/*.o zstd/lib/*.a zstd/lib/common/*.o zstd/lib/compress/*.o zstd/lib/decompress/*.o lzsse/lzsse2/*.o lzsse/lzsse4/*.o lzsse/lzsse8/*.o lzfse/*.o xpack/lib/*.o blosclz/*.o gipfeli/*.o xz/*.o liblzg/*.o lzlib/*.o brieflz/*.o brotli/common/*.o brotli/enc/*.o brotli/dec/*.o libcsc/*.o wflz/*.o lzjb/*.o lzma/*.o density/spookyhash/*.o density/*.o pithy/*.o glza/*.o libzling/*.o yappy/*.o shrinker/*.o fastlz/*.o ucl/*.o zlib/*.o lzham/*.o lzmat/*.o lizard/*.o lz4/*.o crush/*.o lzf/*.o lzrw/*.o lzo/*.o snappy/*.o quicklz/*.o tornado/*.o libdeflate/*.o nakamichi/*.o
