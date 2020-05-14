

#include "lzbench.h"
#include "preprocessing.h"


#ifndef BENCH_REMOVE_FASTPFOR
    #include "fastpfor/deltautil.h"  // for simd delta preproc
#endif
#ifndef BENCH_REMOVE_SPRINTZ
    #include "sprintz/delta.h"  // for simd delta preproc
    #include "sprintz/predict.h"  // for simd xff preproc
    #include "sprintz/format.h"  // for simd delta preproc?
    #include "sprintz/online.hpp"  // for dynamic delta
#endif
#ifndef BENCH_REMOVE_BLOSC
    #include "blosc/shuffle.h"
#endif

namespace lzbench {

static const int64_t kDoubleDeltaThreshold = 1 << 16;


bool is_func_valid(int func) {
    if ((func != DELTA) && (func != DOUBLE_DELTA) && (func != XFF)
        && (func != ZIGZAG)
        && (func != DYNAMIC_DELTA) && (func != DYNAMIC_DELTA_ALT)
        && (func != SPRINTZPACK) && (func != SPRINTZPACK_NOZIGZAG)
        && (func != BITSHUFFLE) && (func != BYTESHUFFLE)
        )
    {
        printf("WARNING: ignoring unrecognized preprocessor function %d\n",
            func);
        return false;
    }
    return true;
}

// void apply_preprocessors(const std::vector<int64_t>& preprocessors,
size_t apply_preprocessors(const std::vector<preproc_params_t>& preprocessors,
    const uint8_t* orig_inbuf, size_t size, int element_sz, uint8_t* outbuf, uint8_t* tmpbuf)
{
    // printf("using %lu preprocessors; size=%lu, element_sz=%d\n", preprocessors.size(), size, element_sz);

    if (preprocessors.size() < 1) { return size; }

    int sz = element_sz;
    if (sz < 1) { sz = 1; }
    int64_t nelements = size / sz;

    // printf("size=%lu, sz=%lu, element_sz=%d\n", size, sz, element_sz);
    // printf("size=%lu, element_sz=%lu, nelements=%lld\n", size, sz, element_sz, nelements);

    // auto orig_inbuf = inbuf;
    auto orig_outbuf = outbuf;
    auto orig_tmpbuf = tmpbuf;

    memcpy(tmpbuf, orig_inbuf, size);
    auto inbuf = (const uint8_t*)tmpbuf;
    // const uint8_t* inbuf = orig_inbuf;

    // for (auto preproc : preprocessors) {
    bool needs_ptr_swap = false;
    int nswaps = 0;
    for (int i = 0; i < preprocessors.size(); i++) {
        auto preproc = preprocessors[i];

        auto func = preproc.func;
        if (!is_func_valid(func)) {
            continue;
        }

        // if (false) { // TODO rm
        // if (i % 2) { // swap buffers
        if (needs_ptr_swap) { // swap buffers
            // printf("swapping input and output buffers\n");
            auto tmp = inbuf;
            inbuf = (const uint8_t*)outbuf;
            outbuf = (uint8_t*)tmp;
            // tmpbuf = outbuf;
            // outbuf = tmpbuf;
            nswaps++;
        }
        needs_ptr_swap = true; // execute each time but the first

        // printf("applying preproc: %lld with nelements=%lld, element_sz=%d\n", preproc, nelements, sz)
        // continue;

        // if ((preproc < 1) || (preproc > 4)) {


//         if (preproc == 0) {
//             printf("WARNING: ignoring unrecognized preprocessor number '0'\n");
//             continue;
//         }
// #ifdef BENCH_REMOVE_SPRINTZ
//         if (preproc < 1) {
//             printf("WARNING: ignoring unrecognized preprocessor number '%lld'\n", preproc);
//             continue;
//         }
// #endif

        // int64_t offset = preproc;  // simplifying hack based on enum values
        // int64_t offset = preproc.offset;
        int64_t stride = preproc.stride;
        // printf("using offset: %lld\n", offset);

        // printf("alright, we got to here; about to maybe apply a preproc...\n");

        // use simd delta if available
#ifndef BENCH_REMOVE_FASTPFOR
        if (sz == 4 && func == DELTA && stride == 4)  {
            memcpy(outbuf, inbuf, size);
            FastPForLib::Delta::deltaSIMD((uint32_t*)outbuf, nelements);
            continue;
        }
#endif
#ifndef BENCH_REMOVE_SPRINTZ   // use simd delta if available
        // ------------------------ delta
        // if (sz == 1 && (offset > 2) && (offset < kDoubleDeltaThreshold)) {
        if (sz == 1 && (stride > 2) && func == DELTA) {
            // printf("applying 8b delta encoding...\n");
            encode_delta_rowmajor_8b(inbuf, nelements, (int8_t*)outbuf, stride, false);
            continue;
        }
        // if (sz == 2 && (offset > 2) && (offset < kDoubleDeltaThreshold)) {
        if (sz == 2 && (stride > 2) && func == DELTA) {
            // printf("applying 16b delta encoding...\n");
            // printf("enc delta: %d elements\n", nelements);
            encode_delta_rowmajor_16b((const uint16_t*)inbuf, nelements,
                (int16_t*)outbuf, stride, false);
            // size = (nelements + 1) / 2;
            continue;
        }
        // ------------------------ double delta
        if (sz == 1 && func == DOUBLE_DELTA) {
            // printf("applying 8b double delta encoding...\n");
            size = encode_doubledelta_rowmajor_8b(
                inbuf, nelements, (int8_t*)outbuf, stride, false);
            nelements = size;
            // printf("got thru encoding without exception...\n");
            continue;
        }
        // if (sz == 2 && offset >= kDoubleDeltaThreshold) {
        if (sz == 2 && func == DOUBLE_DELTA) {
            // printf("applying 16b double delta encoding...\n");
            // stride = offset % kDoubleDeltaThreshold;
            // nelements = encode_doubledelta_rowmajor_16b(
            encode_doubledelta_rowmajor_16b(
                (const uint16_t*)inbuf, nelements, (int16_t*)outbuf,
                stride, false);
            // size = (nelements + 1) / 2;
            continue;
        }
        // ------------------------ xff
        if (sz == 1 && func == XFF) {
            // printf("about to run 8b xff encoding using offset %d...\n", (int)offset);
            // size = encode_xff_rowmajor_8b(inbuf, nelements, (int8_t*)outbuf, stride, false);
            encode_xff_rowmajor_8b(inbuf, nelements, (int8_t*)outbuf, stride, false);
            // nelements = size;
            // printf("...ran xff encoding\n");
            continue;
        }
        if (sz == 2 && func == XFF) {
            // nelements = encode_xff_rowmajor_16b((const uint16_t*)inbuf, nelements,
            encode_xff_rowmajor_16b((const uint16_t*)inbuf, nelements,
                (int16_t*)outbuf, stride, false);
            // size = (nelements + 1) / 2;
            // printf("...ran xff encoding\n");
            continue;
        }

        // ------------------------ dynamic delta
        if (sz == 2 && func == DYNAMIC_DELTA) {
            // printf("initial size, nelements: %d, %d\n", size, nelements);
            size = 2 * dynamic_delta_pack_u16( // 2x to convert to bytes
                (const uint16_t*)inbuf, nelements, (int16_t*)outbuf);
            nelements = (size + sz - 1) / sz;
            // printf("new     size, nelements: %d, %d\n", size, nelements);
            continue;
        }
        if (sz == 2 && func == DYNAMIC_DELTA_ALT) {
            // printf("initial size, nelements: %d, %d\n", size, nelements);
            size = 2 * dynamic_delta_pack_u16_altloss(
                (const uint16_t*)inbuf, nelements, (int16_t*)outbuf);
            nelements = (size + sz - 1) / sz;
            // printf("new     size, nelements: %d, %d\n", size, nelements);
            continue;
        }

        // ------------------------ zigzag
        if (sz == 2 && func == ZIGZAG) {
            // printf("enc buff has length in elements: %d\n", nelements);
            // printf("enc buff has size in bytes: %d\n", size);
            // auto data_in = (const uint16_t*)inbuf;
            // auto data_out = (int16_t*)outbuf;
            // for (len_t i = 0; i < nelements; i++) {
            //     // data_out[i] = zigzag_decode_16b(zigzag_encode_16b(data_in[i]));
            //     data_out[i] = zigzag_encode_16b(data_in[i]);
            //     if (i < 5) {
            //         printf("raw val, enc val: %d, %d\n", data_in[i], data_out[i]);
            //     }
            // }
            // for (len_t i = 0; i < length; i++) {
            //     *data_out++ = zigzag_encode_16b(*data_out++);
            // }
            zigzag_encode_u16( // TODO uncomment
                (const uint16_t*)inbuf, nelements, (int16_t*)outbuf);
            continue;
            // memcpy(outbuf, inbuf, nelements * sz);

            // printf("enc first 5 enc elems: ");
            // for (int i = 0; i < 5; i++) { printf("%d ", ((int16_t*)outbuf)[i]); } printf("\n");
        }

        // ------------------------ sprintz bitpacking
        if (sz == 2 && func == SPRINTZPACK) {
            // printf("enc initial size, nelements: %d, %d\n", size, nelements);
            nelements = sprintzpack_pack_u16_zigzag( // 2x to convert to bytes

            // size = 2 * dynamic_delta_pack_u16( // 2x to convert to bytes // works

                (const uint16_t*)inbuf, nelements, (int16_t*)outbuf);
            // nelements = (size + sz - 1) / sz;
            size = nelements * sz;
            // printf("enc new     size, nelements: %d, %d\n", size, nelements);
            continue;
        }
        if (sz == 2 && func == SPRINTZPACK_NOZIGZAG) {
            // printf("enc initial size, nelements: %d, %d\n", size, nelements);
            // nelements = sprintzpack_pack_u16( // 2x to convert to bytes
            nelements = sprintzpack_pack_u16( // 2x to convert to bytes
                (const uint16_t*)inbuf, nelements, (int16_t*)outbuf);
            size = nelements * sz;
            continue;
        }

        // ------------------------ byteshuffle and bitshuffle
        if (func == BITSHUFFLE) {
            // auto buff = std::
            std::vector<uint8_t> tmp;
            tmp.reserve(2 * size);
            bitshuffle(sz, size, inbuf, outbuf, tmp.data());
            // bitshuffle(const size_t bytesoftype, const size_t blocksize,
            //            const uint8_t* const _src, const uint8_t* _dest,
            //            const uint8_t* _tmp);
        }
        if (func == BYTESHUFFLE) {
            // printf("enc initial size, nelements: %d, %d\n", size, nelements);
            // shuffle(size, sz, inbuf, outbuf);
            shuffle(sz, size, inbuf, outbuf);
        }


        // printf("didn't apply any preproc for offset %lld...\n", offset);

#else
        if ((stride > 4)) { // TODO better err message saying need sprintz
            printf("WARNING: ignoring unrecognized preprocessor number '%lld'\n", preproc);
            continue;
        }
#endif

    #define DELTAS_FOR_OFFSET(OFFSET) \
        switch(sz) { \
        case 1: \
            { \
                auto in = (uint8_t*)inbuf; \
                auto out = (uint8_t*)outbuf; \
                for (int i = OFFSET; i < nelements; i++) { \
                    out[i] = in[i] - in[i-OFFSET]; \
                } \
            } break; \
        case 2: \
            { \
                auto in = (uint16_t*)inbuf; \
                auto out = (uint16_t*)outbuf; \
                for (int i = OFFSET; i < nelements; i++) { \
                    out[i] = in[i] - in[i-OFFSET]; \
                } \
            } break; \
        case 4: \
            { \
                auto in = (uint32_t*)inbuf; \
                auto out = (uint32_t*)outbuf; \
                for (int i = OFFSET; i < nelements; i++) { \
                    out[i] = in[i] - in[i-OFFSET]; \
                } \
            } break; \
        case 8: \
            { \
                auto in = (uint64_t*)inbuf; \
                auto out = (uint64_t*)outbuf; \
                for (int i = OFFSET; i < nelements; i++) { \
                    out[i] = in[i] - in[i-OFFSET]; \
                } \
            } break; \
        default: \
            printf("WARNING: ignoring invalid element size '%d'; must be in {1,2,4,8}\n", element_sz); \
        }

        if (func == DELTA) {
            memcpy(outbuf, inbuf, stride * sz);
            // tell compiler that offset is only going to be one of these 4
            // values (2x speedup or more)
            switch (stride) {
            case 1:
                DELTAS_FOR_OFFSET(1); break;
            case 2:
                DELTAS_FOR_OFFSET(2); break;
            case 3:
                DELTAS_FOR_OFFSET(3); break;
            case 4:
                DELTAS_FOR_OFFSET(4); break;
            default:
                printf("WARNING: ignoring invalid element size '%d'; must be in {1,2,4,8}\n", element_sz);
                // memcpy(outbuf, inbuf, size);
            }
        }
    }
    #undef DELTAS_FOR_OFFSET

    // if (false) { // TODO rm
    // if (outbuf != orig_outbuf) {
    // if (preprocessors.size() % 2 == 0) {
    if (nswaps % 2) {
        // printf("copying tmpbuf to output buffer!\n");
        // we switch off between using original inbuf and outbuf as input
        // and output when there are multiple preprocs; ensure that the
        // final result ends up in outbuf;
        memcpy(orig_outbuf, tmpbuf, nelements * sz);
        // auto tmp = inbuf;
        // inbuf = outbuf;
        // outbuf = tmp;
    }
    return size;
}

// void undo_preprocessors(const std::vector<int64_t>& preprocessors,
size_t undo_preprocessors(const std::vector<preproc_params_t>& preprocessors,
    uint8_t* orig_inbuf, size_t size, int element_sz, uint8_t* outbuf,
    uint8_t* tmpbuf)
{
    // printf("undo preprocs: got size = %d\n", size);
    // printf("undo preprocs: end of orig_inbuf:");//" \n", orig_inbuf[size - 1]);
    // auto tail_len = 8;
    // // uint16_t* tmpptr = ((uint16_t*)inbuf);
    // for (int i = 0; i < tail_len; i++) {
    //     // printf("%d ", tmpptr[nelements - tail_len + i]);
    //     // printf("%d ", orig_inbuf[size - tail_len + i + 1]);
    //     printf("%d ", orig_inbuf[size - tail_len + i]);
    //     // printf("%d ", inbuf[size - tail_len + i]);
    // }
    // printf("\n");

    // printf("undo preprocs: end of orig_inbuf: %d\n", orig_inbuf[size - 1]);
    // printf("using %lu preprocessors; size=%lu, element_sz=%d\n", preprocessors.size(), size, element_sz);
    if (preprocessors.size() < 1) {
        memcpy(outbuf, orig_inbuf, size);
        return size;
    }
    assert(orig_inbuf != outbuf);
    assert(tmpbuf != outbuf);

    int sz = element_sz;
    if (sz < 1) {
        sz = 1;
    }
    int64_t nelements = (size + sz - 1) / sz;

    // printf("size=%lu, sz=%lu, element_sz=%d\n", size, sz, element_sz);
    // printf("size=%lu, element_sz=%lu, nelements=%lld\n", size, sz, element_sz, nelements);

    auto orig_outbuf = outbuf;
    memcpy(tmpbuf, orig_inbuf, size);
    auto inbuf = (const uint8_t*)tmpbuf;

    // printf("undo preprocs: got size = %d\n", size);
    // printf("undo preprocs: end of orig_inbuf:");//" \n", orig_inbuf[size - 1]);
    // auto tail_len = 8;
    // uint16_t* tmpptr = ((uint16_t*)inbuf);
    // for (int i = 0; i < tail_len; i++) {
    //     printf("%d ", tmpptr[nelements - tail_len + i]);
    //     // printf("%d ", inbuf[size - tail_len + i]);
    // }
    // printf("\n");

    // printf("dec sees initial encoded elems: "); dump_elements((uint16_t*)orig_inbuf, 8);

    // for (auto preproc : preprocessors) {
    bool needs_ptr_swap = false;
    int nswaps = 0;
    for (size_t i = 0; i < preprocessors.size(); i++) {
        // traverse preprocessors in reverse order
        auto preproc = preprocessors[preprocessors.size() - 1 - i];
        // printf("undoing preproc: %lld with nelements=%lld, element_sz=%d\n", preproc, nelements, sz);

        auto func = preproc.func;
        if (!is_func_valid(func)) {
            continue;
        }

        if (needs_ptr_swap) { // swap buffers
            // printf("swapping input and output buffers\n");
            auto tmp = inbuf;
            inbuf = (const uint8_t*)outbuf;
            outbuf = (uint8_t*)tmp;
            // tmpbuf = outbuf;
            // outbuf = tmpbuf;
            nswaps++;
        }
        needs_ptr_swap = true; // execute each time but the first
        // needs_ptr_swap = false; // TODO rm

        // memcpy(outbuf, inbuf, size); continue;  // TODO rm

        // int offset = preproc;  // simplifying hack based on enum values
        int stride = preproc.stride;  // simplifying hack based on enum values
#ifndef BENCH_REMOVE_FASTPFOR   // use simd delta if available
        // if (sz == 4 && preproc == DELTA4)  {
        if (sz == 4 && func == DELTA && stride == 4)  {
            memcpy(outbuf, inbuf, size);
            FastPForLib::Delta::inverseDeltaSIMD((uint32_t*)outbuf, nelements);
            continue;
        }
#endif
#ifndef BENCH_REMOVE_SPRINTZ   // use simd delta if available
        // ------------------------ delta
        if (sz == 1 && (stride > 2) && func == DELTA) {
            if (inbuf != outbuf) {
                decode_delta_rowmajor_8b((int8_t*)inbuf, nelements, outbuf, stride);
            } else {
                decode_delta_rowmajor_inplace_8b((uint8_t*)inbuf, nelements, stride);
            }
            continue;
        }
        if (sz == 2 && (stride > 2) && func == DELTA) {
            if (inbuf != outbuf) {
                decode_delta_rowmajor_16b((const int16_t*)inbuf, nelements,
                    (uint16_t*)outbuf, stride);
            } else {
                // printf("dec delta: %d elements\n", nelements);
                decode_delta_rowmajor_inplace_16b((uint16_t*)inbuf,
                    nelements, stride);
            }
            continue;
        }
        // ------------------------ double delta
        if (sz == 1 && func == DOUBLE_DELTA) {
            if (inbuf != outbuf) {
                // printf("inbuf != outbuf!\n");
                decode_delta_rowmajor_8b((int8_t*)inbuf, nelements, outbuf, stride);
                // printf("ran dbl delta decoding without crashing!\n");
            } else {
                // printf("inbuf == outbuf! WTF\n");
                decode_delta_rowmajor_inplace_8b((uint8_t*)inbuf, nelements, stride);
                // printf("ran dbl delta decoding without crashing!\n");
            }
            continue;
        }
        if (sz == 2 && func == DOUBLE_DELTA) {
            if (inbuf != outbuf) {
                decode_doubledelta_rowmajor_16b((const int16_t*)inbuf, nelements,
                    (uint16_t*)outbuf, stride);
            } else {
                decode_doubledelta_rowmajor_inplace_16b((uint16_t*)inbuf,
                    nelements, stride);
            }
            continue;
        }
        // ------------------------ xff
        if (sz == 1 && func == XFF) {
            if (inbuf != outbuf) {
                decode_xff_rowmajor_8b((int8_t*)inbuf, nelements, outbuf, stride);
            } else {
                decode_xff_rowmajor_inplace_8b((uint8_t*)inbuf, nelements, stride);
            }
            continue;
        }
        if (sz == 2 && func == XFF) {
            if (inbuf != outbuf) {
                decode_xff_rowmajor_16b((const int16_t*)inbuf, nelements,
                    (uint16_t*)outbuf, stride);
            } else {
                decode_xff_rowmajor_inplace_16b((uint16_t*)inbuf,
                    nelements, stride);
                // printf("...ran xff decode\n");
            }
            continue;
        }

        // ------------------------ dynamic delta
        if (sz == 2 && (func == DYNAMIC_DELTA || func == DYNAMIC_DELTA_ALT)) {
            // printf("initial size, nelements: %d, %d\n", size, nelements);
            size = 2 * dynamic_delta_unpack_u16( // 2x to convert to bytes
                (const int16_t*)inbuf, (uint16_t*)outbuf);
            nelements = (size + sz - 1) / sz;
            // printf("new     size, nelements: %d, %d\n", size, nelements);
            continue;
        }

        // ------------------------ zigzag
        if (sz == 2 && func == ZIGZAG) {
            // printf("does inbuf == outbuf? %d\n", inbuf == outbuf);
            // // printf("dec buff has length in elements: %d\n", nelements);
            // printf("dec buff has size in bytes: %d\n", size);
            // printf("dec first 5 enc elems: ");
            // for (int i = 0; i < 5; i++) { printf("%d ", ((int16_t*)inbuf)[i]); } printf("\n");

            // memcpy(outbuf, inbuf, nelements * sz);
            zigzag_decode_u16(
                (const int16_t*)inbuf, nelements, (uint16_t*)outbuf);
            continue;
            // // printf("dec of ")
            // auto data_in = (const uint16_t*)inbuf;
            // auto data_out = (int16_t*)outbuf;
            // for (len_t i = 0; i < nelements; i++) {
            //     auto enc_val = data_in[i];
            //     data_out[i] = zigzag_decode_16b(data_in[i]);
            //     // if (i < 5) {
            //     //     printf("raw val, enc val: %d, %d\n", data_out[i], enc_val);
            //     // }
            // }
        }

        // ------------------------ sprintz bitpacking
        if (sz == 2 && func == SPRINTZPACK) {
            assert(outbuf != inbuf);  // sprintpack can't run inplace
            // printf("dec initial size, nelements: %d, %d\n", size, nelements);
            nelements = sprintzpack_unpack_u16_zigzag( // 2x to convert to bytes

            // size = 2 * dynamic_delta_unpack_u16( // 2x to convert to bytes // works

                (const int16_t*)inbuf, (uint16_t*)outbuf);
            size = nelements * sz;
            // nelements = (size + sz - 1) / sz;
            // printf("dec new     size, nelements: %d, %d\n", size, nelements);
            continue;
        }
        if (sz == 2 && func == SPRINTZPACK_NOZIGZAG) {
            assert(outbuf != inbuf);  // sprintpack can't run inplace
            // nelements = sprintzpack_unpack_u16( // 2x to convert to bytes
            nelements = sprintzpack_unpack_u16( // 2x to convert to bytes
                (const int16_t*)inbuf, (uint16_t*)outbuf);
            size = nelements * sz;
            continue;
        }

        // ------------------------ byteshuffle and bitshuffle
        if (func == BITSHUFFLE) {
            // auto buff = std::
            std::vector<uint8_t> tmp;
            tmp.reserve(2 * size);
            bitunshuffle(sz, size, inbuf, outbuf, tmp.data());
            // bitshuffle(const size_t bytesoftype, const size_t blocksize,
            //            const uint8_t* const _src, const uint8_t* _dest,
            //            const uint8_t* _tmp);
        }
        if (func == BYTESHUFFLE) {
            // printf("dec initial size, nelements: %d, %d\n", size, nelements);
            // unshuffle(size, sz, inbuf, outbuf);
            unshuffle(sz, size, inbuf, outbuf);
        }

#else
        if ((stride > 4)) { // TODO better err message saying need sprintz
            printf("WARNING: ignoring unrecognized preprocessor number '%lld'\n", preproc);
            continue;
        }
#endif



#define UNDO_DELTA_FOR_OFFSET(OFFSET) \
        if (sz <= 1) { \
            auto in = (uint8_t*)inbuf; \
            auto out = (uint8_t*)outbuf; \
            for (int i = OFFSET; i < nelements; i++) { \
                out[i] = in[i] + out[i-OFFSET]; \
            } \
        } else if (sz == 2) { \
            auto in = (uint16_t*)inbuf; \
            auto out = (uint16_t*)outbuf; \
            for (int i = OFFSET; i < nelements; i++) { \
                out[i] = in[i] + out[i-OFFSET]; \
            } \
        } else if (sz == 4) { \
            auto in = (uint32_t*)inbuf; \
            auto out = (uint32_t*)outbuf; \
            for (int i = OFFSET; i < nelements; i++) { \
                out[i] = in[i] + out[i-OFFSET]; \
            } \
        } else if (sz == 8) { \
            auto in = (uint64_t*)inbuf; \
            auto out = (uint64_t*)outbuf; \
            for (int i = OFFSET; i < nelements; i++) { \
                out[i] = in[i] + out[i-OFFSET]; \
            } \
        } else { \
            printf("WARNING: ignoring invalid element size '%d'; must be in {1,2,4,8}\n", element_sz); \
        }

        // compiler doesn't know that offset is only going to be one of these
        // 4 values, and so produces 2-3x slower code if we don't this
        if (func == DELTA) {
            memcpy(outbuf, inbuf, stride * sz);
            switch (stride) {
            case 1:
                UNDO_DELTA_FOR_OFFSET(1); break;
            case 2:
                UNDO_DELTA_FOR_OFFSET(2); break;
            case 3:
                UNDO_DELTA_FOR_OFFSET(3); break;
            case 4:
                UNDO_DELTA_FOR_OFFSET(4); break;
            default:
                break; // we checked that offset was in {1,..,4} above
            }
        }
#undef UNDO_DELTA_FOR_OFFSET
    }
    if (nswaps % 2) {
        // we switch off between using original inbuf and outbuf as input
        // and output when there are multiple preprocs; ensure that the
        // final result ends up in outbuf;
        memcpy(orig_outbuf, tmpbuf, nelements * sz);
    }
    return size;
}

} // namespace lzbench
