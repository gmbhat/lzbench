

#include "lzbench.h"
#include "preprocessing.h"


#ifndef BENCH_REMOVE_FASTPFOR
    #include "fastpfor/deltautil.h"  // for simd delta preproc
#endif

void apply_preprocessors(const std::vector<int64_t>& preprocessors,
    const uint8_t* inbuf, size_t size, int element_sz, uint8_t* outbuf)
{
    // printf("using %lu preprocessors; size=%lu, element_sz=%d\n", preprocessors.size(), size, element_sz);

    if (preprocessors.size() < 1) { return; }

    int sz = element_sz;
    if (sz < 1) { sz = 1; }
    int64_t nelements = size / sz;

    // printf("size=%lu, sz=%lu, element_sz=%d\n", size, sz, element_sz);
    // printf("size=%lu, element_sz=%lu, nelements=%lld\n", size, sz, element_sz, nelements);


    for (auto preproc : preprocessors) {
        // printf("applying preproc: %lld with nelements=%lld, element_sz=%d\n", preproc, nelements, sz);
        // continue;

        if ((preproc < 1) || (preproc > 4)) {
            printf("WARNING: ignoring unrecognized preprocessor number '%lld'\n", preproc);
            continue;
        }

#ifndef BENCH_REMOVE_FASTPFOR   // use simd delta if available
        if (sz == 4 && preproc == DELTA4)  {
            memcpy(outbuf, inbuf, size);
            FastPForLib::Delta::deltaSIMD((uint32_t*)outbuf, nelements);
            continue;
        }
#endif
        int offset = preproc;  // simplifying hack based on enum values

        memcpy(outbuf, inbuf, offset * sz);


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

        // tell compiler that offset is only going to be one of these 4
        // values (2x speedup or more)
        switch (offset) {
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

void undo_preprocessors(const std::vector<int64_t>& preprocessors,
    uint8_t* inbuf, size_t size, int element_sz, uint8_t* outbuf)
{
    // printf("using %lu preprocessors; size=%lu, element_sz=%d\n", preprocessors.size(), size, element_sz);

    if (preprocessors.size() < 1) { return; }

    if (outbuf == nullptr) { outbuf = inbuf; }

    int sz = element_sz;
    if (sz < 1) {
        sz = 1;
    }
    int64_t nelements = size / sz;

    // printf("size=%lu, sz=%lu, element_sz=%d\n", size, sz, element_sz);
    // printf("size=%lu, element_sz=%lu, nelements=%lld\n", size, sz, element_sz, nelements);


    for (auto preproc : preprocessors) {
        // printf("applying preproc: %lld with nelements=%lld, element_sz=%d\n", preproc, nelements, sz);
        // continue;

        if ((preproc < 1) || (preproc > 4)) {
            printf("WARNING: ignoring unrecognized preprocessor number '%lld'\n", preproc);
            continue;
        }

        // memcpy(outbuf, inbuf, size); continue;  // TODO rm

#ifndef BENCH_REMOVE_FASTPFOR   // use simd delta if available
        if (sz == 4 && preproc == DELTA4)  {
            memcpy(outbuf, inbuf, size);
            FastPForLib::Delta::inverseDeltaSIMD((uint32_t*)outbuf, nelements);
            continue;
        }
#endif
        int offset = preproc;  // simplifying hack based on enum values

        memcpy(outbuf, inbuf, offset * sz);

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
        switch (offset) {
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

#undef UNDO_DELTA_FOR_OFFSET
    }
}
