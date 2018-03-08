//
// parallel.cpp
// Created by D Blalock 2018-2-28
//

#include "parallel.h"

#include <iostream>  // TODO rm
#include <future>
#include <thread>

#include "preprocessing.h"
#include "query.hpp"
#include "util.h"

size_t _decomp_and_query(lzbench_params_t *params, const compressor_desc_t* desc,
    const uint8_t* comprbuff, size_t comprsize, uint8_t* outbuf, size_t outsize,
    bool already_materialized,
    size_t param1, size_t param2, char* workmem)
{
    // printf("decomp_and_query: running '%s' with insize %lu, outsize %u!\n", desc->name, comprsize, outsize);
    // bool already_materialized = strings_equal(desc->name, "materialized");

    compress_func decompress = desc->decompress;

    size_t dlen;
    if (!already_materialized) {
        // if (comprsize == outsize) { // uncompressed
        if (comprsize == outsize || true) { // TODO rm
            memcpy(outbuf, comprbuff, comprsize);
            dlen = comprsize;
        } else {
            dlen = decompress((char*)comprbuff, comprsize, (char*)outbuf,
                              outsize, param1, param2, workmem);
        }
        undo_preprocessors(params->preprocessors, outbuf, dlen,
            params->data_info.element_sz);

        // prevent compiler from not running above command
        if (params->verbose >= 999) {
            size_t cmn = common(comprbuff, outbuf, outsize);
            LZBENCH_PRINT(999, "ERROR in %s: only first %d / %d decompressed bytes were correct\n",
                desc->name, (int32_t)cmn, (int32_t)outsize);
        }
    }

    // run query if one is specified
    auto qparams = params->query_params;
    if (qparams.type != QUERY_NONE) {
        // printf("got query type: %d; about to run a query...\n", qparams.type);
        // auto& dinfo = params->data_info;
        DataInfo dinfo = params->data_info;
        if (dinfo.ncols < 1) {
            printf("ERROR: Must specify number of columns in data to run query!\n");
            exit(1);
        }
        dinfo.nrows = dlen / (dinfo.ncols * dinfo.element_sz);
        // printf("dinfo nrows, ncols, size: %lu, %lu, %lu\n",
        //     dinfo.nrows, dinfo.ncols, dinfo.nrows * dinfo.ncols);
        QueryResult result = run_query(
            params->query_params, dinfo, outbuf);
        // printf("ran query type: %d\n", qparams.type);
        // printf("number of idxs in result: %lu\n", result.idxs.size());

        // prevent compiler from optiming away query
        if (params->verbose > 999) {
            printf("query u8 result: ");
            for (auto val : result.vals_u8) { printf("%d ", (int)val); }
            printf("\n");
            printf("query u16 result: ");
            for (auto val : result.vals_u16) { printf("%d ", (int)val); }
            printf("\n");
        }
    }

    return dlen;
}


void parallel_decomp(lzbench_params_t *params,
    std::vector<size_t>& chunk_sizes, const compressor_desc_t* desc,
    std::vector<size_t> &compr_sizes, const uint8_t *inbuf, uint8_t *outbuf,
    uint8_t* tmpbuf, bench_rate_t rate, std::vector<uint64_t> comp_times,
    size_t param1, size_t param2, char* workmem)
{
    // printf("calling parallel decomp for algorithm (T=%d): %s!\n", params->nthreads, desc->name);
    printf("calling parallel decomp for algorithm %s!\n", desc->name);
    // if (params) {
    //     printf("using nthreads: %d\n", params->nthreads);
    // }

    std::vector<uint64_t> compressed_chunk_starts;
    compressed_chunk_starts.push_back(0);
    for (auto sz : compr_sizes) {
        compressed_chunk_starts.push_back(compressed_chunk_starts.back() + sz);
    }
    compressed_chunk_starts.pop_back(); // last one is just an end idx

    // printf("compr start idxs: ");
    // for (auto start_idx : compressed_chunk_starts) {
    //     printf("%lld, ", start_idx);
    // }
    // printf("\n");

    printf("param1, param2 = %lu, %lu\n", param1, param2);
    // if (param1 != 80) {
    //     printf("param1 is %lu, not 80!\n", param1);
    // }

    // timing stuff
    bench_timer_t t_start;
    GetTime(t_start);

    uint64_t run_for_nanosecs = (uint64_t)params->dmintime*1000*1000;

    int nthreads = params->nthreads;
    // std::vector<int64_t> total_scanned_sizes(nthreads);
    // std::vector<std::future<int64_t>> total_scanned_sizes(nthreads);
    std::vector<std::future<int64_t>> scanned_sizes_futures;
    std::vector<int64_t> total_scanned_sizes(nthreads);
    // std::vector<std::thread> threads(nthreads);

    auto max_chunk_sz = chunk_sizes[0];
    auto total_raw_sz = 0;
    for (auto sz : chunk_sizes) {
        if (sz > max_chunk_sz) { max_chunk_sz = sz; }
        total_raw_sz += sz;
    }

    bool already_materialized = strings_equal(desc->name, "materialized");

    // for (int i = 0; i < nthreads; i++) {
        // auto& this_total = total_scanned_sizes[i];
        // size_t* this_total = total_scanned_sizes[i];
        // threads[i] = std::thread([&total_scanned_sizes[i]] {
        auto run_in_thread =
            // [i, run_for_nanosecs, max_chunk_sz, total_raw_sz, inbuf, nthreads,
            [run_for_nanosecs, max_chunk_sz, total_raw_sz, inbuf, nthreads,
                params, desc, compr_sizes, chunk_sizes,
                t_start, rate, compressed_chunk_starts,
                already_materialized,
                // &total_scanned_sizes,
                param1, param2, workmem](int i) {

            // this_total = i;

            int64_t total_raw_sz = 0;

            bench_timer_t t_end;
            int64_t max_iters = run_for_nanosecs > 0 ? 1000*1000*1000 : 0;
            int64_t niters = 0;
            auto num_chunks = compressed_chunk_starts.size();
            // XXX this is an ugly way to check this



            // printf("max chunk sz: %lu\n", max_chunk_sz);
            uint8_t* decomp_buff = alloc_data_buffer(max_chunk_sz + 4096);

            do {
                // run multiple iters betwen rtsc calls to avoid sync overhead
                // use nthreads iters as a heuristic so syncs/sec is constant
                for (int it = 0; it < 10*nthreads; it++) {
                    auto chunk_idx = rand() % num_chunks;
                    // auto inptr = inbuf + compressed_chunk_starts[chunk_idx];
                    auto inptr = inbuf; // TODO uncomment above after debug
                    auto insize = compr_sizes[chunk_idx];
                    auto rawsize = chunk_sizes[chunk_idx];

                    _decomp_and_query(params, desc, inptr, insize,
                        decomp_buff, rawsize, already_materialized,
                        param1, param2, workmem);

                    // this_total += rawsize;
                    total_raw_sz += rawsize;
                    // total_scanned_sizes[i] += rawsize;
                    // *(this_total) = *(this_total) + rawsize;
                    niters++;
                }

                // check whether we're done
                GetTime(t_end);
                auto elapsed_nanos = GetDiffTime(rate, t_start, t_end);
                bool done = elapsed_nanos >= run_for_nanosecs;
                done = done || niters >= max_iters;
                if (done) {
                    LZBENCH_PRINT(0, "%d) elapsed iters, time: %lld, %lld/%lldns\n",
                        i, niters, elapsed_nanos, run_for_nanosecs);
                }
                if (done) { break; }
            } while (true);

            // auto total_comp_size = 0;
            // for (auto sz: compr_sizes) {
            //     total_comp_size += sz;
            // }
            // size_t cmn = common(inbuf, decomp_buff, total_raw_sz);
            // if (cmn < insize) {
            // printf("about to check whether decomp is correct...\n");
            // size_t cmn = common(inbuf, decomp_buff, max_chunk_sz);
            // if (cmn < max_chunk_sz) {
            //     LZBENCH_PRINT(999, "ERROR in %s: only first %d / %d decompressed bytes were correct\n",
            //     desc->name, (int32_t)cmn, (int32_t)max_chunk_sz);
            // }

            free_data_buffer(decomp_buff);

            return total_raw_sz;
        // });
        };
    // }

    // auto debug_lambda = [](int64_t i) { std::cout << i << "\n"; return i; };
    for (int i = 0; i < nthreads; i++) {
        scanned_sizes_futures.push_back(std::async(run_in_thread, i));
        // scanned_sizes_futures.push_back(std::async(debug_lambda, i));
    }

    // printf("about to try get()ing all the futures...\n");
    for (int i = 0; i < nthreads; i++) {
        total_scanned_sizes[i] = scanned_sizes_futures[i].get();
        // scanned_sizes_futures[i].get();
    }

    // for (auto& t : threads) {
    //     t.join();
    // }

    printf("total sizes: ");
    for (auto el : total_scanned_sizes) {
        printf("%lld, ", el);
    }
    printf("\n");

    // compute total amount of data all the threads got through
    int64_t total_scanned_bytes = 0;
    for (auto sz : total_scanned_sizes) {
        total_scanned_bytes += sz;
    }

    if (!run_for_nanosecs) { // this case shouldn't be used for real results
        bench_timer_t t_end;
        GetTime(t_end);
        // printf("WARNING: minimum run time not specified\n");
        run_for_nanosecs = GetDiffTime(rate, t_start, t_end);
    }
    auto run_for_usecs = run_for_nanosecs / 1000;
    auto thruput_MB_per_sec = total_scanned_bytes / run_for_usecs;
    printf(">> \1%s avg thruput: %lld\n", desc->name, thruput_MB_per_sec);
    printf("------------------------");
}



