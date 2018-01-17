
#ifndef QUERY_HPP
#define QUERY_HPP

#include "query_common.h"
#include "query_mean.hpp"

// if we weren't just benchmarking, would need to return something in all
// of these

// template<class DataT>
// void sliding_mean(const QueryParams& q, const DataInfo& di, const DataT* buff) {

// }

template<class DataT>
void sliding_min(const QueryParams& q, const DataInfo& di, const DataT* buff) {

}

template<class DataT>
void sliding_max(const QueryParams& q, const DataInfo& di, const DataT* buff) {

}

template<class DataT>
void sliding_l2(const QueryParams& q, const DataInfo& di, const DataT* buff) {
    if (q.reduction == REDUCE_NONE) {

    } else if (q.reduction == REDUCE_THRESH) {

    } else if (q.reduction == REDUCE_TOP_K) {

    } else {
        printf("Invalid reduction %d for L2 query!\n", (int)q.reduction);
        exit(1);
    }
}

template<class DataT>
void sliding_dot(const QueryParams& q, const DataInfo& di, const DataT* buff) {
    if (q.reduction == REDUCE_NONE) {

    } else if (q.reduction == REDUCE_THRESH) {

    } else if (q.reduction == REDUCE_TOP_K) {

    } else {
        printf("Invalid reduction %d for dot product query!\n",
            (int)q.reduction);
        exit(1);
    }
}

template<class DataT>
void run_query(const QueryParams& q, const DataInfo& di, const DataT* buff) {
    switch (q.type) {
        case QUERY_NONE: sliding_mean(q, di, buff); break;
        case QUERY_MIN: sliding_min(q, di, buff); break;
        case QUERY_MAX: sliding_max(q, di, buff); break;
        case QUERY_L2: sliding_l2(q, di, buff); break;
        case QUERY_DOT: sliding_dot(q, di, buff); break;
        default:
            printf("Invalid query type %d!\n", (int)q.type); exit(1);
    }
}

#endif // QUERY_HPP
