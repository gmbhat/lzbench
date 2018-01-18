
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
QueryResult sliding_min(const QueryParams& q, const DataInfo& di,
    const DataT* buff)
{
    return QueryResult{}; // TODO
}

template<class DataT>
QueryResult sliding_max(const QueryParams& q, const DataInfo& di,
    const DataT* buff)
{
    return QueryResult{}; // TODO
}

template<class DataT>
QueryResult sliding_l2(const QueryParams& q, const DataInfo& di,
    const DataT* buff)
{
    if (q.reduction == REDUCE_NONE) {

    } else if (q.reduction == REDUCE_THRESH) {

    } else if (q.reduction == REDUCE_TOP_K) {

    } else {
        printf("Invalid reduction %d for L2 query!\n", (int)q.reduction);
        exit(1);
    }
    return QueryResult{}; // TODO
}

template<class DataT>
QueryResult sliding_dot(const QueryParams& q, const DataInfo& di,
    const DataT* buff)
{
    if (q.reduction == REDUCE_NONE) {

    } else if (q.reduction == REDUCE_THRESH) {

    } else if (q.reduction == REDUCE_TOP_K) {

    } else {
        printf("Invalid reduction %d for dot product query!\n",
            (int)q.reduction);
        exit(1);
    }
    return QueryResult{}; // TODO
}

template<class DataT>
QueryResult corr(const QueryParams& q, const DataInfo& di,
    const DataT* buff)
{

}

template<class DataT>
QueryResult run_query(const QueryParams& q, const DataInfo& di, const DataT* buff) {

    // printf("actually running run_query!\n");

    QueryResult ret;
    switch (q.type) {
        case QUERY_MEAN: ret = sliding_mean(q, di, buff); break;
        case QUERY_MIN: ret = sliding_min(q, di, buff); break;
        case QUERY_MAX: ret = sliding_max(q, di, buff); break;
        case QUERY_L2: ret = sliding_l2(q, di, buff); break;
        case QUERY_DOT: ret = sliding_dot(q, di, buff); break;
        default:
            printf("Invalid query type %d!\n", (int)q.type); exit(1);
    }

    // TODO check if query has a reduction here and do it in this one place
    // if so

    if (q.reduction == REDUCE_NONE) {

    } else if (q.reduction == REDUCE_THRESH) {

    } else if (q.reduction == REDUCE_TOP_K) {

    } else {
        printf("Unsuppored reduction %d!\n",
            (int)q.reduction);
        exit(1);
    }
    return ret;
}

#endif // QUERY_HPP
