// TODO should rename this file to avoid ambiguity with query.hpp

#ifndef QUERY_H
#define QUERY_H

#include <vector>

enum query_type_e { QUERY_NONE = 0, QUERY_MEAN = 1, QUERY_MIN = 2,
    QUERY_MAX = 3, QUERY_L2 = 4, QUERY_DOT = 5 };
enum query_reduction_e { REDUCE_NONE = 0, REDUCE_THRESH = 1, REDUCE_TOP_K = 2};
enum storage_order_e { ROWMAJOR = 0, COLMAJOR = 1};

// // traits for different types of queries
// template<int query_type>
// class query_type_traits { enum { needs_data_window = 0 }; }
// template<>
// class query_type_traits<QUERY_L2> { enum { needs_data_window = 1 }; }
// template<>
// class query_type_traits<QUERY_DOT> { enum { needs_data_window = 1 }; }

typedef struct QueryParams {
    // double version is populated when argv parsed for simplicity; exactly
    // one of the others should be populated
    std::vector<double> window_data_dbl;
    std::vector<int8_t> window_data_i8;
    std::vector<uint8_t> window_data_u8;
    std::vector<int16_t> window_data_i16;
    std::vector<uint16_t> window_data_u16;
    std::vector<uint16_t> which_cols; // TODO populate to enable sparse queries
    int64_t window_nrows;
    int64_t window_ncols;
    int64_t window_stride;
    uint16_t k; // used for topk queries
    query_type_e type;
    query_reduction_e reduction;
} QueryParams;

typedef struct QueryResult {
    std::vector<int64_t> idxs;
    std::vector<int8_t> vals_i8;
    std::vector<uint8_t> vals_u8;
    std::vector<int16_t> vals_i16;
    std::vector<uint16_t> vals_u16;
} QueryResult;

typedef struct DataInfo {
    size_t element_sz;
    size_t nrows; // TODO populate this in decomp func
    size_t ncols;
    bool is_signed;
    storage_order_e storage_order;
} DataInfo;

template <class data_t> struct DataTypeTraits {};
template <> struct DataTypeTraits<uint8_t> { using AccumulatorT = uint16_t; };
template <> struct DataTypeTraits<uint16_t> { using AccumulatorT = uint32_t; };

// pull out reference to appropriate vector of values
template<class DataT> struct QueryResultValsRef {};
template <> struct QueryResultValsRef<int8_t> {
    std::vector<int8_t>& operator()(QueryResult& qr) { return qr.vals_i8; }
};
template <> struct QueryResultValsRef<uint8_t> {
    std::vector<uint8_t>& operator()(QueryResult& qr) { return qr.vals_u8; }
};
template <> struct QueryResultValsRef<int16_t> {
    std::vector<int16_t>& operator()(QueryResult& qr) { return qr.vals_i16; }
};
template <> struct QueryResultValsRef<uint16_t> {
    std::vector<uint16_t>& operator()(QueryResult& qr) { return qr.vals_u16; }
};


#endif // QUERY_HPP
