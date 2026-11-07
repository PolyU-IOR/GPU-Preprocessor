#ifndef CUDA_TAPE_CUH
#define CUDA_TAPE_CUH
#include "cuda_common.cuh"
#include <memory>
#include <vector>

struct DeviceRecord {
    int type, j, k, row, retained;
    int qoffset, qsize, aoffset, asize, goffset, gsize;
    int reserved;
    double alpha, pivot, ratio, c, lower, upper, lower2, upper2;
    int m, n;
};
static_assert(sizeof(DeviceRecord) == 120 && offsetof(DeviceRecord, alpha) == 48,
              "Device recovery records must have initialized, explicit storage");
/* Segmented owning storage avoids doubling a multi-gigabyte tape during growth.
 * Segments are immutable after presolve and shared by independent workspaces. */
struct DeviceTapeSegment {
    int count = 0, entries = 0;
    bool folding = false; /* A complete stage; disjoint classes, replayed cooperatively. */
    Buffer<DeviceRecord> records;
    Buffer<int> indices;
    Buffer<double> values;
};
struct DeviceTape {
    int device, rows, cols, reduced_m = 0, reduced_n = 0;
    size_t count = 0, entries = 0, bytes = 0;
    Buffer<int> row_ids, col_ids;
    std::vector<std::unique_ptr<DeviceTapeSegment>> segments;
    DeviceTape(int d, int m, int n) : device(d), rows(m), cols(n) {}
};
using TapeOwner = std::shared_ptr<DeviceTape>;
#endif
