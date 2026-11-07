#ifndef CUDA_ROW_ACTIVITY_CUH
#define CUDA_ROW_ACTIVITY_CUH
/* Keep the original summation on short rows. Long rows are scanned once per
 * bound snapshot rather than once for each variable/row incidence. */
static constexpr int ROW_ACTIVITY_MIN = 32;
struct LongActivityRow {
    const int *ptr;
    __host__ __device__ bool operator()(int i) const {
        return ptr[i + 1] - ptr[i] >= ROW_ACTIVITY_MIN;
    }
};
struct RowActivity {
    double lower, upper;
    int lower_infinite, upper_infinite;
};
__global__ static void cache_row_activity(View a, const double *l, const double *u,
                                          const double *rl, const double *ru,
                                          RowActivity *out, const int *rows = nullptr,
                                          int count = -1) {
    __shared__ double lower[256], upper[256];
    __shared__ int lower_inf[256], upper_inf[256];
    int lane = threadIdx.x;
    size_t size = count < 0 ? (size_t)a.rows : (size_t)count;
    for (size_t slot = blockIdx.x; slot < size; slot += gridDim.x) {
        size_t i = rows ? rows[slot] : slot;
        if (a.ptr[i + 1] - a.ptr[i] < ROW_ACTIVITY_MIN ||
            (rl[i] == -INFINITY && ru[i] == INFINITY)) {
            if (!lane)
                out[i] = {0., 0., 0, 0};
            continue;
        }
        double lo = 0., hi = 0.;
        int li = 0, ui = 0;
        for (size_t k = (size_t)a.ptr[i] + lane; k < (size_t)a.ptr[i + 1]; k += blockDim.x) {
            int j = a.col[k]; double v = a.val[k];
            double low = mul_down(v, v > 0 ? l[j] : u[j]);
            double high = mul_up(v, v > 0 ? u[j] : l[j]);
            if (isinf(low)) ++li;
            else lo = add_down(lo, low);
            if (isinf(high)) ++ui;
            else hi = add_up(hi, high);
        }
        lower[lane] = lo; upper[lane] = hi;
        lower_inf[lane] = li; upper_inf[lane] = ui;
        __syncthreads();
        for (int step = 128; step; step /= 2) {
            if (lane < step) {
                lower[lane] = add_down(lower[lane], lower[lane + step]);
                upper[lane] = add_up(upper[lane], upper[lane + step]);
                lower_inf[lane] += lower_inf[lane + step];
                upper_inf[lane] += upper_inf[lane + step];
            }
            __syncthreads();
        }
        if (!lane)
            out[i] = {lower[0], upper[0], lower_inf[0], upper_inf[0]};
        __syncthreads();
    }
}
__device__ static void excluded_row_activity(const RowActivity &row, double a, double l,
                                              double u, double &lo, double &hi) {
    double own_lo = mul_down(a, a > 0 ? l : u);
    double own_hi = mul_up(a, a > 0 ? u : l);
    bool li = isinf(own_lo), ui = isinf(own_hi);
    /* row.lower <= sum of the rounded-down finite products. Subtracting the
     * SAME rounded product downward therefore bounds the remaining products;
     * no subtraction of an unbounded endpoint is performed. The upper proof
     * is symmetric. Overflow in the finite sum only widens the enclosure. */
    lo = row.lower_infinite > (int)li ? -INFINITY
         : li ? row.lower : add_down(row.lower, -own_lo);
    hi = row.upper_infinite > (int)ui ? INFINITY
         : ui ? row.upper : add_up(row.upper, -own_hi);
}
#endif
