#ifndef CUDA_FOLDING_CUH
#define CUDA_FOLDING_CUH
#include <thrust/remove.h>
#include <thrust/sequence.h>

/* Fixed-buffer signature products screen equitable partitions. Complete sparse
 * block-sum vectors certify every accepted candidate before model/tape commit.
 * All buffers stay on device; only counts and status cross the host boundary. */
namespace qp_folding {
struct Profile {
    bool enabled = getenv("GPUPRESOLVE_FOLD_PROFILE") != nullptr;
    const char *reason = "unchanged";
    int m, n, rows = 0, cols = 0, rounds = 0;
    double seconds[6] = {}, started = monotonic_seconds();
    Profile(int m, int n) : m(m), n(n) {}
    template <class F> Status timed(int phase, F f) {
        if (!enabled)
            return f();
        cudaStreamSynchronize(0);
        double start = monotonic_seconds();
        Status st = f();
        cudaStreamSynchronize(0);
        seconds[phase] += monotonic_seconds() - start;
        return st;
    }
    void counts(int r, int c) {
        rows = r;
        cols = c;
        if (enabled && (rounds < 8 || rounds % 128 == 0))
            fprintf(stderr, "FOLD_PROFILE round=%d rows=%d/%d cols=%d/%d\n", rounds, r, m, c, n);
    }
    ~Profile() {
        if (enabled)
            fprintf(stderr,
                    "FOLD_PROFILE final rounds=%d rows=%d/%d cols=%d/%d reason=%s "
                    "init=%.6f column_product=%.6f certificate=%.6f columns=%.6f row_product=%.6f "
                    "rows=%.6f total=%.6f\n",
                    rounds, rows, m, cols, n, reason, seconds[0], seconds[1], seconds[2],
                    seconds[3], seconds[4], seconds[5], monotonic_seconds() - started);
    }
};
using Key = unsigned long long;
struct EntryValue {
    Key key;
    double value;
};
struct Block {
    Key key;
    double hi, lo;
};
struct EntryLess {
    __host__ __device__ bool operator()(const EntryValue &a, const EntryValue &b) const {
        return a.key < b.key || (a.key == b.key && a.value < b.value);
    }
};
struct IsZero {
    __host__ __device__ bool operator()(const Block &b) const {
        return b.hi == 0. && b.lo == 0.;
    }
};
__device__ inline void two_sum(double a, double b, double &s, double &e) {
    s = __dadd_rn(a, b);
    double v = __dsub_rn(s, a);
    e = __dadd_rn(__dsub_rn(a, __dsub_rn(s, v)), __dsub_rn(b, v));
}
/* Checked two-component expansion. An unrepresentable third component or
 * intermediate overflow declines this optional stage before any mutation.
 * Thus equal accepted vectors have equal exact sums of the binary64 inputs;
 * a large term cannot hide a small asymmetric coefficient. */
struct ExactSum {
    double hi = 0., lo = 0.;
    __device__ void add(double v, int *error) {
        double s, e, t, tail;
        two_sum(hi, v, s, e);
        two_sum(lo, e, t, tail);
        two_sum(s, t, hi, lo);
        if (tail != 0. || !isfinite(hi) || !isfinite(lo)) {
            atomicExch(error, (int)NUMERICAL_ERROR);
            hi = lo = 0.; /* Keep subsequent optional sorting well ordered. */
        }
        if (hi == 0.)
            hi = 0.;
        if (lo == 0.)
            lo = 0.;
    }
};
__global__ static void make_entries(View a, const int *rows, const int *cols, EntryValue *e) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.rows)
        for (int k = a.ptr[i]; k < a.ptr[i + 1]; ++k)
            e[k] = EntryValue{
                ((Key)(unsigned)(rows ? rows[i] : i) << 32) | (unsigned)cols[a.col[k]], a.val[k]};
}
struct IsStart {
    const EntryValue *e;
    __device__ bool operator()(int k) const {
        return !k || e[k].key != e[k - 1].key;
    }
};
__global__ static void sum_blocks(int groups, int nnz, const int *start, const EntryValue *e,
                                  Block *b, int *error) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < groups) {
        int begin = start[g], end = g + 1 < groups ? start[g + 1] : nnz;
        ExactSum sum;
        for (int k = begin; k < end; ++k)
            sum.add(e[k].value, error);
        b[g] = Block{e[begin].key, sum.hi, sum.lo};
    }
}
__global__ static void count_rows(int nnz, const Block *b, int *ptr) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz)
        atomicAdd(ptr + (unsigned)(b[k].key >> 32) + 1, 1);
}
__device__ inline Key mix(Key x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct Sums {
    int nnz = 0;
    Buffer<Block> data;
    Buffer<int> ptr;
    Buffer<Key> hash;
};
__global__ static void hash_rows(int n, const int *ptr, const Block *b, Key *hash) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Key h = 0;
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
            h = mix(h ^ (unsigned)b[k].key);
            h = mix(h ^ (Key)__double_as_longlong(b[k].hi));
            h = mix(h ^ (Key)__double_as_longlong(b[k].lo));
        }
        hash[i] = h;
    }
}
static Status block_sums(View a, int nnz, const int *rows, const int *cols, int nr, Sums &out,
                         int *error) {
    Buffer<EntryValue> entries;
    Buffer<int> starts;
    GPU_CHECK(out.ptr.reserve((size_t)nr + 1));
    GPU_CHECK(out.hash.reserve((size_t)nr + 1));
    GPU_CHECK(out.data.reserve((size_t)nnz + 1));
    GPU_CHECK(cudaMemset(out.ptr.data, 0, ((size_t)nr + 1) * sizeof(int)));
    if (nnz) {
        GPU_CHECK(entries.reserve(nnz));
        GPU_CHECK(starts.reserve(nnz));
        make_entries<<<((unsigned)a.rows + 255) / 256, 256>>>(a, rows, cols, entries.data);
        thrust::sort(thrust::device, entries.data, entries.data + nnz, EntryLess{});
        auto first = thrust::make_counting_iterator(0);
        int groups = (int)(thrust::copy_if(thrust::device, first, first + nnz, starts.data,
                                           IsStart{entries.data}) -
                           starts.data);
        sum_blocks<<<((unsigned)groups + 255) / 256, 256>>>(groups, nnz, starts.data, entries.data,
                                                            out.data.data, error);
        out.nnz = (int)(thrust::remove_if(thrust::device, out.data.data, out.data.data + groups,
                                          IsZero{}) -
                        out.data.data);
        if (out.nnz)
            count_rows<<<((unsigned)out.nnz + 255) / 256, 256>>>(out.nnz, out.data.data,
                                                                 out.ptr.data);
    }
    thrust::inclusive_scan(thrust::device, out.ptr.data, out.ptr.data + nr + 1, out.ptr.data);
    if (nr)
        hash_rows<<<((unsigned)nr + 255) / 256, 256>>>(nr, out.ptr.data, out.data.data,
                                                       out.hash.data);
    GPU_CHECK(cudaGetLastError());
    return OK;
}
struct SumView {
    const int *ptr = nullptr;
    const Block *data = nullptr;
    const Key *hash = nullptr;
    __device__ int compare(int i, int j) const {
        if (!ptr)
            return 0;
        if (hash[i] != hash[j])
            return hash[i] < hash[j] ? -1 : 1;
        int x = ptr[i], xe = ptr[i + 1], y = ptr[j], ye = ptr[j + 1];
        for (; x < xe && y < ye; ++x, ++y) {
            unsigned a = (unsigned)data[x].key, b = (unsigned)data[y].key;
            if (a != b)
                return a < b ? -1 : 1;
            if (data[x].hi != data[y].hi)
                return data[x].hi < data[y].hi ? -1 : 1;
            if (data[x].lo != data[y].lo)
                return data[x].lo < data[y].lo ? -1 : 1;
        }
        return (x < xe) - (y < ye);
    }
};
static SumView sum_view(const Sums &s) {
    return {s.ptr.data, s.data.data, s.hash.data};
}
struct Part {
    int count = 0;
    Buffer<int> label, order, start, flags;
    Buffer<Key> keys;
};
struct InitialLess {
    const double *l, *u, *c;
    __device__ bool operator()(int i, int j) const {
        if (c && c[i] != c[j])
            return c[i] < c[j];
        if (l[i] != l[j])
            return l[i] < l[j];
        return u[i] < u[j];
    }
};
template <class Less>
__global__ static void mark_parts(int n, const int *order, int *flags, Less less) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n)
        flags[k] = !k || less(order[k - 1], order[k]) || less(order[k], order[k - 1]);
}
__global__ static void assign_parts(int n, const int *order, const int *flags, int *label,
                                    int *start, int *count = nullptr) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        label[order[k]] = flags[k] - 1;
        if (!k || flags[k] != flags[k - 1])
            start[flags[k] - 1] = k;
        if (k + 1 == n) {
            start[flags[k]] = n;
            if (count)
                *count = flags[k];
        }
    }
}
template <class Less> static Status partition(int n, Less less, Part &out) {
    GPU_CHECK(out.order.reserve((size_t)n + 1));
    GPU_CHECK(out.label.reserve((size_t)n + 1));
    GPU_CHECK(out.start.reserve((size_t)n + 1));
    GPU_CHECK(out.flags.reserve(n));
    thrust::sequence(thrust::device, out.order.data, out.order.data + n);
    thrust::stable_sort(thrust::device, out.order.data, out.order.data + n, less);
    mark_parts<<<((unsigned)n + 255) / 256, 256>>>(n, out.order.data, out.flags.data, less);
    thrust::inclusive_scan(thrust::device, out.flags.data, out.flags.data + n, out.flags.data);
    GPU_CHECK(cuda_memcpy(&out.count, out.flags.data + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    assign_parts<<<((unsigned)n + 255) / 256, 256>>>(n, out.order.data, out.flags.data,
                                                     out.label.data, out.start.data);
    GPU_CHECK(cudaGetLastError());
    return OK;
}
static void swap_part(Part &a, Part &b) {
    std::swap(a.count, b.count);
    a.label.swap(b.label);
    a.order.swap(b.order);
    a.start.swap(b.start);
    a.flags.swap(b.flags);
    a.keys.swap(b.keys);
}
// Field arithmetic makes projected sums independent of floating-point order.
// The prime 2^31-1 admits every binary64 power of two, including subnormals.
// Two independently salted projections screen candidates; the exact certificate
// below, rather than a probabilistic assumption, authorizes each transformation.
constexpr unsigned field_prime = 2147483647u;
__device__ inline unsigned field_reduce(Key x) {
    x = (x & field_prime) + (x >> 31);
    x = (x & field_prime) + (x >> 31);
    return (unsigned)(x >= field_prime ? x - field_prime : x);
}
__device__ inline unsigned field_coefficient(double value) {
    Key bits = (Key)__double_as_longlong(value);
    unsigned exponent = (unsigned)((bits >> 52) & 2047);
    Key mantissa = bits & 0xfffffffffffffULL;
    if (exponent)
        mantissa |= 1ULL << 52;
    int power = (exponent ? (int)exponent - 1075 : -1074) % 31;
    if (power < 0)
        power += 31;
    unsigned v = field_reduce((Key)field_reduce(mantissa) << power);
    return (bits >> 63) && v ? field_prime - v : v;
}
__global__ static void encode_coefficients(int n, const double *value, unsigned *out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n)
        out[k] = field_coefficient(value[k]);
}
__device__ inline unsigned weight(int label, Key salt) {
    return field_reduce(mix((unsigned)label ^ salt));
}
__device__ static void project(View a, const unsigned *coefficient, const int *labels, int i,
                               Key salt, unsigned &lo, unsigned &hi) {
    int lane = threadIdx.x & 31;
    for (int k = a.ptr[i] + lane; k < a.ptr[i + 1]; k += 32) {
        int label = labels[a.col[k]];
        lo = field_reduce((Key)lo + field_reduce((Key)coefficient[k] * weight(label, salt)));
        hi = field_reduce((Key)hi + field_reduce((Key)coefficient[k] *
                                                 weight(label, salt ^ 0x9e3779b97f4a7c15ULL)));
    }
}
__global__ static void project_columns(View at, View q, const unsigned *ac, const unsigned *qc,
                                       const int *rows, const int *cols, int round, Key *out) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (i >= at.rows)
        return;
    unsigned a = 0, b = 0, c = 0, d = 0;
    Key salt = mix((Key)round + 0x632be59bd9b4e019ULL);
    project(at, ac, rows, i, salt, a, b);
    project(q, qc, cols, i, salt ^ 0xd1b54a32d192ed03ULL, c, d);
    a = field_reduce((Key)a + c);
    b = field_reduce((Key)b + d);
    for (int offset = 16; offset; offset /= 2) {
        a = field_reduce((Key)a + __shfl_down_sync(0xffffffffu, a, offset));
        b = field_reduce((Key)b + __shfl_down_sync(0xffffffffu, b, offset));
    }
    if (!(threadIdx.x & 31))
        out[i] = ((Key)a << 32) | b;
}
__global__ static void project_rows(View a, const unsigned *ac, const int *cols, int round,
                                    Key *out) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (i >= a.rows)
        return;
    unsigned lo = 0, hi = 0;
    project(a, ac, cols, i, mix((Key)round + 0x94d049bb133111ebULL), lo, hi);
    for (int offset = 16; offset; offset /= 2) {
        lo = field_reduce((Key)lo + __shfl_down_sync(0xffffffffu, lo, offset));
        hi = field_reduce((Key)hi + __shfl_down_sync(0xffffffffu, hi, offset));
    }
    if (!(threadIdx.x & 31))
        out[i] = ((Key)lo << 32) | hi;
}
struct Signatures {
    Buffer<unsigned> a, at, q;
    Buffer<Key> row, col;
    Status prepare(CudaPlan *p) {
        GPU_CHECK(a.reserve((size_t)p->annz + 1));
        GPU_CHECK(at.reserve((size_t)p->annz + 1));
        GPU_CHECK(q.reserve((size_t)p->qnnz + 1));
        GPU_CHECK(row.reserve(p->m));
        GPU_CHECK(col.reserve(p->n));
        if (p->annz) {
            encode_coefficients<<<((unsigned)p->annz + 255) / 256, 256>>>(p->annz, p->A.val.data,
                                                                          a.data);
            encode_coefficients<<<((unsigned)p->annz + 255) / 256, 256>>>(p->annz, p->AT.val.data,
                                                                          at.data);
        }
        if (p->qnnz)
            encode_coefficients<<<((unsigned)p->qnnz + 255) / 256, 256>>>(p->qnnz, p->Q.val.data,
                                                                          q.data);
        GPU_CHECK(cudaGetLastError());
        return OK;
    }
    Status columns(CudaPlan *p, const Part &r, const Part &c, int round) {
        project_columns<<<((size_t)p->n + 7) / 8, 256>>>(
            view(p->AT, p->n, p->m), view(p->Q, p->n, p->n), at.data, q.data, r.label.data,
            c.label.data, round, col.data);
        GPU_CHECK(cudaGetLastError());
        return OK;
    }
    Status rows(CudaPlan *p, const Part &c, int round) {
        project_rows<<<((size_t)p->m + 7) / 8, 256>>>(view(p->A, p->m, p->n), a.data, c.label.data,
                                                      round, row.data);
        GPU_CHECK(cudaGetLastError());
        return OK;
    }
};
struct SignatureLess {
    const int *old;
    const Key *signature;
    __device__ bool operator()(int i, int j) const {
        return old[i] != old[j] ? old[i] < old[j] : signature[i] < signature[j];
    }
};
__global__ static void signature_keys(int n, const Key *signature, int *order, Key *keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        order[i] = i;
        keys[i] = signature[i];
    }
}
__global__ static void label_keys(int n, const int *old, const int *order, int *keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        keys[i] = old[order[i]];
}
static Status partition(int n, SignatureLess less, Part &out, int *device_count) {
    GPU_CHECK(out.order.reserve((size_t)n + 1));
    GPU_CHECK(out.label.reserve((size_t)n + 1));
    GPU_CHECK(out.start.reserve((size_t)n + 1));
    GPU_CHECK(out.flags.reserve(n));
    GPU_CHECK(out.keys.reserve(n));
    signature_keys<<<((unsigned)n + 255) / 256, 256>>>(n, less.signature, out.order.data,
                                                       out.keys.data);
    thrust::stable_sort_by_key(thrust::device, out.keys.data, out.keys.data + n, out.order.data);
    label_keys<<<((unsigned)n + 255) / 256, 256>>>(n, less.old, out.order.data, out.flags.data);
    thrust::stable_sort_by_key(thrust::device, out.flags.data, out.flags.data + n, out.order.data);
    mark_parts<<<((unsigned)n + 255) / 256, 256>>>(n, out.order.data, out.flags.data, less);
    thrust::inclusive_scan(thrust::device, out.flags.data, out.flags.data + n, out.flags.data);
    assign_parts<<<((unsigned)n + 255) / 256, 256>>>(n, out.order.data, out.flags.data,
                                                     out.label.data, out.start.data, device_count);
    GPU_CHECK(cudaGetLastError());
    return OK;
}
__global__ static void certify_classes(int groups, const int *order, const int *start, SumView a,
                                       SumView q, int *error) {
    int g = blockIdx.x;
    if (g < groups) {
        int begin = start[g], end = start[g + 1], representative = order[begin];
        for (int k = begin + 1 + threadIdx.x; k < end; k += blockDim.x)
            if (a.compare(representative, order[k]) || q.compare(representative, order[k]))
                atomicExch(error, (int)NUMERICAL_ERROR);
    }
}
static Status certify(CudaPlan *p, const Part &rows, const Part &cols, int *error) {
    Sums at, q, a;
    Status st =
        block_sums(view(p->AT, p->n, p->m), p->annz, nullptr, rows.label.data, p->n, at, error);
    if (st)
        return st;
    st = block_sums(view(p->Q, p->n, p->n), p->qnnz, nullptr, cols.label.data, p->n, q, error);
    if (st)
        return st;
    certify_classes<<<cols.count, 128>>>(cols.count, cols.order.data, cols.start.data, sum_view(at),
                                         sum_view(q), error);
    st = block_sums(view(p->A, p->m, p->n), p->annz, nullptr, cols.label.data, p->m, a, error);
    if (st)
        return st;
    certify_classes<<<rows.count, 128>>>(rows.count, rows.order.data, rows.start.data, sum_view(a),
                                         SumView{}, error);
    GPU_CHECK(cudaGetLastError());
    return OK;
}
__global__ static void export_matrix(int nnz, const Block *b, const int *start, int *col,
                                     double *value, int *error) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz) {
        int row = (int)(b[k].key >> 32);
        double v = b[k].hi; /* hi is the correctly rounded exact block sum. */
        if (start)
            v /= (double)(start[row + 1] - start[row]);
        if (!isfinite(v) || v == 0.)
            atomicExch(error, (int)NUMERICAL_ERROR);
        col[k] = (int)(unsigned)b[k].key;
        value[k] = v;
    }
}
static Status reduced_matrix(View a, int nnz, const Part &rows, const Part &cols, bool average,
                             DeviceMatrix &out, int &out_nnz, int *error) {
    Sums sums;
    Status st = block_sums(a, nnz, rows.label.data, cols.label.data, rows.count, sums, error);
    if (st)
        return st;
    out_nnz = sums.nnz;
    out.ptr.swap(sums.ptr);
    GPU_CHECK(out.col.reserve((size_t)out_nnz + 1));
    GPU_CHECK(out.val.reserve((size_t)out_nnz + 1));
    if (out_nnz)
        export_matrix<<<((unsigned)out_nnz + 255) / 256, 256>>>(out_nnz, sums.data.data,
                                                                average ? rows.start.data : nullptr,
                                                                out.col.data, out.val.data, error);
    GPU_CHECK(cudaGetLastError());
    return OK;
}
__global__ static void reduced_vectors(int groups, const int *order, const int *start,
                                       const int *ids, const double *l, const double *u,
                                       const double *c, int *newids, double *nl, double *nu,
                                       double *nc, int *error) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < groups) {
        int j = order[start[g]], size = start[g + 1] - start[g];
        newids[g] = ids[j];
        nl[g] = l[j];
        nu[g] = u[j];
        if (c)
            nc[g] = gpu_product2(c[j], (double)size, error);
    }
}
/* Every class includes its representative. Capture once before replacing the
 * model; later rules use that representative's stable slot. Separate owning
 * segments enforce the fold -> main recovery boundary. */
__global__ static void capture(int groups, const int *order, const int *start, const int *ids,
                               bool columns, int recbase, int entrybase, int m, int n,
                               DeviceRecord *records, int *index) {
    int g = blockIdx.x;
    if (g < groups) {
        int begin = start[g], end = start[g + 1];
        if (!threadIdx.x) {
            DeviceRecord r = {};
            r.type = columns ? REC_FOLD_COL : REC_FOLD_ROW;
            r.j = ids[order[begin]];
            r.row = r.j;
            r.m = m;
            r.n = n;
            r.goffset = entrybase + begin;
            r.gsize = end - begin;
            records[recbase + g] = r;
        }
        for (int k = begin + threadIdx.x; k < end; k += blockDim.x)
            index[entrybase + k] = ids[order[k]];
    }
}
static Status fold(State *s, void **opaque, double deadline, int &removed) {
    CudaPlan *p = (CudaPlan *)*opaque;
    if (!p->m || !p->n)
        return OK;
    if ((size_t)p->m + p->n > INT_MAX)
        return OK;
    Profile profile(p->m, p->n);
    Buffer<int> error;
    GPU_CHECK(error.reserve(1));
    GPU_CHECK(cudaMemset(error.data, 0, sizeof(int)));
    Part rows, cols;
    Status st = profile.timed(
        0, [&] { return partition(p->m, InitialLess{p->rl.data, p->ru.data, nullptr}, rows); });
    if (st)
        return st;
    st = profile.timed(
        0, [&] { return partition(p->n, InitialLess{p->l.data, p->u.data, p->c.data}, cols); });
    if (st)
        return st;
    profile.counts(rows.count, cols.count);
    if (rows.count == p->m && cols.count == p->n) {
        profile.reason = "small_reduction";
        return OK;
    }
    Signatures signatures;
    st = signatures.prepare(p);
    if (st)
        return st;
    Part nextcols, nextrows;
    Buffer<int> counts;
    GPU_CHECK(counts.reserve(2));
    // Keep reduction strength on high-diameter inputs. Unlike LP, Q can
    // split columns even when the row partition does not change.
    const long long max_rounds = 1LL + p->m - rows.count + p->n - cols.count;
    bool stable = false;
    for (int round = 0; round < max_rounds; ++round) {
        if (monotonic_seconds() >= deadline)
            return TIME_LIMIT;
        st = profile.timed(1, [&] { return signatures.columns(p, rows, cols, round); });
        if (st)
            return st;
        st = profile.timed(3, [&] {
            return partition(p->n, SignatureLess{cols.label.data, signatures.col.data}, nextcols,
                             counts.data + 1);
        });
        if (st)
            return st;
        st = profile.timed(4, [&] { return signatures.rows(p, nextcols, round); });
        if (st)
            return st;
        st = profile.timed(5, [&] {
            return partition(p->m, SignatureLess{rows.label.data, signatures.row.data}, nextrows,
                             counts.data);
        });
        if (st)
            return st;
        // Row signatures depend only on the new column labels, not their host
        // count. Both updates stay in stream order until this one round barrier.
        int host_counts[2];
        GPU_CHECK(counts.download(host_counts, 2));
        nextrows.count = host_counts[0];
        nextcols.count = host_counts[1];
        stable = rows.count == nextrows.count && cols.count == nextcols.count;
        swap_part(cols, nextcols);
        swap_part(rows, nextrows);
        ++s->execution.gpu_candidate_passes;
        ++profile.rounds;
        profile.counts(rows.count, cols.count);
        if (rows.count == p->m && cols.count == p->n) {
            profile.reason = "small_reduction";
            return OK;
        }
        if (stable)
            break;
    }
    if (!stable) {
        profile.reason = "work_limit";
        return OK;
    }
    // Projected signatures are only candidates. Certify all three block-sum
    // identities with the owning exact vectors before touching model or tape.
    st = profile.timed(2, [&] { return certify(p, rows, cols, error.data); });
    if (st)
        return st;
    int certification;
    GPU_CHECK(error.download(&certification, 1));
    if (certification) {
        profile.reason = "certificate_rejected";
        ++s->batches.numerical_rejections;
        return OK;
    }
    if (monotonic_seconds() >= deadline)
        return TIME_LIMIT;
    profile.reason = "accepted";
    std::unique_ptr<CudaPlan> next(new CudaPlan(p->device, rows.count, cols.count, p->transfers));
    st = reduced_matrix(view(p->A, p->m, p->n), p->annz, rows, cols, true, next->A, next->annz,
                        error.data);
    if (st)
        return st;
    st = reduced_matrix(view(p->Q, p->n, p->n), p->qnnz, cols, cols, false, next->Q, next->qnnz,
                        error.data);
    if (st)
        return st;
    GPU_CHECK(next->row_ids.reserve((size_t)next->m + 1));
    GPU_CHECK(next->col_ids.reserve((size_t)next->n + 1));
    GPU_CHECK(next->rl.reserve((size_t)next->m + 1));
    GPU_CHECK(next->ru.reserve((size_t)next->m + 1));
    GPU_CHECK(next->l.reserve((size_t)next->n + 1));
    GPU_CHECK(next->u.reserve((size_t)next->n + 1));
    GPU_CHECK(next->c.reserve((size_t)next->n + 1));
    GPU_CHECK(next->offset.reserve(1));
    GPU_CHECK(
        cuda_memcpy(next->offset.data, p->offset.data, sizeof(double), cudaMemcpyDeviceToDevice));
    reduced_vectors<<<((unsigned)next->m + 255) / 256, 256>>>(
        next->m, rows.order.data, rows.start.data, p->row_ids.data, p->rl.data, p->ru.data, nullptr,
        next->row_ids.data, next->rl.data, next->ru.data, nullptr, error.data);
    reduced_vectors<<<((unsigned)next->n + 255) / 256, 256>>>(
        next->n, cols.order.data, cols.start.data, p->col_ids.data, p->l.data, p->u.data, p->c.data,
        next->col_ids.data, next->l.data, next->u.data, next->c.data, error.data);
    int code;
    GPU_CHECK(error.download(&code, 1));
    if (code) {
        ++s->batches.numerical_rejections;
        return OK;
    }
    st = device_transpose(next->A, next->m, next->n, next->annz, next->AT);
    if (st)
        return st;
    validate_transformed<<<((unsigned)std::max(next->m, next->n) + 256) / 256, 256>>>(
        view(next->A, next->m, next->n), view(next->Q, next->n, next->n), next->annz, next->qnnz,
        next->c.data, next->l.data, next->u.data, next->rl.data, next->ru.data, error.data);
    GPU_CHECK(error.download(&code, 1));
    if (code) {
        ++s->batches.numerical_rejections;
        return OK;
    }
    if (monotonic_seconds() >= deadline)
        return TIME_LIMIT;
    std::unique_ptr<DeviceTapeSegment> segment(new DeviceTapeSegment());
    segment->folding = true;
    segment->count = next->m + next->n;
    segment->entries = p->m + p->n;
    GPU_CHECK(segment->records.reserve(segment->count));
    GPU_CHECK(segment->indices.reserve(segment->entries));
    GPU_CHECK(segment->values.reserve(segment->entries));
    GPU_CHECK(cudaMemset(segment->values.data, 0, (size_t)segment->entries * sizeof(double)));
    capture<<<next->m, 256>>>(next->m, rows.order.data, rows.start.data, p->row_ids.data, false, 0,
                              0, p->m, p->n, segment->records.data, segment->indices.data);
    capture<<<next->n, 256>>>(next->n, cols.order.data, cols.start.data, p->col_ids.data, true,
                              next->m, p->m, p->m, p->n, segment->records.data,
                              segment->indices.data);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaStreamSynchronize(0));
    DeviceTape &tape = **(TapeOwner *)s->device_tape;
    size_t count = segment->count, entries = segment->entries;
    tape.segments.push_back(std::move(segment)); /* Last fallible host operation. */
    tape.count += count;
    tape.entries += entries;
    tape.bytes += count * sizeof(DeviceRecord) + entries * (sizeof(int) + sizeof(double));
    s->count = tape.count;
    removed = p->m - next->m + p->n - next->n;
    s->reduced_m = next->m;
    s->reduced_n = next->n;
    p->pending = next.release();
    cuda_plan_commit(opaque);
    ++s->execution.gpu_batches;
    s->execution.gpu_reductions += removed;
    s->execution.largest_batch = std::max(s->execution.largest_batch, (size_t)removed);
    return OK;
}
} // namespace qp_folding
extern "C" int cuda_resident_fold(State *s, void **plan, double deadline) {
    DeviceGuard guard(s->device);
    TransferScope traffic(&s->transfers);
    if (guard.status != cudaSuccess)
        return -(int)CUDA_ERROR;
    try {
        Status ready = ensure_transpose((CudaPlan *)*plan);
        if (ready) return -(int)ready;
        int removed = 0;
        Status st = qp_folding::fold(s, plan, deadline, removed);
        return st ? -(int)st : removed;
    } catch (const std::bad_alloc &) {
        return -(int)OUT_OF_MEMORY;
    } catch (const thrust::system_error &) {
        return -(int)CUDA_ERROR;
    }
}
#endif
