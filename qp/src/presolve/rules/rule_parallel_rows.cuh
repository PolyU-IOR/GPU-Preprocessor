#pragma once

__device__ static __forceinline__ void detect_parallel_rows_rule(View A, int first, int count, int position, const int * order, const unsigned long long * hashes, Candidate &r) {
    int begin = 0, end = position;
    while (begin < end) {
        int mid = begin + (end - begin) / 2;
        if (hashes[mid] < hashes[position])
            begin = mid + 1;
        else
            end = mid;
    }
    for (int slot = begin; slot < position; ++slot) {
        int k = order[slot];
        int start = A.ptr[k];
        if (A.ptr[k + 1] - start != count)
            continue;
        double ratio = A.val[first] / A.val[start];
        bool match = ratio != 0. && isfinite(ratio);
        for (int t = 0; t < count && match; ++t)
            match = A.col[first + t] == A.col[start + t] &&
                    scaled(A.val[start + t], A.val[first + t], ratio);
        if (match) {
            r.action = GPU_PAR_ROW;
            r.k = k;
            r.ratio = ratio;
            break;
        }
    }
}
