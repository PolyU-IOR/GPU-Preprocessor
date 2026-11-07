#include "backend/cuda_common.cuh"
#include "backend/cuda_tape.cuh"
#include <algorithm>
#include <cub/device/device_scan.cuh>
#include <memory>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>
#include <vector>


/* Rule implementations remain private to this translation unit. Device helpers
 * inline into the existing kernels; file boundaries add no launches or copies. */
#include "cuda_plan.cuh"
#include "cuda_rule_common.cuh"
#include "cuda_matrix_transform.cuh"
#include "rules/rule_empty_rows.cuh"
#include "rules/rule_singleton_rows.cuh"
#include "rules/rule_activity.cuh"
#include "rules/rule_parallel_rows.cuh"
#include "rules/rule_singleton_eq.cuh"
#include "rules/rule_doubleton_eq.cuh"
#include "rules/rule_fixed_cols.cuh"
#include "rules/rule_propagation.cuh"
#include "rules/rule_singleton_dual.cuh"
#include "rules/rule_parallel_cols.cuh"
#include "rules/rule_empty_cols.cuh"
#include "rules/rule_dual_fix.cuh"
#include "cuda_detection.cuh"
#include "cuda_batch.cuh"
#include "cuda_row_transform.cuh"
#include "cuda_transform.cuh"
#include "cuda_presolve_resident.cuh"
#include "rules/rule_folding.cuh"
