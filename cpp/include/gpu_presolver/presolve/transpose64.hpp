#pragma once
#include "gpu_presolver/presolve/presolve_structs.hpp"
namespace gpu_presolver::presolve {
// Sorted CSR transpose. Uses bounded 32-bit library calls with 64-bit global offsets.
DeviceCsrMatrix transpose_csr64(const DeviceCsrMatrix& source);
}
