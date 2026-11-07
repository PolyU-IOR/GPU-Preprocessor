/* Optional audit adapter; the production library never calls this module. */
#include "support/qp_oracle.h"
int test_oracle_solve(const Problem *p, double *x, double *y, double *z) {
    if (!p || p->A.cols > 8 || p->A.rows > 8)
        return 0;
    return test_solve(p, x, y, z);
}
