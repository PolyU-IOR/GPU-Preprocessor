#ifndef TEST_ORACLE_H
#define TEST_ORACLE_H
#include "gpupresolve_qp.h"
/* Small independent active-face solver; no presolve implementation calls. */
int test_solve(const Problem *p, double *x, double *y, double *z);
#endif
