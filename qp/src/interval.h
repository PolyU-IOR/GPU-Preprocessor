#ifndef INTERVAL_H
#define INTERVAL_H
#include <math.h>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/* Enclose every operation, including finite overflow and subnormal products.
 * One outward ulp is conservative for IEEE round-to-nearest arithmetic.
 * An indeterminate infinity expression contributes no usable certificate. */
static HD inline double round_down(double value) {
    return isnan(value) ? -(double)INFINITY : nextafter(value, -(double)INFINITY);
}
static HD inline double round_up(double value) {
    return isnan(value) ? (double)INFINITY : nextafter(value, (double)INFINITY);
}
static HD inline double add_down(double a, double b) {
    if (a == 0.)
        return b;
    if (b == 0.)
        return a;
    if (isfinite(a) && a == -b)
        return 0.;
    return round_down(a + b);
}
static HD inline double add_up(double a, double b) {
    if (a == 0.)
        return b;
    if (b == 0.)
        return a;
    if (isfinite(a) && a == -b)
        return 0.;
    return round_up(a + b);
}
static HD inline double mul_down(double a, double b) {
    if (a == 0. || b == 0.)
        return 0.;
    return round_down(a * b);
}
static HD inline double mul_up(double a, double b) {
    if (a == 0. || b == 0.)
        return 0.;
    return round_up(a * b);
}
static HD inline double div_down(double a, double b) {
    if (a == 0.)
        return 0.;
    return round_down(a / b);
}
static HD inline double div_up(double a, double b) {
    if (a == 0.)
        return 0.;
    return round_up(a / b);
}
/* Affine substitutions can leave a sub-ulp inconsistency between a fixed
 * variable and a derived bound. Defer that proposal within the configured
 * feasibility tolerance; never widen an input bound or commit an empty box. */
static HD inline int small_bound_conflict(double lower, double upper, double tolerance) {
    return tolerance > 0. && isfinite(lower) && isfinite(upper) && lower > upper &&
           (lower - upper) / (1. + fmax(fabs(lower), fabs(upper))) <= tolerance;
}
/* Test real proportionality after exponent normalization. A raw FMA residual
 * can round to zero for subnormal coefficients even when curvature differs. */
static HD inline int coefficient_exact_scaled(double a, double b, double ratio) {
    if (!isfinite(ratio) || ratio == 0.)
        return 0;
    if (a == 0.)
        return b == 0.;
    if (b == 0.)
        return 0;
    int ea, eb, er;
    double ma = frexp(a, &ea), mb = frexp(b, &eb), mr = frexp(ratio, &er);
    int shift = eb - ea - er;
    if (shift < -4 || shift > 4)
        return 0;
    double target = ldexp(mb, shift);
    return target == mr * ma && fma(-mr, ma, target) == 0.;
}
#undef HD
#endif
