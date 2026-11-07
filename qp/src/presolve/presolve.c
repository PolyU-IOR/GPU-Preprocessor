#include "internal.h"

static const uint32_t rule_order[] = {
    EMPTY_ROWS,    SINGLETON_ROWS, ACTIVITY,   PROPAGATION,
    PARALLEL_ROWS, FIXED_COLS,     EMPTY_COLS, SINGLETON_DUAL,
    SINGLETON_EQ,  DOUBLETON_EQ,   DUAL_FIX,   PARALLEL_COLS};
static Status run_stage(Result *out, const Config *cfg, const ExecutionOptions *options,
                           uint32_t mask, size_t budget, double start, void **plan, int *changed) {
    for (size_t k = 0; k < sizeof(rule_order) / sizeof(*rule_order); ++k)
        if (cfg->rules & mask & rule_order[k]) {
            size_t rounds = 0;
            int batches = 0, thin_batches = 0;
            for (;;) {
                if (monotonic_seconds() - start >= cfg->max_seconds)
                    return TIME_LIMIT;
                int slot = 0;
                while ((1u << slot) != rule_order[k])
                    ++slot;
                RuleInfo *info = &out->state->rules[slot];
                double rule_start = monotonic_seconds();
                int limit = options->batch_limit;
                if ((size_t)limit > budget - rounds)
                    limit = (int)(budget - rounds);
                int rc = cuda_resident_apply(out->state, cfg, rule_order[k], limit, plan,
                                             start + cfg->max_seconds);
                out->reduced.A.rows = out->state->reduced_m;
                out->reduced.A.cols = out->reduced.Q.rows = out->reduced.Q.cols =
                    out->state->reduced_n;
                ++info->calls;
                info->seconds += monotonic_seconds() - rule_start;
                if (rc > 0)
                    info->reductions += (uint64_t)rc;
                if (rc < 0)
                    return (Status)-rc;
                if (!rc)
                    break;
                if (out->reductions > INT_MAX - rc)
                    return NUMERICAL_ERROR;
                out->reductions += rc;
                *changed = 1;
                rounds += (size_t)rc;
                if (rounds >= budget)
                    break;
                if (options->schedule == SCHEDULE_TIERED &&
                    out->reduced.A.cols >= 4096) {
                    ++batches;
                    if (rule_order[k] == PROPAGATION && batches >= 32)
                        break;
                    if (rule_order[k] == DOUBLETON_EQ) {
                        int minimum = out->reduced.A.cols / 65536;
                        if (minimum < 4)
                            minimum = 4;
                        thin_batches = rc < minimum ? thin_batches + 1 : 0;
                        if (thin_batches >= 16)
                            break;
                    }
                }
            }
        }
    return OK;
}
Status presolve(const Problem *p, const Config *config, Result *out) {
    return presolve_with_options(p, config, NULL, out);
}
static Status presolve_impl(const Problem *p, const Config *user,
                               const ExecutionOptions *execution, Result *out, int download,
                               const DeviceProblemView *device_source) {
    if (!out)
        return INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    void *gpu_plan = NULL;
    ExecutionOptions options = execution ? *execution : default_execution_options();
    Config c = user ? *user : default_config();
    if (options.schedule < SCHEDULE_FIXED || options.schedule > SCHEDULE_TIERED ||
        options.batch_limit < 1 || options.cleanup_rounds < 1 || options.global_period < 1) {
        out->status = INVALID_ARGUMENT;
        set_message(out->message, sizeof(out->message), "Invalid execution options.");
        return INVALID_ARGUMENT;
    }
    Status st = device_source ? OK : validate(p, out->message, sizeof(out->message));
    if (st)
        goto finish;
    if ((c.backend != CUDA && c.backend != AUTO) || c.device < 0 || c.max_iterations < 0 ||
        !isfinite(c.max_seconds) || c.max_seconds < 0. || !isfinite(c.feasibility_tol) ||
        c.feasibility_tol < 0. || !isfinite(c.bound_tol) || c.bound_tol < 0. ||
        !isfinite(c.zero_tol) || c.zero_tol < 0. || isnan(c.max_q_fill_ratio) ||
        c.max_q_fill_ratio < 1. || (c.rules & ~(uint32_t)ALL_RULES)) {
        st = INVALID_ARGUMENT;
        set_message(out->message, sizeof(out->message), "Invalid presolve configuration.");
        goto finish;
    }
    c.backend = CUDA;
    out->backend = CUDA;
    if (!cuda_available(c.device)) {
        st = CUDA_ERROR;
        goto finish;
    }
    out->state = state_create(0, 0);
    if (!out->state) {
        st = OUT_OF_MEMORY;
        goto finish;
    }
    out->state->backend = c.backend;
    out->state->device = c.device;
    out->state->tolerance = fmax(c.bound_tol, 1e-10);
    out->state->original_m = out->state->reduced_m = p->A.rows;
    out->state->original_n = out->state->reduced_n = p->A.cols;
    st = device_source ? cuda_resident_create_device(device_source, out->state, &gpu_plan,
                                                     &out->reduced)
                       : cuda_resident_create(p, out->state, &gpu_plan, &out->reduced);
    if (st)
        goto finish;
    double start = monotonic_seconds();
    if (c.max_seconds == 0.) {
        st = TIME_LIMIT;
        goto timed_finish;
    }
    size_t budget = 2 * ((size_t)p->A.rows + (size_t)p->A.cols) + 10;
    uint32_t cleanup = EMPTY_ROWS | SINGLETON_ROWS | FIXED_COLS | EMPTY_COLS;
    uint32_t global = PARALLEL_ROWS | DOUBLETON_EQ | PARALLEL_COLS;
    if ((c.rules & FOLDING) && c.max_iterations > 0) {
        double fold_start = monotonic_seconds();
        int rc = cuda_resident_fold(out->state, &gpu_plan, start + c.max_seconds);
        RuleInfo *info = &out->state->rules[12];
        ++info->calls;
        info->seconds += monotonic_seconds() - fold_start;
        if (rc < 0) {
            st = (Status)-rc;
            goto timed_finish;
        }
        info->reductions += (uint64_t)rc;
        if (out->reductions > INT_MAX - rc) {
            st = NUMERICAL_ERROR;
            goto timed_finish;
        }
        out->reductions += rc;
        out->reduced.A.rows = out->state->reduced_m;
        out->reduced.A.cols = out->reduced.Q.rows = out->reduced.Q.cols = out->state->reduced_n;
    }
    for (int iter = 0; iter < c.max_iterations; ++iter) {
        int changed = 0;
        out->iterations = iter + 1;
        if (options.schedule == SCHEDULE_FIXED)
            st = run_stage(out, &c, &options, ALL_RULES, budget, start, &gpu_plan, &changed);
        else {
            for (int round = 0; round < options.cleanup_rounds; ++round) {
                int delta = 0;
                st = run_stage(out, &c, &options, cleanup, budget, start, &gpu_plan, &delta);
                changed |= delta;
                if (st || !delta)
                    break;
            }
            if (!st)
                st = run_stage(out, &c, &options, ALL_RULES & ~(cleanup | global), budget,
                               start, &gpu_plan, &changed);
            if (!st && (!changed || (iter + 1) % options.global_period == 0 ||
                        iter + 1 == c.max_iterations))
                st = run_stage(out, &c, &options, global, budget, start, &gpu_plan, &changed);
        }
        if (st || !changed)
            break;
    }
timed_finish:
    out->presolve_seconds = monotonic_seconds() - start;
    if (st == OK || st == TIME_LIMIT) {
        Status end_status = cuda_resident_finish(&gpu_plan, out->state, &out->reduced);
        if (!end_status && download)
            end_status = result_download(out);
        if (end_status)
            st = end_status;
    }
    if ((st == OK || st == TIME_LIMIT) && (out->state->reduced_m != out->reduced.A.rows ||
                                                   out->state->reduced_n != out->reduced.A.cols))
        st = INVALID_ARGUMENT;
finish:
    cuda_plan_free(&gpu_plan);
    out->status = st;
    if (st != OK && st != TIME_LIMIT) {
        state_free(&out->state);
        problem_free(&out->reduced);
    }
    if (!out->message[0])
        set_message(out->message, sizeof(out->message), status_string(st));
    return st;
}
Status presolve_with_options(const Problem *p, const Config *c,
                                    const ExecutionOptions *o, Result *r) {
    return presolve_impl(p, c, o, r, 1, NULL);
}
Status presolve_resident(const Problem *p, const Config *c,
                                const ExecutionOptions *o, Result *r) {
    Config cfg = c ? *c : default_config();
    if (cfg.backend == CPU) {
        if (r) {
            memset(r, 0, sizeof(*r));
            r->status = INVALID_ARGUMENT;
        }
        return INVALID_ARGUMENT;
    }
    if (cfg.backend == AUTO)
        cfg.backend = CUDA;
    return presolve_impl(p, &cfg, o, r, 0, NULL);
}
Status presolve_device(const DeviceProblemView *p, const Config *c,
                              const ExecutionOptions *o, Result *r) {
    if (!p || !r)
        return INVALID_ARGUMENT;
    Config cfg = c ? *c : default_config();
    if (!c)
        cfg.device = p->device;
    if (cfg.backend < CPU || cfg.backend > AUTO || cfg.backend == CPU ||
        cfg.device != p->device || p->device < 0 || p->A.rows < 0 || p->A.cols < 0 ||
        p->A.nnz < 0 || p->Q.nnz < 0 || p->Q.rows != p->A.cols || p->Q.cols != p->A.cols) {
        memset(r, 0, sizeof(*r));
        r->status = INVALID_ARGUMENT;
        return INVALID_ARGUMENT;
    }
    cfg.backend = CUDA;
    Problem shape = {0};
    shape.A.rows = p->A.rows;
    shape.A.cols = shape.Q.rows = shape.Q.cols = p->A.cols;
    shape.A.nnz = p->A.nnz;
    shape.Q.nnz = p->Q.nnz;
    return presolve_impl(&shape, &cfg, o, r, 0, p);
}
