#define _POSIX_C_SOURCE 200809L
#include "gpupresolve_qp.h"
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(int code) {
    fprintf(code ? stderr : stdout,
        "Usage: gpu_preprocessor_qp INPUT.mps[.gz] [REDUCED.mps] [options]\n"
        "       gpu_preprocessor_qp --dataset DATA_DIRECTORY NEW_OUTPUT.csv [options]\n"
        "  --gpu N               Visible CUDA device index (default: 0)\n"
        "  --folding true|false   Enable folding (default: true)\n"
        "  --max-iters N          Presolve iteration budget (default: 10)\n"
        "  --max-time SECONDS     Soft presolve time limit (default: 1000)\n"
        "  --schedule fixed|tiered  Rule schedule (default: fixed)\n"
        "  --batch-limit N        Maximum candidates per batch (default: 65536)\n"
        "  --help                Show this help\n");
    return code;
}
static int parse_int(const char *s, int *v) {
    char *e;
    errno = 0;
    long n = strtol(s, &e, 10);
    if (errno || !*s || *e || n < 0 || n > INT_MAX)
        return 0;
    *v = (int)n;
    return 1;
}
static int has_suffix(const char *s, const char *suffix) {
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && !strcmp(s + a - b, suffix);
}
static int compare_names(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}
static void csv_string(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"')
            fputc('"', f);
        fputc(*s, f);
    }
    fputc('"', f);
}
static int dataset(const char *dir, const char *output, const Config *config,
                   const ExecutionOptions *execution) {
    DIR *d = opendir(dir);
    if (!d) {
        perror(dir);
        return 1;
    }
    char **names = NULL;
    size_t count = 0, capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(d))) {
        if (!has_suffix(entry->d_name, ".mps") && !has_suffix(entry->d_name, ".mps.gz"))
            continue;
        if (count == capacity) {
            size_t next = capacity ? 2 * capacity : 32;
            char **p = realloc(names, next * sizeof(*p));
            if (!p)
                goto failed;
            names = p;
            capacity = next;
        }
        names[count] = strdup(entry->d_name);
        if (!names[count])
            goto failed;
        count++;
    }
    closedir(d);
    d = NULL;
    if (count)
        qsort(names, count, sizeof(*names), compare_names);
    FILE *f = fopen(output, "wx");
    if (!f) {
        perror(output);
        goto failed;
    }
    fprintf(f, "input,status,m,n,reduced_m,reduced_n,nnz_A,nnz_Q,seconds,message\n");
    int failures = 0;
    for (size_t k = 0; k < count; k++) {
        size_t length = strlen(dir) + strlen(names[k]) + 2;
        char *path = malloc(length);
        if (!path) {
            failures++;
            continue;
        }
        snprintf(path, length, "%s/%s", dir, names[k]);
        Problem p = {0};
        Result r = {0};
        char message[256] = "";
        Status st = read_mps(path, &p, message, sizeof(message));
        if (st == OK) {
            st = presolve_with_options(&p, config, execution, &r);
            snprintf(message, sizeof(message), "%s", r.message);
        }
        csv_string(f, names[k]);
        fprintf(f, ",%s,%d,%d,%d,%d,%d,%d,%.9g,", status_string(st), p.A.rows, p.A.cols,
                r.reduced.A.rows, r.reduced.A.cols, r.reduced.A.nnz, r.reduced.Q.nnz,
                r.presolve_seconds);
        csv_string(f, message);
        fputc('\n', f);
        printf("%s: %s\n", names[k], status_string(st));
        if (st != OK && st != TIME_LIMIT && st != INFEASIBLE && st != UNBOUNDED)
            failures++;
        result_free(&r);
        problem_free(&p);
        free(path);
    }
    {
        int bad = ferror(f);
        if (fclose(f))
            bad = 1;
        if (bad)
            failures++;
    }
    for (size_t k = 0; k < count; k++)
        free(names[k]);
    free(names);
    return failures ? 1 : 0;
failed:
    if (d)
        closedir(d);
    for (size_t k = 0; k < count; k++)
        free(names[k]);
    free(names);
    return 1;
}
int main(int argc, char **argv) {
    Config config = default_config();
    ExecutionOptions execution = default_execution_options();
    const char *paths[2] = {NULL, NULL};
    const char *dataset_dir = NULL, *dataset_output = NULL;
    int path_count = 0, positional_only = 0;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!positional_only && !strcmp(arg, "--help"))
            return usage(0);
        if (!positional_only && !strcmp(arg, "--")) {
            positional_only = 1;
            continue;
        }
        if (positional_only || arg[0] != '-') {
            if (path_count == 2)
                return usage(2);
            paths[path_count++] = arg;
            continue;
        }
        if (!strcmp(arg, "--no-folding")) {
            config.rules &= ~(uint32_t)FOLDING;
            continue;
        }
        if (++i >= argc)
            return usage(2);
        const char *value = argv[i];
        if (!strcmp(arg, "--gpu")) {
            if (!parse_int(value, &config.device))
                return usage(2);
        } else if (!strcmp(arg, "--folding")) {
            if (!strcmp(value, "true") || !strcmp(value, "1"))
                config.rules |= FOLDING;
            else if (!strcmp(value, "false") || !strcmp(value, "0"))
                config.rules &= ~(uint32_t)FOLDING;
            else
                return usage(2);
        } else if (!strcmp(arg, "--max-iters")) {
            if (!parse_int(value, &config.max_iterations))
                return usage(2);
        } else if (!strcmp(arg, "--schedule")) {
            if (!strcmp(value, "fixed"))
                execution.schedule = SCHEDULE_FIXED;
            else if (!strcmp(value, "tiered"))
                execution.schedule = SCHEDULE_TIERED;
            else
                return usage(2);
        } else if (!strcmp(arg, "--batch-limit")) {
            if (!parse_int(value, &execution.batch_limit) || !execution.batch_limit)
                return usage(2);
        } else if (!strcmp(arg, "--max-time") || !strcmp(arg, "--max-seconds")) {
            char *end;
            errno = 0;
            config.max_seconds = strtod(value, &end);
            if (errno || !*value || *end || !isfinite(config.max_seconds) || config.max_seconds < 0.)
                return usage(2);
        } else if (!strcmp(arg, "--dataset")) {
            if (dataset_dir || ++i >= argc)
                return usage(2);
            dataset_dir = value;
            dataset_output = argv[i];
        } else {
            return usage(2);
        }
    }
    if (dataset_dir)
        return path_count ? usage(2) : dataset(dataset_dir, dataset_output, &config, &execution);
    if (!path_count)
        return usage(2);
    Problem p = {0};
    Result result = {0};
    char message[256];
    int code = 1;
    Status status = read_mps(paths[0], &p, message, sizeof(message));
    if (status) {
        fprintf(stderr, "%s: %s\n", status_string(status), message);
        goto done;
    }
    status = presolve_with_options(&p, &config, &execution, &result);
    printf("backend=%s status=%s message=%s\n", result.backend == CUDA ? "CUDA" : "CPU",
           status_string(status), result.message);
    if (status == OK || status == TIME_LIMIT) {
        printf("m,n: (%d,%d) -> (%d,%d)\n", p.A.rows, p.A.cols, result.reduced.A.rows,
               result.reduced.A.cols);
        printf("nnz(A),nnz(Q): (%d,%d) -> (%d,%d)\n", p.A.nnz, p.Q.nnz, result.reduced.A.nnz,
               result.reduced.Q.nnz);
        printf("presolve_seconds=%.9g iterations=%d reductions=%d\n", result.presolve_seconds,
               result.iterations, result.reductions);
        ExecutionInfo info;
        RuleInfo folding;
        if (rule_info(result.state, FOLDING, &folding) == OK)
            printf("folding_reductions=%llu folding_seconds=%.9g\n",
                   (unsigned long long)folding.reductions, folding.seconds);
        if (execution_info(result.state, &info) == OK)
            printf("gpu_batches=%llu gpu_reductions=%llu largest_batch=%zu cpu_fallbacks=%llu\n",
                   (unsigned long long)info.gpu_batches, (unsigned long long)info.gpu_reductions,
                   info.largest_batch, (unsigned long long)info.cpu_fallbacks);
        if (path_count == 2) {
            status = write_mps(paths[1], &result.reduced, message, sizeof(message));
            if (status) {
                fprintf(stderr, "%s: %s\n", status_string(status), message);
                goto done;
            }
            printf("Exported reduced QP: %s\n", paths[1]);
        }
    }
    code = (status == OK || status == TIME_LIMIT || status == INFEASIBLE ||
            status == UNBOUNDED)
               ? 0
               : 1;
done:
    result_free(&result);
    problem_free(&p);
    return code;
}
