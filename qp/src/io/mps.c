#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include <ctype.h>
#include <errno.h>
#include <zlib.h>

typedef struct {
    char **names;
    int *slots;
    size_t count, capacity, buckets;
} Names;
static uint64_t name_hash(const char *s) {
    uint64_t h = UINT64_C(1469598103934665603);
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= UINT64_C(1099511628211);
    }
    return h;
}
static void names_free(Names *t) {
    for (size_t i = 0; i < t->count; i++)
        free(t->names[i]);
    free(t->names);
    free(t->slots);
    memset(t, 0, sizeof(*t));
}
static int names_find(const Names *t, const char *s) {
    if (!t->buckets)
        return -1;
    size_t pos = (size_t)name_hash(s) & (t->buckets - 1);
    while (t->slots[pos]) {
        int i = t->slots[pos] - 1;
        if (!strcmp(t->names[i], s))
            return i;
        pos = (pos + 1) & (t->buckets - 1);
    }
    return -1;
}
static int names_add(Names *t, const char *s) {
    int old = names_find(t, s);
    if (old >= 0)
        return old;
    if (t->count >= INT_MAX - 1)
        return -1;
    if (!t->buckets || 2 * (t->count + 1) >= t->buckets) {
        size_t buckets = t->buckets ? 2 * t->buckets : 64;
        int *slots = checked_calloc(buckets, sizeof(int));
        if (!slots)
            return -1;
        for (size_t i = 0; i < t->count; i++) {
            size_t pos = (size_t)name_hash(t->names[i]) & (buckets - 1);
            while (slots[pos])
                pos = (pos + 1) & (buckets - 1);
            slots[pos] = (int)i + 1;
        }
        free(t->slots);
        t->slots = slots;
        t->buckets = buckets;
    }
    if (t->count == t->capacity) {
        size_t cap = t->capacity ? 2 * t->capacity : 32;
        char **p = realloc(t->names, cap * sizeof(*p));
        if (!p)
            return -1;
        t->names = p;
        t->capacity = cap;
    }
    char *copy = strdup(s);
    if (!copy)
        return -1;
    size_t pos = (size_t)name_hash(s) & (t->buckets - 1);
    while (t->slots[pos])
        pos = (pos + 1) & (t->buckets - 1);
    t->names[t->count] = copy;
    t->slots[pos] = (int)t->count + 1;
    return (int)t->count++;
}
static int number(const char *text, double *value) {
    size_t n = strlen(text);
    if (n > 127)
        return 0;
    char copy[128];
    memcpy(copy, text, n + 1);
    for (size_t i = 0; i < n; i++)
        if (copy[i] == 'D' || copy[i] == 'd')
            copy[i] = 'E';
    char *end;
    errno = 0;
    *value = strtod(copy, &end);
    return end != copy && !*end && isfinite(*value) && (errno != ERANGE || *value != 0.);
}
enum Section { S_NONE, S_ROWS, S_COLS, S_RHS, S_RANGES, S_BOUNDS, S_QUAD, S_SENSE, S_OBJNAME };
typedef struct {
    char type;
    double rhs, range;
    int has_range;
} Row;
typedef struct {
    double c, l, u;
    int explicit_lower;
} Col;
static int grow_rows(Row **rows, size_t *capacity, size_t need) {
    if (need <= *capacity)
        return 1;
    size_t cap = *capacity ? 2 * *capacity : 32;
    while (cap < need)
        cap *= 2;
    Row *p = realloc(*rows, cap * sizeof(*p));
    if (!p)
        return 0;
    memset(p + *capacity, 0, (cap - *capacity) * sizeof(*p));
    *rows = p;
    *capacity = cap;
    return 1;
}
static int grow_cols(Col **cols, size_t *capacity, size_t need) {
    if (need <= *capacity)
        return 1;
    size_t cap = *capacity ? 2 * *capacity : 32;
    while (cap < need)
        cap *= 2;
    Col *p = realloc(*cols, cap * sizeof(*p));
    if (!p)
        return 0;
    for (size_t j = *capacity; j < cap; j++)
        p[j] = (Col){0., 0., INFINITY, 0};
    *cols = p;
    *capacity = cap;
    return 1;
}
Status read_mps(const char *path, Problem *out, char *msg, size_t cap) {
    if (!path || !out)
        return INVALID_ARGUMENT;
    gzFile f = gzopen(path, "rb");
    if (!f) {
        set_message(msg, cap, "Cannot open MPS input.");
        return IO_ERROR;
    }
    Names rows = {0}, cols = {0};
    Row *rd = NULL;
    Col *cd = NULL;
    size_t rc = 0, cc = 0;
    COO A = {0}, Q = {0}, cost = {0};
    Problem p = {0};
    Matrix rawq = {0}, costs = {0};
    Status status = INVALID_ARGUMENT;
    const char *error = "Malformed or unsupported MPS input.";
    enum Section section = S_NONE;
    char objective[256] = "", rhs_set[256] = "", range_set[256] = "", bound_set[256] = "",
         last_col[256] = "";
    double offset = 0.;
    int line_number = 0, saw_end = 0, saw_rows = 0, saw_cols = 0, saw_objective = 0, saw_quad = 0;
    int seen_rhs = 0, seen_range = 0, use_rhs = 1, use_range = 1, qsection = 0;
    char line[8192];
#define BAD(text)                                                                                  \
    do {                                                                                           \
        error = (text);                                                                            \
        if (status == OK)                                                                      \
            status = INVALID_ARGUMENT;                                                         \
        goto cleanup;                                                                              \
    } while (0)
#define OOM()                                                                                      \
    do {                                                                                           \
        status = OUT_OF_MEMORY;                                                                \
        BAD("MPS allocation failed.");                                                             \
    } while (0)
    while (gzgets(f, line, sizeof(line))) {
        line_number++;
        if (!strchr(line, '\n') && !gzeof(f))
            BAD("MPS line exceeds 8191 bytes.");
        char *start = line;
        while (isspace((unsigned char)*start))
            start++;
        if (!*start || *start == '*')
            continue;
        char *t[8], *save = NULL;
        int nt = 0;
        for (char *v = strtok_r(start, " \t\r\n", &save); v; v = strtok_r(NULL, " \t\r\n", &save)) {
            if (*v == '$')
                break;
            if (nt == 8)
                BAD("Too many fields on an MPS line.");
            t[nt++] = v;
        }
        if (!nt)
            continue;
        if (section == S_SENSE) {
            if (nt != 1 || (strcmp(t[0], "MIN") && strcmp(t[0], "MINIMIZE")))
                BAD("Only minimization QPs are supported.");
            section = S_NONE;
            continue;
        }
        if (section == S_OBJNAME) {
            if (nt != 1 || strlen(t[0]) >= sizeof(objective))
                BAD("Invalid objective name.");
            strcpy(objective, t[0]);
            section = S_NONE;
            continue;
        }
        /* Section names are only accepted in header position, or alone. This
         * permits variables and row names that happen to equal a keyword. */
        int header = !isspace((unsigned char)line[0]) || nt == 1;
        if (header) {
            if (nt <= 2 && !strcmp(t[0], "NAME")) {
                if (saw_rows)
                    BAD("NAME must precede ROWS.");
                section = S_NONE;
                continue;
            }
            if (nt == 1 && !strcmp(t[0], "ROWS")) {
                if (saw_rows || saw_cols)
                    BAD("Duplicate or out-of-order ROWS section.");
                section = S_ROWS;
                saw_rows = 1;
                continue;
            }
            if (nt == 1 && !strcmp(t[0], "COLUMNS")) {
                if (!saw_rows || !saw_objective || saw_cols)
                    BAD("COLUMNS requires a defined objective row.");
                section = S_COLS;
                saw_cols = 1;
                continue;
            }
            if (nt == 1 && !strcmp(t[0], "RHS")) {
                if (!saw_cols)
                    BAD("Constraint data must follow COLUMNS.");
                section = S_RHS;
                continue;
            }
            if (nt == 1 && !strcmp(t[0], "RANGES")) {
                if (!saw_cols)
                    BAD("Constraint data must follow COLUMNS.");
                section = S_RANGES;
                continue;
            }
            if (nt == 1 && !strcmp(t[0], "BOUNDS")) {
                if (!saw_cols)
                    BAD("Constraint data must follow COLUMNS.");
                section = S_BOUNDS;
                continue;
            }
            if (nt <= 2 && (!strcmp(t[0], "QUADOBJ") || !strcmp(t[0], "QMATRIX") ||
                            !strcmp(t[0], "QSECTION"))) {
                if (!saw_cols || saw_quad)
                    BAD("Duplicate or out-of-order quadratic section.");
                if (!strcmp(t[0], "QSECTION")) {
                    if (nt != 2 || strcmp(t[1], objective))
                        BAD("Only objective QSECTION is supported; quadratic constraints are "
                            "rejected.");
                } else if (nt != 1)
                    BAD("Unexpected quadratic section argument.");
                qsection = !strcmp(t[0], "QSECTION");
                saw_quad = 1;
                section = S_QUAD;
                continue;
            }
            if (nt == 1 && !strcmp(t[0], "ENDATA")) {
                saw_end = 1;
                break;
            }
            if (nt <= 2 && !strcmp(t[0], "OBJSENSE")) {
                if (saw_rows)
                    BAD("OBJSENSE must precede ROWS.");
                if (nt == 2) {
                    if (strcmp(t[1], "MIN") && strcmp(t[1], "MINIMIZE"))
                        BAD("Only minimization QPs are supported.");
                    section = S_NONE;
                } else
                    section = S_SENSE;
                continue;
            }
            if (nt <= 2 && !strcmp(t[0], "OBJNAME")) {
                if (saw_rows)
                    BAD("OBJNAME must precede ROWS.");
                if (nt == 2) {
                    if (strlen(t[1]) >= sizeof(objective))
                        BAD("Objective name too long.");
                    strcpy(objective, t[1]);
                    section = S_NONE;
                } else
                    section = S_OBJNAME;
                continue;
            }
        }
        if (section == S_ROWS) {
            if (nt != 2 || strlen(t[0]) != 1 || !strchr("NLEG", t[0][0]) || strlen(t[1]) >= 256)
                BAD("Invalid ROWS record.");
            if (t[0][0] == 'N' && (!objective[0] || !strcmp(objective, t[1]))) {
                if (saw_objective)
                    BAD("Duplicate objective row.");
                saw_objective = 1;
                if (!objective[0])
                    strcpy(objective, t[1]);
                continue;
            }
            if (!strcmp(t[1], objective) || names_find(&rows, t[1]) >= 0)
                BAD("Duplicate row name.");
            int i = names_add(&rows, t[1]);
            if (i < 0 || !grow_rows(&rd, &rc, rows.count))
                OOM();
            rd[i].type = t[0][0];
            continue;
        }
        if (section == S_COLS) {
            if (nt >= 3 && (!strcmp(t[1], "'MARKER'") ||
                            (!strcmp(t[1], "MARKER") &&
                             (!strcmp(t[2], "'INTORG'") || !strcmp(t[2], "'INTEND'")))))
                BAD("Only continuous variables are supported.");
            int begin = 1;
            const char *name = t[0];
            if (nt == 2 || nt == 4) {
                name = last_col;
                begin = 0;
            }
            if (!*name || strlen(name) >= 256 || !(nt - begin == 2 || nt - begin == 4))
                BAD("Invalid COLUMNS record.");
            int j = names_add(&cols, name);
            if (j < 0 || !grow_cols(&cd, &cc, cols.count))
                OOM();
            strcpy(last_col, name);
            for (int k = begin; k < nt; k += 2) {
                double v;
                if (!number(t[k + 1], &v))
                    BAD("Invalid matrix coefficient.");
                if (!strcmp(t[k], objective))
                    coo_add(&cost, 0, j, v);
                else {
                    int i = names_find(&rows, t[k]);
                    if (i < 0)
                        BAD("Unknown row in COLUMNS.");
                    coo_add(&A, i, j, v);
                }
            }
            continue;
        }
        if (section == S_RHS || section == S_RANGES) {
            int begin = (nt == 3 || nt == 5) ? 1 : 0;
            if (!(nt - begin == 2 || nt - begin == 4))
                BAD("Invalid RHS/RANGES record.");
            char *chosen = section == S_RHS ? rhs_set : range_set;
            int *seen = section == S_RHS ? &seen_rhs : &seen_range;
            int *use = section == S_RHS ? &use_rhs : &use_range;
            if (begin) {
                if (strlen(t[0]) >= 256)
                    BAD("Vector name too long.");
                if (!*seen)
                    strcpy(chosen, t[0]);
                *use = !strcmp(chosen, t[0]);
            }
            if (!*seen)
                *seen = 1;
            if (!*use)
                continue;
            for (int k = begin; k < nt; k += 2) {
                double v;
                if (!number(t[k + 1], &v))
                    BAD("Invalid RHS or range value.");
                if (!strcmp(t[k], objective)) {
                    if (section == S_RHS)
                        offset = -v;
                    continue;
                }
                int i = names_find(&rows, t[k]);
                if (i < 0)
                    BAD("Unknown row in RHS/RANGES.");
                if (section == S_RHS)
                    rd[i].rhs = v;
                else {
                    rd[i].range = v;
                    rd[i].has_range = 1;
                }
            }
            continue;
        }
        if (section == S_BOUNDS) {
            if (nt < 3 || nt > 4 || strlen(t[1]) >= 256)
                BAD("Invalid BOUNDS record.");
            if (!bound_set[0])
                strcpy(bound_set, t[1]);
            if (strcmp(bound_set, t[1]))
                continue;
            int j = names_find(&cols, t[2]);
            if (j < 0)
                BAD("Unknown variable in BOUNDS.");
            double v = 0.;
            if (nt == 4 && !number(t[3], &v))
                BAD("Invalid bound value.");
            if (!strcmp(t[0], "LO")) {
                if (nt != 4)
                    BAD("LO requires a value.");
                cd[j].l = v;
                cd[j].explicit_lower = 1;
            } else if (!strcmp(t[0], "UP")) {
                if (nt != 4)
                    BAD("UP requires a value.");
                cd[j].u = v;
                if (v < 0 && !cd[j].explicit_lower)
                    cd[j].l = -INFINITY;
            } else if (!strcmp(t[0], "FX")) {
                if (nt != 4)
                    BAD("FX requires a value.");
                cd[j].l = cd[j].u = v;
                cd[j].explicit_lower = 1;
            } else if (!strcmp(t[0], "FR")) {
                cd[j].l = -INFINITY;
                cd[j].u = INFINITY;
                cd[j].explicit_lower = 1;
            } else if (!strcmp(t[0], "MI")) {
                cd[j].l = -INFINITY;
                cd[j].explicit_lower = 1;
            } else if (!strcmp(t[0], "PL"))
                cd[j].u = INFINITY;
            else
                BAD("Unsupported bound type; integer and semicontinuous variables are rejected.");
            continue;
        }
        if (section == S_QUAD) {
            if (nt != 3 && !(qsection && nt == 5))
                BAD("Invalid quadratic objective record.");
            int i = names_find(&cols, t[0]);
            for (int k = 1; k < nt; k += 2) {
                int j = names_find(&cols, t[k]);
                double v;
                if (i < 0 || j < 0 || !number(t[k + 1], &v))
                    BAD("Unknown quadratic variable or invalid value.");
                coo_add(&Q, i, j, v);
            }
            continue;
        }
        BAD("Unknown MPS section or unexpected data.");
    }
    /* Read through the gzip trailer even after ENDATA, validating CRC and
     * truncation before committing a model to the caller. */
    if (saw_end) {
        int got;
        do {
            got = gzread(f, line, sizeof(line));
        } while (got > 0);
    }
    int gz_status;
    gzerror(f, &gz_status);
    if (gz_status != Z_OK && gz_status != Z_STREAM_END) {
        status = IO_ERROR;
        BAD("MPS decompression or input read failed.");
    }
    if (gzclose(f) != Z_OK) {
        f = NULL;
        status = IO_ERROR;
        BAD("MPS input close failed.");
    }
    f = NULL;
    if (!saw_end || !saw_rows || !saw_cols || !saw_objective)
        BAD("MPS requires ROWS, COLUMNS, an objective and ENDATA.");
    status = problem_alloc(&p, (int)rows.count, (int)cols.count);
    if (status)
        goto cleanup;
    status = coo_finish(&A, (int)rows.count, (int)cols.count, &p.A);
    if (status)
        goto cleanup;
    status = coo_finish(&Q, (int)cols.count, (int)cols.count, &rawq);
    if (status)
        goto cleanup;
    for (int i = 0; i < rawq.rows; i++)
        for (int k = rawq.row_ptr[i]; k < rawq.row_ptr[i + 1]; k++) {
            int j = rawq.col_ind[k];
            double v = rawq.values[k], mirror = matrix_get(&rawq, j, i);
            if (i != j && mirror != 0. && v != mirror) {
                status = INVALID_ARGUMENT;
                BAD("Asymmetric full QMATRIX.");
            }
            coo_add(&Q, i, j, v);
            if (i != j && mirror == 0.)
                coo_add(&Q, j, i, v);
        }
    status = coo_finish(&Q, (int)cols.count, (int)cols.count, &p.Q);
    if (status)
        goto cleanup;
    status = coo_finish(&cost, 1, (int)cols.count, &costs);
    if (status)
        goto cleanup;
    for (int j = 0; j < p.A.cols; j++) {
        p.c[j] = matrix_get(&costs, 0, j);
        p.lower[j] = cd[j].l;
        p.upper[j] = cd[j].u;
    }
    for (int i = 0; i < p.A.rows; i++) {
        if (rd[i].type == 'E')
            p.row_lower[i] = p.row_upper[i] = rd[i].rhs;
        else if (rd[i].type == 'L')
            p.row_upper[i] = rd[i].rhs;
        else if (rd[i].type == 'G')
            p.row_lower[i] = rd[i].rhs;
        if (rd[i].has_range) {
            double v = rd[i].range;
            if (rd[i].type == 'L' || (rd[i].type == 'E' && v < 0.))
                p.row_lower[i] = p.row_upper[i] - fabs(v);
            else if (rd[i].type == 'G' || rd[i].type == 'E')
                p.row_upper[i] = p.row_lower[i] + fabs(v);
            if (rd[i].type != 'N' && (!isfinite(p.row_lower[i]) || !isfinite(p.row_upper[i]))) {
                status = NUMERICAL_ERROR;
                BAD("Ranged constraint bounds overflow double precision.");
            }
        }
    }
    p.objective_offset = offset;
    status = validate(&p, msg, cap);
    if (status) {
        error = "Invalid QP model in MPS input.";
        goto cleanup;
    }
    problem_free(out);
    *out = p;
    memset(&p, 0, sizeof(p));
    set_message(msg, cap, "");
cleanup:
    if (status != OK && msg && cap)
        snprintf(msg, cap, "%s (line %d)", error, line_number);
    if (f)
        gzclose(f);
    names_free(&rows);
    names_free(&cols);
    free(rd);
    free(cd);
    free(A.data);
    free(Q.data);
    free(cost.data);
    matrix_free(&costs);
    matrix_free(&rawq);
    problem_free(&p);
    return status;
#undef BAD
#undef OOM
}
Status write_mps(const char *path, const Problem *p, char *msg, size_t cap) {
    if (!path || validate(p, msg, cap) != OK)
        return INVALID_ARGUMENT;
    for (int i = 0; i < p->A.rows; ++i) {
        if (p->row_lower[i] > p->row_upper[i]) {
            set_message(msg, cap, "Cannot export contradictory row bounds as one ranged row.");
            return INVALID_ARGUMENT;
        }
        if (isfinite(p->row_lower[i]) && isfinite(p->row_upper[i]) &&
            !isfinite(p->row_upper[i] - p->row_lower[i])) {
            set_message(msg, cap, "MPS range width overflows double precision.");
            return NUMERICAL_ERROR;
        }
    }
    FILE *f = fopen(path, "wx");
    if (!f) {
        set_message(msg, cap, "Cannot create output; existing files are never replaced.");
        return IO_ERROR;
    }
    fprintf(f, "NAME          REDUCED\nROWS\n N  OBJ\n");
    for (int i = 0; i < p->A.rows; i++) {
        char type = p->row_lower[i] == p->row_upper[i] ? 'E'
                    : isfinite(p->row_upper[i])        ? 'L'
                    : isfinite(p->row_lower[i])        ? 'G'
                                                       : 'N';
        fprintf(f, " %c  R%d\n", type, i);
    }
    fprintf(f, "COLUMNS\n");
    Matrix at = {0};
    Status status = matrix_transpose(&p->A, &at);
    if (status) {
        fclose(f);
        set_message(msg, cap, "Cannot allocate MPS transpose.");
        return status;
    }
    for (int j = 0; j < p->A.cols; j++) {
        fprintf(f, "    X%d  OBJ  %.17g\n", j, p->c[j]);
        for (int k = at.row_ptr[j]; k < at.row_ptr[j + 1]; k++)
            fprintf(f, "    X%d  R%d  %.17g\n", j, at.col_ind[k], at.values[k]);
    }
    fprintf(f, "RHS\n    RHS1  OBJ  %.17g\n", -p->objective_offset);
    for (int i = 0; i < p->A.rows; i++)
        if (isfinite(p->row_lower[i]) || isfinite(p->row_upper[i]))
            fprintf(f, "    RHS1  R%d  %.17g\n", i,
                    isfinite(p->row_upper[i]) ? p->row_upper[i] : p->row_lower[i]);
    int ranges = 0;
    for (int i = 0; i < p->A.rows; i++)
        if (isfinite(p->row_lower[i]) && isfinite(p->row_upper[i]) &&
            p->row_lower[i] != p->row_upper[i]) {
            if (!ranges) {
                fprintf(f, "RANGES\n");
                ranges = 1;
            }
            fprintf(f, "    RNG1  R%d  %.17g\n", i, p->row_upper[i] - p->row_lower[i]);
        }
    fprintf(f, "BOUNDS\n");
    for (int j = 0; j < p->A.cols; j++) {
        if (p->lower[j] == p->upper[j])
            fprintf(f, " FX BND1  X%d  %.17g\n", j, p->lower[j]);
        else {
            if (isfinite(p->lower[j]))
                fprintf(f, " LO BND1  X%d  %.17g\n", j, p->lower[j]);
            else
                fprintf(f, " MI BND1  X%d\n", j);
            if (isfinite(p->upper[j]))
                fprintf(f, " UP BND1  X%d  %.17g\n", j, p->upper[j]);
            else
                fprintf(f, " PL BND1  X%d\n", j);
        }
    }
    fprintf(f, "QUADOBJ\n");
    for (int i = 0; i < p->Q.rows; i++)
        for (int k = p->Q.row_ptr[i]; k < p->Q.row_ptr[i + 1]; k++)
            if (p->Q.col_ind[k] <= i)
                fprintf(f, "    X%d  X%d  %.17g\n", i, p->Q.col_ind[k], p->Q.values[k]);
    fprintf(f, "ENDATA\n");
    matrix_free(&at);
    int failed = ferror(f);
    if (fclose(f))
        failed = 1;
    set_message(msg, cap, failed ? "Error writing MPS output." : "");
    return failed ? IO_ERROR : OK;
}
