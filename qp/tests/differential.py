"""Optional independent audit: ctypes calls the native library; SciPy solves reduced QPs.
Not used by the library or its C-only build/tests. Requires NumPy and SciPy.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint, linprog, lsq_linear, minimize

PINT = C.POINTER(C.c_int)
PDOUBLE = C.POINTER(C.c_double)


class Matrix(C.Structure):
    _fields_ = [("rows", C.c_int), ("cols", C.c_int), ("nnz", C.c_int),
                ("row_ptr", PINT), ("col_ind", PINT), ("values", PDOUBLE)]


class Problem(C.Structure):
    _fields_ = [("Q", Matrix), ("A", Matrix)] + [
        (s, PDOUBLE) for s in ("c", "lower", "upper", "row_lower", "row_upper")
    ] + [("objective_offset", C.c_double)]


class Config(C.Structure):
    _fields_ = [("backend", C.c_int), ("device", C.c_int), ("max_iterations", C.c_int),
                ("rules", C.c_uint32)] + [(s, C.c_double) for s in
                ("max_seconds", "feasibility_tol", "bound_tol", "zero_tol")] + [
                ("max_substitution_support", C.c_int), ("max_q_fill", C.c_int),
                ("max_q_fill_ratio", C.c_double), ("monotone_empty_columns", C.c_int)]


class ExecutionInfo(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("passes", "batches", "reductions", "fallbacks")] + [("largest", C.c_size_t)]


class ExecutionOptions(C.Structure):
    _fields_ = [(name, C.c_int) for name in ("schedule", "batch_limit", "cleanup_rounds", "global_period")]


class Result(C.Structure):
    _fields_ = [("status", C.c_int), ("backend", C.c_int), ("reduced", Problem),
                ("state", C.c_void_p), ("presolve_seconds", C.c_double),
                ("iterations", C.c_int), ("reductions", C.c_int), ("message", C.c_char * 256)]


def pointer(a):
    return np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(PDOUBLE)


def vector(p, n):
    return np.ctypeslib.as_array(p, shape=(n,)).copy() if n else np.empty(0)


def dense(a):
    out = np.zeros((a.rows, a.cols))
    for i in range(a.rows):
        for k in range(a.row_ptr[i], a.row_ptr[i + 1]):
            out[i, a.col_ind[k]] = a.values[k]
    return out


def unpack(p):
    return (dense(p.Q), dense(p.A), vector(p.c, p.A.cols), vector(p.lower, p.A.cols),
            vector(p.upper, p.A.cols), vector(p.row_lower, p.A.rows),
            vector(p.row_upper, p.A.rows), p.objective_offset)


def solve(q, a, c, lower, upper, al, au, offset):
    n = len(c)
    if not n:
        assert np.all(al <= 1e-8) and np.all(au >= -1e-8)
        return np.empty(0), np.zeros(len(al)), np.empty(0), True
    scale = np.maximum(1., np.max(np.abs(a), axis=1)) if len(al) else np.empty(0)
    aa, ll, uu = a / scale[:, None], al / scale, au / scale
    eq = np.isfinite(ll) & np.isfinite(uu) & (np.abs(ll-uu) <= 1e-11*(1+np.maximum(np.abs(ll),np.abs(uu))))
    rows, rhs = [], []
    for i in range(len(al)):
        if not eq[i]:
            if np.isfinite(uu[i]): rows.append(aa[i]); rhs.append(uu[i])
            if np.isfinite(ll[i]): rows.append(-aa[i]); rhs.append(-ll[i])
    feasible = linprog(np.zeros(n), A_ub=np.array(rows).reshape(-1, n) if rows else None,
                      b_ub=np.array(rhs) if rows else None,
                      A_eq=aa[eq] if eq.any() else None, b_eq=ll[eq] if eq.any() else None,
                      bounds=list(zip(lower, upper)), method="highs")
    if not feasible.success:
        raise RuntimeError("reduced feasibility: " + feasible.message)
    if not np.any(q):
        optimum = linprog(c, A_ub=np.array(rows).reshape(-1, n) if rows else None,
                          b_ub=np.array(rhs) if rows else None,
                          A_eq=aa[eq] if eq.any() else None, b_eq=ll[eq] if eq.any() else None,
                          bounds=list(zip(lower, upper)), method="highs")
        if not optimum.success: raise RuntimeError("LP optimum: " + optimum.message)
        x = optimum.x
    else:
        # Eliminate equality and fixed-bound directions independently of the C
        # presolver. SLSQP otherwise falsely converges on degenerate faces.
        fixed = np.isfinite(lower) & np.isfinite(upper) & (np.abs(lower-upper) <= 1e-11*(1+np.maximum(np.abs(lower),np.abs(upper))))
        equality = np.vstack((aa[eq], np.eye(n)[fixed]))
        basis = null_space(equality) if len(equality) else np.eye(n)
        x0 = feasible.x
        if basis.shape[1] == 0:
            x = x0
        else:
            constraint_matrix = np.vstack((aa[~eq], np.eye(n)))
            constraint_lo = np.r_[ll[~eq], lower] - constraint_matrix@x0
            constraint_hi = np.r_[uu[~eq], upper] - constraint_matrix@x0
            directions = constraint_matrix@basis
            norms = np.max(np.abs(directions), axis=1)
            keep = (norms > 1e-12) & (np.isfinite(constraint_lo) | np.isfinite(constraint_hi))
            constraints = []
            if keep.any():
                constraints = [LinearConstraint(directions[keep]/norms[keep,None],
                                constraint_lo[keep]/norms[keep], constraint_hi[keep]/norms[keep])]
            hessian, gradient = basis.T@q@basis, basis.T@(q@x0+c)
            oscale = max(1., np.linalg.norm(hessian, ord=np.inf), np.linalg.norm(gradient, ord=np.inf))
            result = minimize(lambda w: (.5*w@hessian@w+gradient@w)/oscale,
                              np.zeros(basis.shape[1]), jac=lambda w: (hessian@w+gradient)/oscale,
                              method="SLSQP", constraints=constraints,
                              options={"ftol":1e-12,"maxiter":1000})
            if not result.success:
                result = minimize(lambda w: (.5*w@hessian@w+gradient@w)/oscale,
                                  np.zeros(basis.shape[1]), jac=lambda w: (hessian@w+gradient)/oscale,
                                  hess=lambda w: hessian/oscale, method="trust-constr",
                                  constraints=constraints,
                                  options={"gtol":1e-10,"xtol":1e-12,"maxiter":2000})
                if not result.success: raise RuntimeError("independent QP solve: " + result.message)
            x = x0+basis@result.x
    candidates, signs, owners, right_sides = [], [], [], []
    for kind, matrix, lo, hi in ((0, a, al, au), (1, np.eye(n), lower, upper)):
        for i, row in enumerate(matrix):
            value = row@x
            if np.isfinite(lo[i]) and lo[i] == hi[i]: sign = 0
            elif np.isfinite(lo[i]) and abs(value-lo[i]) < 3e-7*(1+abs(lo[i])): sign = 1
            elif np.isfinite(hi[i]) and abs(value-hi[i]) < 3e-7*(1+abs(hi[i])): sign = -1
            else: continue
            candidates.append(row); signs.append(sign); owners.append((kind, i))
            right_sides.append(.5*lo[i]+.5*hi[i] if sign == 0 else lo[i] if sign > 0 else hi[i])
    y, z, g = np.zeros(len(al)), np.zeros(n), q@x+c
    if candidates:
        b = np.array(candidates).T
        norms = np.maximum(1., np.max(np.abs(b), axis=0))
        # Independently polish the candidate active face before reconstructing
        # duals. Accept it only after checking all original reduced bounds.
        bb = b.T/norms[:,None]
        kkt = np.block([[q, -bb.T], [bb, np.zeros((len(bb),len(bb)))]])
        candidate = np.linalg.lstsq(kkt, np.r_[-c, np.array(right_sides)/norms], rcond=1e-12)[0][:n]
        tol = 1e-8
        if (np.all(candidate >= lower-tol) and np.all(candidate <= upper+tol) and
            np.all(a@candidate >= al-tol) and np.all(a@candidate <= au+tol) and
            .5*candidate@q@candidate+c@candidate <= .5*x@q@x+c@x+tol):
            x = candidate; g = q@x+c
        gscale = max(1., np.max(np.abs(g)))
        lb = np.array([0. if s == 1 else -np.inf for s in signs])
        ub = np.array([0. if s == -1 else np.inf for s in signs])
        dual = lsq_linear(b/norms, g/gscale, bounds=(lb, ub), method="bvls", tol=1e-13, max_iter=1000)
        for v, (kind, i) in zip(dual.x*gscale/norms, owners):
            (z if kind else y)[i] = v
    residual = np.max(np.abs(q@x+c-a.T@y-z)) / (1+np.max(np.abs(g)))
    return x, y, z, residual < 1e-6


def run(args):
    lib = C.CDLL(str(Path(args.library).resolve()))
    lib.default_config.restype = Config
    lib.problem_alloc.argtypes = [C.POINTER(Problem), C.c_int, C.c_int]
    lib.matrix_from_triplets.argtypes = [C.POINTER(Matrix), C.c_int, C.c_int, C.c_int, PINT, PINT, PDOUBLE]
    lib.presolve.argtypes = [C.POINTER(Problem), C.POINTER(Config), C.POINTER(Result)]
    if args.tiered:
        lib.default_execution_options.restype = ExecutionOptions
        lib.presolve_with_options.argtypes = [C.POINTER(Problem), C.POINTER(Config),
                                                  C.POINTER(ExecutionOptions), C.POINTER(Result)]
    execution_totals = dict(passes=0, batches=0, reductions=0, fallbacks=0, largest=0)
    lib.execution_info.argtypes = [C.c_void_p, C.POINTER(ExecutionInfo)]
    lib.postsolve.argtypes = [C.c_void_p, PDOUBLE, C.c_int, PDOUBLE, C.c_int, PDOUBLE, C.c_int, PDOUBLE, PDOUBLE, PDOUBLE]
    lib.problem_free.argtypes = [C.POINTER(Problem)]
    lib.result_free.argtypes = [C.POINTER(Result)]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    oracle = None
    if args.oracle:
        oracle_module = C.CDLL(str(Path(args.oracle).resolve()))
        oracle = oracle_module.test_oracle_solve
        oracle.argtypes = [C.POINTER(Problem), PDOUBLE, PDOUBLE, PDOUBLE]
        oracle.restype = C.c_int
    def enumerate_solution(problem):
        ox, oy, oz = np.empty(problem.A.cols), np.empty(problem.A.rows), np.empty(problem.A.cols)
        if oracle is None or not oracle(C.byref(problem), pointer(ox), pointer(oy), pointer(oz)):
            return None
        return ox, oy, oz, True
    count = certified = unresolved = fallback_count = 0
    unresolved_cases = []
    max_primal = max_objective = max_dual = 0.
    failures = []
    for case in range(args.cases):
        n, m = int(rng.integers(1, 9)), int(rng.integers(0, 9))
        a = rng.integers(-2, 3, size=(m, n)).astype(float)
        a[rng.random(a.shape) < .5] = 0.
        kind = case % 4
        if kind == 0:
            b = rng.normal(size=(n, n)); q = b.T@b + np.eye(n)
        elif kind == 1: q = np.diag(rng.uniform(.1, 3, size=n))
        elif kind == 2:
            v = rng.choice([-1., 1.], size=n); q = np.outer(v, v)
            if m: a[:] = rng.integers(-2, 3, size=(m, 1))*v
        else: q = np.zeros((n, n))
        q = (q+q.T)/2
        lower, upper = np.full(n, -2.), np.full(n, 2.)
        target = rng.integers(-4, 5, size=n)/4.
        zy = np.zeros(n)
        for j in range(n):
            side = int(rng.integers(0, 5))
            if kind == 2: continue
            if side == 1: target[j] = lower[j]; zy[j] = rng.uniform(.1, 2.)
            if side == 2: target[j] = upper[j]; zy[j] = -rng.uniform(.1, 2.)
            if side == 3: lower[j] = upper[j] = target[j]; zy[j] = rng.normal()
        ax = a@target
        al, au, yy = np.full(m, -np.inf), np.full(m, np.inf), np.zeros(m)
        for i in range(m):
            side = int(rng.integers(0, 5))
            if side == 0: al[i] = au[i] = ax[i]; yy[i] = rng.normal()
            if side == 1: al[i] = ax[i]; au[i] = ax[i]+1.; yy[i] = rng.uniform(.1, 2.)
            if side == 2: au[i] = ax[i]; al[i] = ax[i]-1.; yy[i] = -rng.uniform(.1, 2.)
            if side == 3: al[i] = ax[i]-1.; au[i] = ax[i]+1.
        c = -q@target + a.T@yy + zy
        p, result = Problem(), Result()
        try:
            assert lib.problem_alloc(C.byref(p), m, n) == 0
            for member, data in (("Q", q), ("A", a)):
                rr, cc = np.nonzero(data)
                rr, cc, vv = rr.astype(np.int32), cc.astype(np.int32), np.ascontiguousarray(data[rr, cc])
                assert lib.matrix_from_triplets(C.byref(getattr(p, member)), *data.shape,
                         len(vv), rr.ctypes.data_as(PINT), cc.ctypes.data_as(PINT), pointer(vv)) == 0
            for member, data in zip(("c", "lower", "upper", "row_lower", "row_upper"), (c, lower, upper, al, au)):
                if len(data): np.ctypeslib.as_array(getattr(p, member), shape=(len(data),))[:] = data
            config = lib.default_config()
            config.backend = 1
            all_rules = config.rules
            config.rules = all_rules if case % 2 else int(rng.integers(1, all_rules + 1))
            config.max_iterations, config.bound_tol, config.feasibility_tol = 20, 1e-10, 1e-10
            config.max_substitution_support = config.max_q_fill = -1
            config.max_q_fill_ratio = float("inf")
            config.monotone_empty_columns = 1
            if args.tiered:
                options = lib.default_execution_options()
                options.schedule = 1
                status = lib.presolve_with_options(C.byref(p), C.byref(config), C.byref(options), C.byref(result))
            else:
                status = lib.presolve(C.byref(p), C.byref(config), C.byref(result))
            assert status == 0, f"known finite optimum: native status={status}, rules={config.rules}"
            info = ExecutionInfo()
            assert lib.execution_info(result.state, C.byref(info)) == 0
            for key in ("passes", "batches", "reductions", "fallbacks"):
                execution_totals[key] += getattr(info, key)
            execution_totals["largest"] = max(execution_totals["largest"], info.largest)
            reduced = unpack(result.reduced)
            try:
                xr, yr, zr, dual_ok = solve(*reduced)
            except RuntimeError as error:
                replacement = enumerate_solution(result.reduced)
                if replacement is None:
                    unresolved_cases.append({"case":case,"error":str(error)})
                    unresolved += 1
                    continue
                xr, yr, zr, dual_ok = replacement
                fallback_count += 1
            if not dual_ok:
                replacement = enumerate_solution(result.reduced)
                if replacement is not None:
                    xr, yr, zr, dual_ok = replacement
                    fallback_count += 1
            x, y, z = np.empty(n), np.empty(m), np.empty(n)
            status = lib.postsolve(result.state, pointer(xr), len(xr), pointer(yr), len(yr),
                                      pointer(zr), len(zr), pointer(x), pointer(y), pointer(z))
            assert status == 0, f"postsolve status={status}"
            primal = max(np.max(np.maximum(lower-x, x-upper), initial=0.),
                         np.max(np.maximum(al-a@x, a@x-au), initial=0.)/(1+np.max(np.abs(a@x), initial=0.)))
            objective = .5*x@q@x+c@x
            expected = .5*target@q@target+c@target
            objective_error = abs(objective-expected)/(1+abs(expected))
            assert primal < 2e-6 and objective_error < 2e-6, f"primal={primal}, objective={objective_error}"
            max_primal, max_objective = max(max_primal, primal), max(max_objective, objective_error)
            count += 1
            if count % 200 == 0: print(f"checked {count}/{args.cases}", flush=True)
            if dual_ok:
                dual = np.max(np.abs(q@x+c-a.T@y-z))/(1+np.max(np.abs(c)))
                support = 0.
                for mult, value, lo, hi in zip(np.r_[y,z], np.r_[a@x,x], np.r_[al,lower], np.r_[au,upper]):
                    if abs(mult) < 1e-7: continue
                    bound = lo if mult > 0 else hi
                    support = max(support, abs(mult*(value-bound))/(1+abs(expected)))
                assert dual < 3e-6 and support < 3e-6, f"dual={dual}, complementarity={support}, rules={config.rules}"
                certified += 1; max_dual = max(max_dual, dual)
        except Exception as error:
            failures.append({"case": case, "error": str(error)})
            np.savez(output/f"failure-{case}.npz", Q=q, A=a, c=c, lower=lower, upper=upper,
                     row_lower=al, row_upper=au, target=target)
            if len(failures) >= 10: break
        finally:
            lib.result_free(C.byref(result)); lib.problem_free(C.byref(p))
    report = dict(requested=args.cases, primal_checked=count, dual_checked=certified,
                  oracle_unresolved=unresolved, unresolved_cases=unresolved_cases, failures=failures, max_primal=max_primal,
                  max_objective_error=max_objective, max_stationarity=max_dual, gpu=True,
                  seed=args.seed, tiered=args.tiered, enumeration_fallbacks=fallback_count, execution=execution_totals)
    (output/"differential.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return bool(failures) or count + unresolved != args.cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library"); parser.add_argument("output")
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--tiered", action="store_true")
    parser.add_argument("--oracle", help="Optional test-only active-set oracle module")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raise SystemExit(run(parser.parse_args()))
