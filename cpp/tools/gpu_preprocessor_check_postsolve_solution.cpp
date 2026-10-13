#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver_tool_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

class Fnv1a64 {
 public:
  template <typename T>
  void add_scalar(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    add_bytes(&value, sizeof(value));
  }

  template <typename T>
  void add_vector(const std::vector<T>& values) {
    const std::uint64_t size = static_cast<std::uint64_t>(values.size());
    add_scalar(size);
    if (!values.empty()) {
      add_bytes(values.data(), values.size() * sizeof(T));
    }
  }

  void add_string(const std::string& value) {
    const std::uint64_t size = static_cast<std::uint64_t>(value.size());
    add_scalar(size);
    add_bytes(value.data(), value.size());
  }

  std::uint64_t value() const { return hash_; }

 private:
  void add_bytes(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
      hash_ ^= static_cast<std::uint64_t>(bytes[i]);
      hash_ *= UINT64_C(1099511628211);
    }
  }

  std::uint64_t hash_ = UINT64_C(14695981039346656037);
};

template <typename T>
std::uint64_t raw_vector_hash(const std::vector<T>& values) {
  Fnv1a64 hash;
  hash.add_vector(values);
  return hash.value();
}

std::uint64_t record_hash(const gpu_presolver::presolve::PresolveRecordGpu& record) {
  Fnv1a64 hash;
  hash.add_scalar(record.m0);
  hash.add_scalar(record.n0);
  hash.add_scalar(record.m1);
  hash.add_scalar(record.n1);
  hash.add_vector(record.row_org2red);
  hash.add_vector(record.row_red2org);
  hash.add_vector(record.col_org2red);
  hash.add_vector(record.col_red2org);
  hash.add_vector(record.fixed_idx);
  hash.add_vector(record.fixed_val);
  hash.add_vector(record.removed_row_idx);
  hash.add_vector(record.removed_col_idx);
  hash.add_scalar(record.obj_constant_old);
  hash.add_scalar(record.obj_constant_new);
  hash.add_scalar(static_cast<std::uint8_t>(
      record.has_primal_only_variable_pair_aggregation_reduction));
  hash.add_scalar(static_cast<std::uint8_t>(
      record.has_primal_only_column_domination_reduction));
  hash.add_scalar(static_cast<std::uint8_t>(
      record.has_primal_only_projected_auxiliary_reduction));

  const std::uint64_t recovery_count =
      static_cast<std::uint64_t>(record.absolute_value_epigraph_recoveries.size());
  hash.add_scalar(recovery_count);
  for (const auto& recovery : record.absolute_value_epigraph_recoveries) {
    hash.add_string(recovery.pattern);
    const std::uint64_t split_count = static_cast<std::uint64_t>(recovery.splits.size());
    hash.add_scalar(split_count);
    for (const auto& split : recovery.splits) {
      hash.add_scalar(split.t_col);
      hash.add_scalar(split.e_col);
      hash.add_scalar(split.rho);
    }
    const std::uint64_t outer_count = static_cast<std::uint64_t>(recovery.outer_pairs.size());
    hash.add_scalar(outer_count);
    for (const auto& outer : recovery.outer_pairs) {
      hash.add_scalar(outer.bound_col);
      hash.add_scalar(outer.free_col);
    }
    const std::uint64_t linked_count =
        static_cast<std::uint64_t>(recovery.linked_slacks.size());
    hash.add_scalar(linked_count);
    for (const auto& linked : recovery.linked_slacks) {
      hash.add_scalar(linked.slack_col);
      hash.add_scalar(linked.t_col);
      hash.add_scalar(linked.factor);
    }
    const std::uint64_t max_count = static_cast<std::uint64_t>(recovery.max_slacks.size());
    hash.add_scalar(max_count);
    for (const auto& max_slack : recovery.max_slacks) {
      hash.add_scalar(max_slack.slack_col);
      hash.add_vector(max_slack.t_cols);
      hash.add_vector(max_slack.factors);
    }
  }

  const std::uint64_t variable_pair_shifted_antipodal_recovery_count =
      static_cast<std::uint64_t>(
          record.variable_pair_aggregation_recoveries.size());
  hash.add_scalar(variable_pair_shifted_antipodal_recovery_count);
  for (const auto& recovery : record.variable_pair_aggregation_recoveries) {
    hash.add_vector(recovery.plus_cols);
    hash.add_vector(recovery.minus_cols);
    hash.add_vector(recovery.root_plus_cols);
    hash.add_vector(recovery.root_minus_cols);
    hash.add_vector(recovery.plus_shift);
    hash.add_vector(recovery.minus_shift);
  }

  const std::uint64_t checkpoint_count = static_cast<std::uint64_t>(
      record.primal_recovery_timeline.size());
  hash.add_scalar(checkpoint_count);
  for (const auto& checkpoint : record.primal_recovery_timeline) {
    hash.add_scalar(static_cast<std::uint8_t>(checkpoint.kind));
    hash.add_scalar(checkpoint.payload_index);
    hash.add_scalar(checkpoint.tape_position);
  }

  hash.add_vector(record.tape.types);
  hash.add_vector(record.tape.index_starts);
  hash.add_vector(record.tape.value_starts);
  hash.add_vector(record.tape.dual_modes);
  hash.add_vector(record.tape.indices);
  hash.add_vector(record.tape.vals);
  return hash.value();
}

double finite_bound_norm_l2(const std::vector<double>& lower,
                            const std::vector<double>& upper) {
  double sum_sq = 0.0;
  for (const double v : lower) {
    if (std::isfinite(v)) {
      sum_sq += v * v;
    }
  }
  for (const double v : upper) {
    if (std::isfinite(v)) {
      sum_sq += v * v;
    }
  }
  return 1.0 + std::sqrt(sum_sq);
}

struct PrimalFeasibility {
  double normalized = 0.0;
  double row_l2 = 0.0;
  double row_max = 0.0;
  double col_l2 = 0.0;
  double col_max = 0.0;
};

struct ViolationDetail {
  int index = -1;
  double violation = 0.0;
  double value = 0.0;
  double lower = 0.0;
  double upper = 0.0;
};

void keep_top_violation(std::vector<ViolationDetail>& top, ViolationDetail item, std::size_t limit) {
  if (item.violation <= 0.0 || limit == 0) {
    return;
  }
  top.push_back(item);
  std::sort(top.begin(), top.end(), [](const ViolationDetail& lhs, const ViolationDetail& rhs) {
    if (lhs.violation != rhs.violation) {
      return lhs.violation > rhs.violation;
    }
    return lhs.index < rhs.index;
  });
  if (top.size() > limit) {
    top.resize(limit);
  }
}

PrimalFeasibility compute_primal_feasibility(const cpu_presolve::LpModel& model,
                                             const std::vector<double>& x) {
  const cpu_presolve::CsrMatrix& A = model.csr();
  const std::vector<double>& AL = model.row_lower();
  const std::vector<double>& AU = model.row_upper();
  const std::vector<double>& l = model.col_lower();
  const std::vector<double>& u = model.col_upper();

  PrimalFeasibility out;
  double row_sq = 0.0;
  for (int i = 0; i < A.rows(); ++i) {
    double activity = 0.0;
    for (NnzOffset p = A.row_ptr()[static_cast<std::size_t>(i)];
         p < A.row_ptr()[static_cast<std::size_t>(i + 1)];
         ++p) {
      const int col = A.col_idx()[static_cast<std::size_t>(p)];
      activity += A.values()[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(col)];
    }
    const double violation = std::max({0.0, AL[static_cast<std::size_t>(i)] - activity,
                                       activity - AU[static_cast<std::size_t>(i)]});
    row_sq += violation * violation;
    out.row_max = std::max(out.row_max, violation);
  }
  out.row_l2 = std::sqrt(row_sq);

  double col_sq = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) {
    const double violation = std::max({0.0, l[j] - x[j], x[j] - u[j]});
    col_sq += violation * violation;
    out.col_max = std::max(out.col_max, violation);
  }
  out.col_l2 = std::sqrt(col_sq);
  out.normalized = std::max(out.row_l2, out.col_l2) / finite_bound_norm_l2(AL, AU);
  return out;
}

void print_top_violations(const cpu_presolve::LpModel& model,
                          const std::vector<double>& x,
                          std::size_t limit) {
  const cpu_presolve::CsrMatrix& A = model.csr();
  const std::vector<double>& AL = model.row_lower();
  const std::vector<double>& AU = model.row_upper();
  const std::vector<double>& l = model.col_lower();
  const std::vector<double>& u = model.col_upper();

  std::vector<ViolationDetail> top_rows;
  std::vector<ViolationDetail> top_cols;
  for (int i = 0; i < A.rows(); ++i) {
    double activity = 0.0;
    for (NnzOffset p = A.row_ptr()[static_cast<std::size_t>(i)];
         p < A.row_ptr()[static_cast<std::size_t>(i + 1)];
         ++p) {
      const int col = A.col_idx()[static_cast<std::size_t>(p)];
      activity += A.values()[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(col)];
    }
    const double violation = std::max({0.0, AL[static_cast<std::size_t>(i)] - activity,
                                       activity - AU[static_cast<std::size_t>(i)]});
    keep_top_violation(top_rows,
                       ViolationDetail{i, violation, activity, AL[static_cast<std::size_t>(i)],
                                       AU[static_cast<std::size_t>(i)]},
                       limit);
  }

  for (std::size_t j = 0; j < x.size(); ++j) {
    const double violation = std::max({0.0, l[j] - x[j], x[j] - u[j]});
    keep_top_violation(top_cols,
                       ViolationDetail{static_cast<int>(j), violation, x[j], l[j], u[j]},
                       limit);
  }

  for (std::size_t k = 0; k < top_rows.size(); ++k) {
    const ViolationDetail& item = top_rows[k];
    std::cout << "top_row_violation_" << (k + 1) << " "
              << item.index << "," << item.violation << "," << item.value << ","
              << item.lower << "," << item.upper << "\n";
  }
  for (std::size_t k = 0; k < top_cols.size(); ++k) {
    const ViolationDetail& item = top_cols[k];
    std::cout << "top_col_violation_" << (k + 1) << " "
              << item.index << "," << item.violation << "," << item.value << ","
              << item.lower << "," << item.upper << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = gpu_presolver::tools::parse_command_line(argc, argv, 2, true);
    if (options.help) {
      gpu_presolver::tools::print_usage(std::cout, "gpu_preprocessor_check_postsolve_solution",
                                       "<model.mps|-> <solution_dir>", true);
      return 0;
    }
    const std::string input_path = options.positional[0];
    const std::filesystem::path solution_dir = options.positional[1];
    const auto& params = options.params;
    cpu_presolve::MpsModel model = gpu_presolver::tools::read_mps_file(input_path);
    gpu_presolver::tools::DeviceLpOwner device_lp = gpu_presolver::tools::upload_lp(model.lp);

    gpu_presolver::tools::FoldingPipelineOwner pipeline;
    pipeline.summary = gpu_presolver::folding::run_gpu_presolve_with_folding(
        device_lp.lp, params, false, true);
    gpu_presolver::presolve::export_recovery_record(pipeline.summary.presolve.record);
    const auto& summary = pipeline.summary.presolve;
    if (summary.has_infeasible || summary.has_unbounded) {
      throw std::runtime_error("no reduced solution for infeasible/unbounded model");
    }

    const std::vector<double> x_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "x_f64.bin",
                                                         static_cast<std::size_t>(summary.reduced_cols));
    const std::vector<double> y_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "y_f64.bin",
                                                         static_cast<std::size_t>(summary.reduced_rows));
    const std::vector<double> z_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "z_f64.bin",
                                                         static_cast<std::size_t>(summary.reduced_cols));

    double* x_red = gpu_presolver::tools::copy_to_device(x_red_h);
    double* y_red = gpu_presolver::tools::copy_to_device(y_red_h);
    double* z_red = gpu_presolver::tools::copy_to_device(z_red_h);

    const auto unfolded = gpu_presolver::folding::postsolve_and_unfold_to_host(
        x_red, y_red, z_red, pipeline.summary, &device_lp.lp);
    const auto& x_org = unfolded.x;
    const auto& y_org = unfolded.y;
    const auto& z_org = unfolded.z;
    const PrimalFeasibility feas = compute_primal_feasibility(model.lp, x_org);

    std::cout << "status " << (summary.has_infeasible ? "infeasible" : (summary.has_unbounded ? "unbounded" : "ok")) << "\n";
    std::cout << "reduced_rows " << summary.reduced_rows << "\n";
    std::cout << "reduced_cols " << summary.reduced_cols << "\n";
    std::cout << "primal_feas_l2_normalized " << feas.normalized << "\n";
    std::cout << "row_violation_l2 " << feas.row_l2 << "\n";
    std::cout << "row_violation_max " << feas.row_max << "\n";
    std::cout << "col_violation_l2 " << feas.col_l2 << "\n";
    std::cout << "col_violation_max " << feas.col_max << "\n";
    std::cout << "record_hash " << record_hash(summary.record) << "\n";
    std::cout << "x_org_hash " << raw_vector_hash(x_org) << "\n";
    std::cout << "y_org_hash " << raw_vector_hash(y_org) << "\n";
    std::cout << "z_org_hash " << raw_vector_hash(z_org) << "\n";
    if (options.print_top_violations) {
      print_top_violations(model.lp, x_org, 10);
    }

    cudaFree(x_red);
    cudaFree(y_red);
    cudaFree(z_red);
    return feas.normalized <= 1.0e-6 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
