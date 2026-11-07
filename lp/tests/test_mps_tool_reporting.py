#!/usr/bin/env python3
"""Check CLI reporting, solution input validation and export failures."""
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile

model = """NAME BOUNDS
ROWS
 N OBJ
 L R0
 L R1
COLUMNS
 X0 OBJ 1 R0 1
 X0 R1 1
 X1 OBJ 1 R0 1
 X1 R1 -1
RHS
 RHS1 R0 5 R1 2
BOUNDS
 LO BND X0 0
 UP BND X0 10
 LO BND X1 0
 UP BND X1 10
ENDATA
"""
with tempfile.TemporaryDirectory(prefix="mps-tool-report-") as directory:
    path = Path(directory) / "bounds.mps"
    path.write_text(model)
    for binary in sys.argv[1:]:
        help_result = subprocess.run([binary, "--help"], text=True, capture_output=True, timeout=10)
        assert help_result.returncode == 0 and "--gpu" in help_result.stdout, help_result.stderr
        # Only device zero is visible under the test runner. All LP tools must
        # honor --gpu instead of silently executing on the default device.
        args = [binary, str(path)] + ([] if binary == sys.argv[1] else [str(Path(directory) / "unused")])
        invalid = subprocess.run(args + ["--gpu", "2147483647"], text=True, capture_output=True, timeout=10)
        assert invalid.returncode == 2 and "select CUDA device" in invalid.stderr, invalid.stdout + invalid.stderr
    for enabled in (False, True):
        env = os.environ.copy()
        result = subprocess.run(
            [sys.argv[1], str(path), "--gpu", "0", "--folding", "false", "--max-iters", "2", "--presolve", str(enabled).lower()],
            env=env, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        fields = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
        assert fields["original_rows"] == fields["original_cols"] == "2", result.stdout
        if not enabled:
            assert fields["reduced_rows"] == fields["reduced_cols"] == "2", result.stdout
        assert fields["changed"] == str(int(enabled)), result.stdout
        assert fields["fixed_columns"] == "unavailable", result.stdout

    solution = Path(directory) / "solution"
    solution.mkdir()
    vectors = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [1.0, 1.0]}

    def write_solution():
        for name, values in vectors.items():
            (solution / (name + "_f64.bin")).write_bytes(struct.pack("2d", *values))

    def check_solution(expected, message=""):
        result = subprocess.run(
            [sys.argv[2], str(path), str(solution), "--presolve", "false", "--folding", "false"],
            text=True, capture_output=True, timeout=60)
        assert result.returncode == expected, result.stdout + result.stderr
        assert message in result.stderr, result.stderr

    write_solution()
    check_solution(0)
    for name in vectors:
        for invalid in (float("nan"), float("inf"), -float("inf")):
            write_solution()
            (solution / (name + "_f64.bin")).write_bytes(struct.pack("2d", invalid, 0.0))
            check_solution(2, "nonfinite")
        for length in (0, 8, 15, 24):
            write_solution()
            (solution / (name + "_f64.bin")).write_bytes(bytes(length))
            check_solution(2, "unexpected binary vector size")
    write_solution()
    (solution / "x_f64.bin").write_bytes(struct.pack("2d", 20.0, 0.0))
    check_solution(1)

    # Linux /dev/full reliably reports write failures without filling a disk.
    for failed_file in ("meta.txt", "c_f64.bin"):
        output = Path(directory) / failed_file.replace(".", "_")
        output.mkdir()
        (output / failed_file).symlink_to("/dev/full")
        result = subprocess.run(
            [sys.argv[3], str(path), str(output), "--presolve", "false", "--folding", "false"],
            text=True, capture_output=True, timeout=60)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "failed to write" in result.stderr, result.stderr
        assert "status ok" not in result.stdout, result.stdout
print("MPS reporting, invalid solution inputs and export failure checks passed")
