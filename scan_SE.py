# scan_SE.py
#
# Scan S_E(M, delta) using the paper-consistent instanton_solver.
# Usage:
#   python scan_SE.py > scan_output.txt

import numpy as np

from instanton_solver import (
    find_pb0_by_bracketing,
    solve_instanton,
    compute_actions_and_diagnostics,
)

# Avoid delta = 0 to sidestep analytic limit issues
M_list = [20, 30, 40, 50, 60]
delta_list = [0.02, 0.05, 0.10]

print("# M, delta, S_E, S_bulk, max|C|")

for M in M_list:
    for delta in delta_list:
        print(f"\n[Running] M={M}, delta={delta}")

        try:
            # 1) shooting for p_b(0)
            p_b0 = find_pb0_by_bracketing(
                M=M,
                delta_b=delta,
                delta_c=delta,
                tau_max=200.0,
                rtol=1e-10,
                atol=1e-12,
                gamma=0.2375,
                G=1.0,
            )

            # 2) integrate EOM
            inst = solve_instanton(
                M=M,
                delta_b=delta,
                delta_c=delta,
                p_b0=p_b0,
                tau_max=200.0,
                rtol=1e-10,
                atol=1e-12,
                gamma=0.2375,
                G=1.0,
            )

            # 3) compute action + diagnostics
            res = compute_actions_and_diagnostics(
                inst,
                delta_b=delta,
                delta_c=delta,
                gamma=0.2375,
                G=1.0,
                sample_N=1500,
            )

            print(f"{M}, {delta}, {res.S_E}, {res.S_bulk}, {res.max_constraint_abs}")

        except Exception as e:
            print(f"# WARNING: failed for M={M}, delta={delta}: {type(e).__name__}: {e}")
            continue
