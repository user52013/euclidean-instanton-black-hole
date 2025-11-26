# scan_SE.py
#
# This script scans S_E(M, delta) for several values of M and delta
# using the instanton_solver.py module.
#
# Usage:
#     python scan_SE.py > scan_output.txt

import numpy as np
from instanton_solver import find_shooting_solution, solve_instanton, compute_actions

# Masses to scan
M_list = [10, 20, 30, 40, 50, 60]

# Polymer parameters to scan
delta_list = [0.0, 0.02, 0.05, 0.10]

print("# M, delta, S_E, S_bulk")

for M in M_list:
    for delta in delta_list:
        print(f"\n[Running] M={M}, delta={delta}")

        # Find p_b(0)
        p_b0 = find_shooting_solution(M, delta)

        # Solve trajectory
        sol = solve_instanton(M, delta, p_b0)

        # Compute Euclidean actions
        S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

        # Print in machine-friendly format
        print(f"{M}, {delta}, {S_E}, {S_bulk}")
