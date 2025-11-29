# scan_SE.py
#
# Scan S_E(M, delta) for several values of M and delta.
# Usage (inside the repo directory):
#     python scan_SE.py > scan_output.txt

import numpy as np
from instanton_solver import find_shooting_solution, solve_instanton, compute_actions

M_list = [20, 30, 40, 50, 60]
delta_list = [0.02, 0.05, 0.10]

print("# M, delta, S_E, S_bulk")

for M in M_list:
    for delta in delta_list:
        print(f"\n[Running] M={M}, delta={delta}")

        try:
            # 1) 
            p_b0 = find_shooting_solution(M, delta)

            # 2) 
            sol = solve_instanton(M, delta, p_b0)

            # 3) 
            S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

            # 4) 
            print(f"{M}, {delta}, {S_E}, {S_bulk}")

        except Exception as e:
            print(f"# WARNING: failed for M={M}, delta={delta}: {type(e).__name__}: {e}")
            continue
