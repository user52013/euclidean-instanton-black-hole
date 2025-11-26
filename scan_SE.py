# scan_SE.py
#
# Scan S_E(M, delta) for several values of M and delta.
# Usage (inside the repo directory):
#     python scan_SE.py > scan_output.txt

import numpy as np
from instanton_solver import find_shooting_solution, solve_instanton, compute_actions

# 這裡先刻意避開 delta = 0，避免 0/0 的經典極限問題
# 之後如果真的需要 δ=0，我們再專門為 eom_tau 加一個 analytic limit 分支。
M_list = [20, 30, 40, 50, 60]
delta_list = [0.02, 0.05, 0.10]

print("# M, delta, S_E, S_bulk")

for M in M_list:
    for delta in delta_list:
        print(f"\n[Running] M={M}, delta={delta}")

        try:
            # 1) 射擊找 p_b(0)
            p_b0 = find_shooting_solution(M, delta)

            # 2) 積分 EOM
            sol = solve_instanton(M, delta, p_b0)

            # 3) 計算作用量
            S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

            # 4) 輸出方便後續分析的格式
            print(f"{M}, {delta}, {S_E}, {S_bulk}")

        except Exception as e:
            # 此點失敗就跳過，不要讓整個掃描終止
            print(f"# WARNING: failed for M={M}, delta={delta}: {type(e).__name__}: {e}")
            continue
