import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp, simps
from scipy.optimize import root_scalar

# -----------------------------------------------------
# Physical constants (Planck units: G = 1)
# -----------------------------------------------------
G = 1.0
gamma = 0.2375  # Barbero-Immirzi parameter


# -----------------------------------------------------
# Euclidean equations of motion (polymer-corrected KS interior)
# -----------------------------------------------------
def eom_tau(tau, u, M, delta):
    """
    Euclidean equations of motion for the polymer-corrected
    Kantowski–Sachs interior.

    State vector:
        u = [b_E, c_E, p_b, p_c]
    """
    bE, cE, p_b, p_c = u

    # db_E / dτ
    db_dtau = (1.0 / G) * (sin(delta * cE) / delta)

    # dc_E / dτ
    dc_dtau = (1.0 / G) * (bE * cos(delta * cE))

    # dp_b / dτ
    dpb_dtau = -(1.0 / (2.0 * G)) * sin(2.0 * delta * bE) / delta

    # dp_c / dτ
    # p_c stays strictly positive along the instanton trajectory.
    dpc_dtau = (1.0 / (2.0 * G)) * (p_b**2 / (p_c**2))

    return [db_dtau, dc_dtau, dpb_dtau, dpc_dtau]


# -----------------------------------------------------
# Horizon event: b_E = 2 G M
# -----------------------------------------------------
def horizon_event(tau, u, M, delta):
    """
    Event function that triggers when b_E reaches the
    horizon radius 2 G M.
    """
    bE = u[0]
    return bE - 2.0 * G * M


# Mark the event as terminal and only trigger when bE is increasing.
horizon_event.terminal = True
horizon_event.direction = +1.0


# -----------------------------------------------------
# Single instanton solution for given (M, delta)
# -----------------------------------------------------
def solve_instanton(M, delta, p_b0, tau_max=200.0):
    """
    Integrate the Euclidean EOM from the bounce to the horizon
    for given mass M, polymer scale delta and initial momentum p_b0.
    """
    # Bounce data (regularity conditions)
    b0 = gamma        # b_E(0) = gamma
    c0 = 0.0          # c_E(0) = 0
    p_c0 = gamma**2   # p_c(0) = gamma^2
    u0 = [b0, c0, p_b0, p_c0]

    sol = solve_ivp(
        fun=eom_tau,
        t_span=(0.0, tau_max),
        y0=u0,
        args=(M, delta),
        method="Radau",
        events=horizon_event,
        rtol=1e-10,
        atol=1e-12,
    )

    return sol


# -----------------------------------------------------
# Shooting residual: p_b(τ_H) - (2GM)^2
# -----------------------------------------------------
def shooting_residual(p_b0, M, delta):
    """
    Residual for the one-parameter shooting problem:
    enforce p_b(τ_H) = (2 G M)^2 at the horizon.
    """
    sol = solve_instanton(M, delta, p_b0)

    # If horizon event was not reached, penalize the residual.
    if len(sol.t_events) == 0 or len(sol.t_events[0]) == 0:
        return 1.0e3

    p_bH = sol.y[2, -1]
    target = (2.0 * G * M) ** 2
    return p_bH - target


def find_shooting_solution(M, delta, bracket=None):
    """
    Find the correct initial momentum p_b(0) by solving
    shooting_residual(p_b0, M, delta) = 0 with a 1D root finder.
    """
    if bracket is None:
        base = (2.0 * G * M) ** 2
        bracket = (0.1 * base, 2.0 * base)

    result = root_scalar(
        shooting_residual,
        args=(M, delta),
        bracket=bracket,
        method="brentq",
        xtol=1e-10,
        rtol=1e-10,
        maxiter=100,
    )

    if not result.converged:
        raise RuntimeError(
            f"Shooting did not converge for M={M}, delta={delta}."
        )

    return result.root


# -----------------------------------------------------
# Euclidean action: boundary expression + bulk check
# -----------------------------------------------------
def compute_actions(sol, M, delta, p_b0):
    """
    Compute the on-shell Euclidean action using the canonical
    boundary expression, and (optionally) a bulk integral as a check.

    Returns
    -------
    S_E_boundary : float
        On-shell Euclidean action from boundary data.
    S_bulk : float
        Canonical bulk integral ∫ (p_b db/dτ + p_c dc/dτ) dτ,
        useful as a numerical cross-check.
    """
    tau = sol.t
    bE, cE, p_b, p_c = sol.y

    # Boundary expression: S_E = (1 / (G gamma)) [p_b b_E]_0^{τ_H}
    bH = bE[-1]
    p_bH = p_b[-1]
    S_E_boundary = (1.0 / (G * gamma)) * (p_bH * bH - p_b0 * gamma)

    # Bulk check: recompute db/dτ, dc/dτ along the trajectory
    db_dtau = np.empty_like(bE)
    dc_dtau = np.empty_like(cE)

    for i, (ti, ui) in enumerate(zip(tau, sol.y.T)):
        db_dtau[i], dc_dtau[i], _, _ = eom_tau(ti, ui, M, delta)

    integrand = p_b * db_dtau + p_c * dc_dtau
    S_bulk = simps(integrand, tau)

    return S_E_boundary, S_bulk


# -----------------------------------------------------
# Parameter scan utility
# -----------------------------------------------------
def scan_parameters(M_list, delta_list, bracket=None):
    """
    Scan over a list of masses M and polymer scales delta.

    Returns
    -------
    results : dict
        Dictionary keyed by (M, delta) with entries:
            {
                "p_b0": ...,
                "S_E": ...,
                "S_bulk": ...,
                "bH": ...,
                "residual": ...,
                "status": "ok" or "failed"
            }
    """
    results = {}

    for M in M_list:
        for delta in delta_list:
            key = (M, delta)
            try:
                p_b0 = find_shooting_solution(M, delta, bracket=bracket)
                sol = solve_instanton(M, delta, p_b0)

                if len(sol.t_events) == 0 or len(sol.t_events[0]) == 0:
                    results[key] = {
                        "status": "failed_no_horizon",
                    }
                    continue

                S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

                bH = sol.y[0, -1]
                p_bH = sol.y[2, -1]
                target = (2.0 * G * M) ** 2
                res = p_bH - target

                results[key] = {
                    "status": "ok",
                    "p_b0": p_b0,
                    "S_E": S_E,
                    "S_bulk": S_bulk,
                    "bH": bH,
                    "residual": res,
                }

            except Exception as e:
                results[key] = {
                    "status": "failed_exception",
                    "error": str(e),
                }

    return results


# --------------------------------------------------------------------
# Example run
# --------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Example parameters
    M = 30.0
    # 確保 delta 設置為論文中使用的 0.05
    delta = 0.05 
    print(f"--- Euclidean instanton solver: M={M}, delta={delta} ---")
    
    # 初始化變量，防止在 try 失敗時報 NameError
    p_b0 = None
    
    try:
        # 這裡的代碼將執行核心計算，並捕捉所有可能的數值失敗
        
        # A. 尋找初始條件
        print("\n[Shooting] Attempting to find p_b(0) via shooting...")
        p_b0 = find_shooting_solution(M, delta) 
        
        # B. 求解 ODE
        print(f"  Found p_b(0) = {p_b0:.10e}")
        sol = solve_instanton(M, delta, p_b0)
        
        # C. 計算作用量
        S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

        # ----------------------------------------
        # D. 結果打印
        # ----------------------------------------
        if len(sol.t_events) == 0 or len(sol.t_events[0]) == 0:
            print("Warning: horizon event was not detected.")
        else:
            print("Horizon reached at τ_H =", sol.t_events[0][0])

        bH = sol.y[0, -1]
        p_bH = sol.y[2, -1]
        target = (2.0 * G * M) ** 2
        residual = p_bH - target

        print("\n[Shooting Summary]")
        print(f"  p_b(0)         = {p_b0:.10e}")
        print(f"  b_E(τ_H)       = {bH:.10e}")
        print(f"  p_b(τ_H)       = {p_bH:.10e}")
        print(f"  Target p_bH    = {target:.10e}")
        print(f"  Residual       = {residual:.3e}")

        print("\n[Action Summary]")
        print(f"  S_E (Total)    = {S_E:.10e}")
        print(f"  S_bulk (Check) = {S_bulk:.10e}")
        print(f"  |S_E - S_bulk| = {abs(S_E - S_bulk):.3e}")
        
        # ----------------------------------------
        # E. 數值驗證 (供 CI 系統使用)
        # ----------------------------------------
        EXPECTED_S_E = 11333.0 
        # 設置相對容差 (0.005%)，如果誤差小於此值則視為通過
        TOLERANCE = 5.0e-5 
        
        # 如果計算出的 S_E 在容差範圍內，則視為成功
        if np.abs(S_E - EXPECTED_S_E) / EXPECTED_S_E < TOLERANCE:
            print("\n[CI_VALIDATION] SUCCESS: Computed S_E matches expected value (within tolerance).")
            sys.exit(0) # 成功退出碼，CI 顯示 PASS
        else:
            print(f"\n[CI_VALIDATION] FAILED: Computed S_E {S_E:.10e} does not match expected {EXPECTED_S_E:.10e}")
            sys.exit(1) # 失敗退出碼，CI 顯示 FAIL

    except Exception as e:
        # D. 捕捉任何失敗 (包括 find_shooting_solution 內部的錯誤)
        print("\n[CRITICAL FAILURE] Test Run Failed.")
        print(f"Actual Error Message: {type(e).__name__}: {e}")
        
        # 讓 CI 流程知道測試失敗
        sys.exit(1)
