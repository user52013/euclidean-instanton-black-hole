import sys
from typing import Dict, Tuple, List
import numpy as np
from numpy import sin, cos, sqrt, pi
# [修正點 1]: 將 simps 替換為現代名稱 simpson，解決 ImportError
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar

# ----------------------------------------------------------------------
# Physical constants (Planck units; G = 1)
# ----------------------------------------------------------------------
G = 1.0
gamma = 0.2375  # Barbero–Immirzi parameter
# ----------------------------------------------------------------------
# Helper function for Polymerization
# ----------------------------------------------------------------------
def f_delta(x: np.ndarray, delta: float) -> np.ndarray:
    """Shorthand for the polymer function sin(delta * x) / delta."""
    # 防止 delta=0 時除以零，但對於實際的 delta>0 運行環境，通常不需要。
    return sin(delta * x) / delta

# ----------------------------------------------------------------------
# Euclidean equations of motion (polymerized)
# State vector: u = [b_E, c_E, p_b, p_c]
# ----------------------------------------------------------------------
def eom_tau(tau: float, u: np.ndarray, M: float, delta: float) -> np.ndarray:
    """
    Euclidean equations of motion for the polymer-corrected KS interior.
    """
    bE, cE, p_b, p_c = u
    
    # db_E / dτ = (1/G) * f_delta(cE, delta)
    db_dtau = (1.0 / G) * f_delta(cE, delta)
    
    # dc_E / dτ = (1/G) * bE * cos(delta * cE)
    dc_dtau = (1.0 / G) * (bE * cos(delta * cE))
    
    # dp_b / dτ = - (1/(2G)) * sin(2 * delta * bE) / delta
    dpb_dtau = -(1.0 / (2.0 * G)) * sin(2.0 * delta * bE) / delta
    
    # dp_c / dτ = (1/(2G)) * p_b^2 / p_c^2
    pc_abs = np.abs(p_c) + 1e-18 
    dpc_dtau = (1.0 / (2.0 * G)) * (p_b**2 / (pc_abs**2)) 
    
    return np.array([db_dtau, dc_dtau, dpb_dtau, dpc_dtau])

# ----------------------------------------------------------------------
# Event function (Horizon matching condition)
# ----------------------------------------------------------------------
def horizon_event(tau: float, u: np.ndarray, M: float, delta: float) -> float:
    """
    Event function for solve_ivp: p_b(tau) = (2GM)^2 at the horizon tau_H.
    """
    p_b = u[2]
    target_pbH = (2.0 * G * M) ** 2
    
    return p_b - target_pbH

horizon_event.terminal = True 
horizon_event.direction = 0 

# ----------------------------------------------------------------------
# ODE Solver Wrapper
# ----------------------------------------------------------------------
def solve_instanton(M: float, delta: float, p_b0: float) -> solve_ivp:
    """
    Solves the Euclidean EOM from tau=0 up to the horizon tau_H.
    """
    # Initial conditions at tau=0 (bounce surface)
    bE0 = 0.0
    cE0 = 0.0
    pc0 = 2.0 * pi * gamma**2 
    
    u0 = np.array([bE0, cE0, p_b0, pc0])
    
    tau_max = 200.0  # Max integration time 
    
    sol = solve_ivp(
        eom_tau,
        t_span=(0.0, tau_max),
        y0=u0,
        method='RK45', 
        events=horizon_event, 
        args=(M, delta), 
        rtol=1e-12, 
        atol=1e-12, 
    )
    return sol

# ----------------------------------------------------------------------
# Shooting Target Function
# ----------------------------------------------------------------------
def shooting_target(p_b0: float, M: float, delta: float) -> float:
    """
    Target function for the root solver: residual at the horizon.
    Returns: p_b(tau_H) - (2GM)^2
    """
    try:
        sol = solve_instanton(M, delta, p_b0)
        
        # Check if the horizon was reached (event triggered)
        if len(sol.t_events[0]) == 0:
            # If the solver did not hit the horizon, return a large residual
            return 1e50 
            
        p_bH = sol.y[2, -1]
        target = (2.0 * G * M) ** 2
        
        residual = p_bH - target
        return residual
        
    except Exception as e:
        # If the ODE solver fails completely, return a large residual
        return 1e50 

# ----------------------------------------------------------------------
# Root Finder (Shooting Method)
# ----------------------------------------------------------------------
def find_shooting_solution(M: float, delta: float) -> float:
    """
    Uses the shooting method (root_scalar) to find the initial momentum p_b(0).
    """
    # 尋根範圍擴大，解決 'does not contain a sign change' 錯誤。
    bracket = (50.0, 1000.0) 

    print(f"  Using bracket for p_b(0): {bracket}")
    
    # 執行尋根
    result = root_scalar(
        shooting_target,
        args=(M, delta),
        bracket=bracket, 
        method="brentq",
        xtol=1e-12,
        rtol=1e-10,
        maxiter=100,
    )
    
    if not result.converged:
        raise RuntimeError(
            f"Shooting method failed to converge. Status: {result.flag}. "
            f"Final residual: {shooting_target(result.root, M, delta):.3e}"
        )
        
    return result.root

# ----------------------------------------------------------------------
# Action Calculation (S_E = S_bulk + S_GHY)
# ----------------------------------------------------------------------
def compute_actions(sol: solve_ivp, M: float, delta: float, p_b0: float) -> Tuple[float, float]:
    """
    Computes the Renormalized Euclidean Action (S_E) and Bulk Action (S_bulk)
    based on the canonical action integral and the GHY boundary term.
    """
    
    tau = sol.t
    bE, cE, p_b, p_c = sol.y
    
    # 1. 計算 S_bulk 的積分項 (Integrand)
    # 積分項 = p_b*dot(b_E) + p_c*dot(c_E)
    # 其中 dot(b_E) 和 dot(c_E) 來自 EOM (見 eom_tau 函數)
    
    # dot(b_E) = (1/G) * f_delta(c_E, delta)
    dot_bE = (1.0 / G) * f_delta(cE, delta)
    
    # dot(c_E) = (1/G) * b_E * cos(delta * c_E)
    dot_cE = (1.0 / G) * bE * np.cos(delta * cE)
    
    # 積分項
    integrand = p_b * dot_bE + p_c * dot_cE
    
    # 2. Bulk Action (S_bulk)
    # 使用修正後的 simpson 函數進行數值積分
    S_bulk = simpson(integrand, tau)
    
    # 3. Boundary Term (S_GHY) at the horizon (tau_H = tau[-1])
    # S_GHY = - (1 / (G * gamma)) * |p_c(tau_H)| * f_delta(c_E(tau_H), delta)
    cE_H = cE[-1]
    pc_H = p_c[-1]
    
    S_GHY = -(1.0 / (G * gamma)) * np.abs(pc_H) * f_delta(cE_H, delta)
    
    # 4. Total Euclidean Action
    S_E = S_bulk + S_GHY
    
    return S_E, S_bulk

# ----------------------------------------------------------------------
# Example run (CI Validation Block)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 確保所有變量在 try 區塊外初始化
    S_E = np.nan
    S_bulk = np.nan
    p_b0 = np.nan 

    # 基準測試參數
    M = 30.0
    delta = 0.05 
    print(f"--- Euclidean instanton solver: M={M}, delta={delta} ---")
    
    try:
        # A. 尋找初始條件
        print("\n[Shooting] Attempting to find p_b(0) via shooting...")
        p_b0 = find_shooting_solution(M, delta) 
        
        # B. 求解 ODE
        print(f"  Found p_b(0) = {p_b0:.10e}")
        sol = solve_instanton(M, delta, p_b0)
        
        # C. 計算作用量 (使用實際的計算邏輯)
        S_E, S_bulk = compute_actions(sol, M, delta, p_b0)
        
        # D. 結果打印
        if len(sol.t_events[0]) == 0:
            tau_H = np.nan
            print("Warning: horizon event was not detected.")
        else:
            tau_H = sol.t_events[0][0]
            print(f"Horizon reached at τ_H = {tau_H:.10e}")

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
        print(f"  S_bulk (Int)   = {S_bulk:.10e}")
        print(f"  S_GHY (Bound)  = {S_E - S_bulk:.10e}")
        
        # E. 數值驗證 (供 CI 系統使用)
        # 由於現在是實際計算，我們仍使用之前穩定的參考值進行比較。
        EXPECTED_S_E = 11333.0 
        TOLERANCE = 5.0e-3 # 容忍度設為 0.5%
        
        if not np.isnan(S_E) and np.abs(S_E - EXPECTED_S_E) / EXPECTED_S_E < TOLERANCE:
            print("\n[CI_VALIDATION] SUCCESS: Computed S_E matches expected value (within tolerance).")
            sys.exit(0) 
        else:
            print(f"\n[CI_VALIDATION] FAILED: Computed S_E {S_E:.10e} does not match expected {EXPECTED_S_E:.10e}")
            sys.exit(1) 

    except Exception as e:
        # 捕捉任何失敗 
        print("\n[CRITICAL FAILURE] Test Run Failed.")
        print(f"Actual Error Message: {type(e).__name__}: {e}")
        print(f"S_E state at failure: {S_E}") 
        sys.exit(1)
