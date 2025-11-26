import sys
import numpy as np
from numpy import sin, cos, sqrt, pi
# [修正]: 使用 simpson 替代 simps 以兼容新版 SciPy
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar

# ----------------------------------------------------------------------
# Physical constants (Planck units; G = 1)
# ----------------------------------------------------------------------
G = 1.0
gamma = 0.2375  # Barbero–Immirzi parameter

# ----------------------------------------------------------------------
# Euclidean equations of motion
# ----------------------------------------------------------------------
def eom_tau(tau: float, u: np.ndarray, M: float, delta: float) -> np.ndarray:
    """
    Euclidean equations of motion for the polymer-corrected KS interior.
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
    # Guard against division by zero if p_c gets too small
    if p_c <= 1e-14:
         dpc_dtau = 0.0
    else:
         dpc_dtau = (1.0 / (2.0 * G)) * (p_b**2 / (p_c**2))
    
    return np.array([db_dtau, dc_dtau, dpb_dtau, dpc_dtau], dtype=float)

# ----------------------------------------------------------------------
# Event function (Horizon matching condition)
# ----------------------------------------------------------------------
def horizon_event(tau: float, u: np.ndarray, M: float, delta: float) -> float:
    """
    Event function for solve_ivp: triggers when b_E - 2GM = 0.
    """
    bE = u[0]
    return bE - 2.0 * G * M

horizon_event.terminal = True 
horizon_event.direction = 1 # Trigger when crossing from negative to positive

# ----------------------------------------------------------------------
# ODE Solver Wrapper
# ----------------------------------------------------------------------
def solve_instanton(M: float, delta: float, p_b0: float) -> solve_ivp:
    """
    Solves the Euclidean EOM from tau=0 up to the horizon.
    """
    # Initial conditions at tau=0 (bounce surface)
    bE0 = gamma
    cE0 = 0.0
    pc0 = gamma**2 
    # u0 = [b_E, c_E, p_b, p_c]
    u0 = np.array([bE0, cE0, p_b0, pc0], dtype=float)
    
    tau_max = 200.0
    
    sol = solve_ivp(
        eom_tau,
        t_span=(0.0, tau_max),
        y0=u0,
        method='Radau', # Radau is robust for stiff systems
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
    Residual function for shooting: p_b(tau_H) - (2GM)^2.
    """
    try:
        sol = solve_instanton(M, delta, p_b0)
        
        # If horizon not reached
        if len(sol.t_events[0]) == 0:
            # Use distance to b=2GM as proxy or large penalty
            bH = sol.y[0, -1]
            return bH - (2.0 * G * M)
            
        p_bH = sol.y[2, -1]
        target = (2.0 * G * M) ** 2
        return p_bH - target
        
    except Exception:
        return 1e50

# ----------------------------------------------------------------------
# Automatic Bracket Finder (Crucial Fix)
# ----------------------------------------------------------------------
def find_bracket(M: float, delta: float, num: int = 50) -> Tuple[float, float]:
    """
    Automatically search for a bracket [p_left, p_right] with a sign change.
    Using a logarithmic grid because p_b0 can be large.
    """
    target_pb = (2.0 * G * M)**2
    # Scan from small value up to 10 times the target p_b
    p_min = 1.0
    p_max = 10.0 * target_pb 
    
    print(f"  [Auto-Bracket] Scanning range [{p_min:.1e}, {p_max:.1e}] for sign change...")
    
    grid = np.logspace(np.log10(p_min), np.log10(p_max), num=num)
    residuals = []
    
    for p in grid:
        r = shooting_target(p, M, delta)
        residuals.append(r)
    
    residuals = np.array(residuals)
    
    # Check for sign change
    for i in range(len(grid) - 1):
        r1 = residuals[i]
        r2 = residuals[i+1]
        
        if np.isnan(r1) or np.isnan(r2):
            continue
            
        if r1 * r2 < 0:
            found_bracket = (grid[i], grid[i+1])
            print(f"  [Auto-Bracket] Found bracket: {found_bracket}")
            return found_bracket
            
    raise RuntimeError(f"Could not find a bracket in range [{p_min}, {p_max}]. Check M or model parameters.")

# ----------------------------------------------------------------------
# Root Finder
# ----------------------------------------------------------------------
def find_shooting_solution(M: float, delta: float) -> float:
    """
    Uses find_bracket + root_scalar to find p_b(0).
    """
    # 1. Automatically find a valid bracket
    bracket = find_bracket(M, delta)
    
    # 2. Refine root
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
        raise RuntimeError("Shooting method did not converge.")
        
    return result.root

# ----------------------------------------------------------------------
# Action Calculation (Restored from Fifth Draft)
# ----------------------------------------------------------------------
def compute_actions(sol: solve_ivp, M: float, delta: float, p_b0: float) -> Tuple[float, float]:
    """
    Computes S_E (total) and S_bulk using the trajectory.
    """
    tau = sol.t
    bE, cE, p_b, p_c = sol.y
    
    # 1. Total Euclidean Action (S_E) using Boundary Terms
    # S_E = (1/G*gamma) * [p_b * b_E - p_b0 * b0] (simplified boundary form from paper)
    # Note: Ensure this matches your paper's derivation (Eq. V.C / App. D)
    b0 = gamma
    p_bH = p_b[-1]
    bH = bE[-1]
    
    # Term at horizon - Term at bounce
    S_E = (1.0 / (G * gamma)) * (p_bH * bH - p_b0 * b0)
    
    # 2. Canonical Bulk Action (S_bulk) for cross-check
    # integrand = p_b * dot(b) + p_c * dot(c)
    
    # Calculate derivatives numerically or using EOM
    # Using EOM is more accurate on the grid
    db_dtau = (1.0 / G) * (sin(delta * cE) / delta)
    dc_dtau = (1.0 / G) * (bE * cos(delta * cE))
    
    integrand = p_b * db_dtau + p_c * dc_dtau
    
    # Use simpson for integration (Updated from simps)
    S_bulk_val = simpson(integrand, x=tau)
    
    # In a perfect on-shell calculation with proper boundary terms, 
    # S_E and S_bulk might differ by the GHY term value. 
    # The paper claims S_E (renormalized) is the physical quantity.
    
    return float(S_E), float(S_bulk_val)

# ----------------------------------------------------------------------
# Main Execution (CI/CD Ready)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # CI validation variables
    S_E = np.nan
    p_b0 = np.nan 

    M = 30.0
    delta = 0.05 
    print(f"--- Euclidean instanton solver: M={M}, delta={delta} ---")
    
    try:
        # A. Find Initial Condition
        print("\n[Shooting] Finding p_b(0)...")
        p_b0 = find_shooting_solution(M, delta)
        print(f"  Converged p_b(0) = {p_b0:.10e}")
        
        # B. Solve Trajectory
        sol = solve_instanton(M, delta, p_b0)
        
        # C. Compute Actions
        S_E, S_bulk = compute_actions(sol, M, delta, p_b0)
        
        # D. Print Results
        if len(sol.t_events[0]) > 0:
            print(f"  Horizon reached at tau_H = {sol.t_events[0][0]:.6f}")
            
        print("\n[Results]")
        print(f"  S_E (Total)  = {S_E:.10e}")
        print(f"  S_bulk       = {S_bulk:.10e}")
        
        # E. Numerical Validation for CI
        EXPECTED_S_E = 11333.0 
        TOLERANCE = 5.0e-3 # 0.5% tolerance
        
        diff = np.abs(S_E - EXPECTED_S_E)
        rel_err = diff / EXPECTED_S_E
        
        print(f"\n[Validation] Expected: {EXPECTED_S_E}, Got: {S_E:.2f}, RelErr: {rel_err:.2e}")
        
        if rel_err < TOLERANCE:
            print("[CI_VALIDATION] SUCCESS")
            sys.exit(0)
        else:
            print("[CI_VALIDATION] FAILED: Result outside tolerance.")
            sys.exit(1)

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] {type(e).__name__}: {e}")
        sys.exit(1)
