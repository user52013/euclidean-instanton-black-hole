import sys
import numpy as np
from numpy import sin, cos
# [修正]: 使用 simpson 替代 simps 以兼容新版 SciPy，解決 ImportError
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar

# ----------------------------------------------------------------------
# Physical constants (Planck units; G = 1)
# ----------------------------------------------------------------------
G = 1.0
gamma = 0.2375  # Barbero–Immirzi parameter

# ----------------------------------------------------------------------
# Euclidean equations of motion (polymerized)
# ----------------------------------------------------------------------
def eom_tau(tau, u, M, delta):
    """
    Euclidean equations of motion for the polymer-corrected KS interior.
    Parameters
    ----------
    tau : float
        Euclidean time parameter.
    u : array_like, shape (4,)
        Phase space variables [b_E, c_E, p_b, p_c].
    M : float
        Black-hole mass.
    delta : float
        Polymer scale (delta_b = delta_c = delta).
    """
    bE, cE, p_b, p_c = u
    
    # db_E / dτ
    db_dtau = (1.0 / G) * (sin(delta * cE) / delta)
    # dc_E / dτ
    dc_dtau = (1.0 / G) * (bE * cos(delta * cE))
    # dp_b / dτ
    dpb_dtau = -(1.0 / (2.0 * G)) * sin(2.0 * delta * bE) / delta
    # dp_c / dτ (guard against p_c=0)
    if p_c <= 1e-14:
        dpc_dtau = 0.0
    else:
        dpc_dtau = (1.0 / (2.0 * G)) * (p_b**2 / (p_c**2))
        
    return [db_dtau, dc_dtau, dpb_dtau, dpc_dtau]

# ----------------------------------------------------------------------
# Horizon event (b_E = 2 G M)
# ----------------------------------------------------------------------
def horizon_event(tau, u, M, delta):
    """
    Event function locating the matching hypersurface (the "horizon").
    The event is triggered when b_E reaches 2 G M.
    """
    bE = u[0]
    return bE - 2.0 * G * M

# Mark the event as terminal and only trigger when bE is increasing.
horizon_event.terminal = True
horizon_event.direction = +1.0

# ----------------------------------------------------------------------
# Instanton solver
# ----------------------------------------------------------------------
def solve_instanton(M, delta, p_b0, tau_max=200.0):
    """
    Integrate the Euclidean EOM from the bounce to the horizon.
    """
    # Bounce initial data (Regularity conditions)
    b0 = gamma
    c0 = 0.0
    p_c0 = gamma**2
    u0 = [b0, c0, p_b0, p_c0]
    
    sol = solve_ivp(
        fun=eom_tau,
        t_span=(0.0, tau_max),
        y0=u0,
        args=(M, delta),
        method="Radau", # Implicit method for stiffness
        events=horizon_event,
        rtol=1e-10,
        atol=1e-12,
    )
    return sol

# ----------------------------------------------------------------------
# Shooting residual
# ----------------------------------------------------------------------
def shooting_residual(p_b0, M, delta):
    """
    Residual for the shooting method: enforce p_b(τ_H) = (2GM)^2.
    """
    try:
        sol = solve_instanton(M, delta, p_b0)
        
        # If horizon event was not reached
        if sol.t_events is None or len(sol.t_events[0]) == 0:
            # Return a large penalty to guide the solver back
            bH = sol.y[0, -1]
            return (bH - 2.0 * G * M) * 1e5 
            
        p_bH = sol.y[2, -1]
        target = (2.0 * G * M)**2
        return p_bH - target
    except Exception:
        return 1e50 # Fallback for solver failures

# ----------------------------------------------------------------------
# Automatic Bracket Finder (From Fifth Draft)
# ----------------------------------------------------------------------
def find_bracket(M, delta, p_min=1.0, num=50):
    """
    Automatically search for a bracket [p_left, p_right] with a sign change.
    This is crucial for robustness when M is large.
    """
    # Estimate target scale
    target_pb = (2.0 * G * M)**2
    p_max = 10.0 * target_pb
    
    print(f"  [Auto-Bracket] Scanning range [{p_min:.1e}, {p_max:.1e}]...")
    
    grid = np.logspace(np.log10(p_min), np.log10(p_max), num=num)
    residuals = []
    
    for p in grid:
        r = shooting_residual(p, M, delta)
        residuals.append(r)
    
    residuals = np.array(residuals)
    
    # Check for sign change
    for i in range(len(grid) - 1):
        r1 = residuals[i]
        r2 = residuals[i+1]
        
        if np.isnan(r1) or np.isnan(r2):
            continue
            
        if r1 * r2 < 0.0:
            print(f"  [Auto-Bracket] Found bracket: [{grid[i]:.2f}, {grid[i+1]:.2f}]")
            return grid[i], grid[i+1]
            
    raise RuntimeError(f"Could not find a bracket in range [{p_min}, {p_max}].")

# ----------------------------------------------------------------------
# Root Finder (Shooting Method)
# ----------------------------------------------------------------------
def find_shooting_solution(M, delta):
    """
    Find the correct initial momentum p_b(0) using auto-bracketing and Brent's method.
    """
    # 1. Find a valid bracket first
    bracket = find_bracket(M, delta)
    
    # 2. Refine the root
    result = root_scalar(
        shooting_residual,
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
# Euclidean Action Evaluation (Restored from your drafts)
# ----------------------------------------------------------------------
def compute_actions(sol, M, delta, p_b0):
    """
    Compute the on-shell Euclidean action using the canonical boundary expression,
    and a bulk integral as a consistency check.
    
    *** FIXED: Uses 'simpson' instead of 'simps' for SciPy compatibility ***
    """
    tau = sol.t
    bE, cE, p_b, p_c = sol.y
    
    # Boundary expression: S_E = (1 / (G gamma)) [p_b b_E]_0^{τ_H}
    bH = bE[-1]
    p_bH = p_b[-1]
    
    # Term at horizon - Term at bounce (b0=gamma)
    S_E_boundary = (1.0 / (G * gamma)) * (p_bH * bH - p_b0 * gamma)
    
    # Bulk check: integral of p dq
    # We recompute derivatives using the EOM for accuracy on the grid
    db_dtau = np.empty_like(bE)
    dc_dtau = np.empty_like(cE)
    
    for i in range(len(tau)):
        # Unpack state at step i
        u_i = sol.y[:, i]
        derivs = eom_tau(tau[i], u_i, M, delta)
        db_dtau[i] = derivs[0]
        dc_dtau[i] = derivs[1]
        
    integrand = p_b * db_dtau + p_c * dc_dtau
    
    # [FIX] Use simpson instead of simps
    S_bulk = simpson(integrand, x=tau)
    
    return S_E_boundary, S_bulk

# ----------------------------------------------------------------------
# Main Execution (CI/CD Ready)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # Initialize variables for safety
    S_E = np.nan
    S_bulk = np.nan
    p_b0 = np.nan 

    # Parameters from Paper Sec. V.E
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
        
        # C. Compute Actions (Using your original logic)
        S_E, S_bulk = compute_actions(sol, M, delta, p_b0)
        
        # D. Print Results
        if sol.t_events and len(sol.t_events[0]) > 0:
            print(f"  Horizon reached at tau_H = {sol.t_events[0][0]:.6f}")
        else:
            print("  Warning: Horizon not detected perfectly.")
            
        bH = sol.y[0, -1]
        p_bH = sol.y[2, -1]
        target = (2.0 * G * M)**2
        residual = p_bH - target

        print("\n[Results]")
        print(f"  p_b(τ_H)     = {p_bH:.10e}")
        print(f"  Target       = {target:.10e}")
        print(f"  Residual     = {residual:.3e}")
        print(f"  S_E (Total)  = {S_E:.10e}")
        print(f"  S_bulk       = {S_bulk:.10e}")
        print(f"  Diff         = {abs(S_E - S_bulk):.3e}")
        
        # E. Numerical Validation for CI
        # Expected value from paper: 11333.0
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
        import traceback
        traceback.print_exc()
        sys.exit(1)
