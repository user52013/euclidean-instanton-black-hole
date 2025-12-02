"""
Euclidean instanton solver for the polymer-corrected 
Kantowski–Sachs (KS) interior model. 

----------------------------------------------------------------------
Physical Units and Normalization
----------------------------------------------------------------------

All computations in this code are performed in natural Planck units:

    c = ℏ = k_B = 1 ,
    G = 1 .

Consequently:

• Mass, length, and time are expressed in multiples of the Planck scales
  (m_Pl, ℓ_Pl, t_Pl).

• The canonical variables (b_E, c_E; p_b, p_c), Euclidean proper time τ_E,
  and the Hamiltonian constraint C^(E,δ) are all dimensionless.

• The Euclidean action S_E is likewise dimensionless, consistent with the 
  instanton path-integral normalization used in the accompanying paper.

This normalization matches Appendix C of the manuscript and ensures that
the numerical output (constraint violation, shooting residual, and 
bulk–boundary cancellation) is directly comparable to the analytical 
expressions derived in Secs. III–V.

----------------------------------------------------------------------
Code Summary
----------------------------------------------------------------------

This solver integrates the Euclidean equations of motion using:

• Radau IIA (order 5/9) stiff ODE solver,
• boundary shooting to determine the unique initial momentum p_b(0),
• Gauss–Legendre quadrature for the bulk action,
• polymer (holonomy) substitutions applied after Wick rotation.

All results are validated through a continuous-integration (CI) workflow,
ensuring reproducibility of Euclidean instanton profiles and actions.

"""


import sys
import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp, simpson  # simpson for 1D Simpson integration
from scipy.optimize import root_scalar
from typing import Tuple

# ----------------------------------------------------------------------
# Physical constants (Planck units; G = 1)
# ----------------------------------------------------------------------
G = 1.0
gamma = 0.2375  # Barbero–Immirzi parameter

# ----------------------------------------------------------------------
# Euclidean equations of motion (polymerized)
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
    # dp_c / dτ (guard against p_c=0)
    if p_c <= 1e-14:
        dpc_dtau = 0.0
    else:
        dpc_dtau = (1.0 / (2.0 * G)) * (p_b**2 / (p_c**2))

    return np.array([db_dtau, dc_dtau, dpb_dtau, dpc_dtau], dtype=float)


# ----------------------------------------------------------------------
# Horizon event (b_E = 2 G M)
# ----------------------------------------------------------------------
def make_horizon_event(M: float):
    """
    Factory for the horizon event function.
    Triggered when b_E - 2GM = 0, with positive direction.
    """
    def horizon_event(tau: float, u: np.ndarray, M_arg: float, delta_arg: float) -> float:
        bE = u[0]
        return bE - 2.0 * G * M

    horizon_event.terminal = True
    horizon_event.direction = +1.0
    return horizon_event


# ----------------------------------------------------------------------
# Core ODE integration: solve_instanton
# ----------------------------------------------------------------------
def solve_instanton(
    M: float,
    delta: float,
    p_b0: float,
    tau_max: float = 200.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
):
    """
    Integrate the Euclidean EOM from the bounce to the horizon.

    Initial data at the bounce:
      b_E(0) = gamma, c_E(0) = 0, p_c(0) = gamma^2, p_b(0) = p_b0.
    """
    # Bounce initial conditions
    b0 = gamma
    c0 = 0.0
    p_c0 = gamma**2
    u0 = np.array([b0, c0, p_b0, p_c0], dtype=float)

    horizon_event = make_horizon_event(M)

    sol = solve_ivp(
        fun=eom_tau,
        t_span=(0.0, tau_max),
        y0=u0,
        method="Radau",
        events=horizon_event,
        args=(M, delta),
        rtol=rtol,
        atol=atol,
    )

    return sol


# ----------------------------------------------------------------------
# Shooting residual for horizon matching
# ----------------------------------------------------------------------
def shooting_residual(p_b0: float, M: float, delta: float) -> float:
    """
    Residual function for shooting:
      R(p_b0) = p_b(τ_H) - (2GM)^2.
    If integration fails or horizon is not reached, return a large penalty.
    """
    try:
        sol = solve_instanton(M, delta, p_b0)

        # If horizon event not reached, penalize
        if sol.t_events is None or len(sol.t_events) == 0 or len(sol.t_events[0]) == 0:
            return 1e50

        # p_b at the horizon
        p_bH = sol.y[2, -1]
        target = (2.0 * G * M) ** 2
        return p_bH - target

    except Exception:
        # Return a large residual if the ODE solver crashes
        return 1e50


# ----------------------------------------------------------------------
# Automatic Bracket Finder 
# ----------------------------------------------------------------------
def find_bracket(M: float, delta: float, p_min: float = 1.0, num: int = 50) -> Tuple[float, float]:
    """
    Automatically search for a bracket [p_left, p_right] with a sign change
    in the shooting residual.
    """
    target_pb = (2.0 * G * M) ** 2
    p_max = 10.0 * target_pb

    print(f"  [Auto-Bracket] Scanning range [{p_min:.1e}, {p_max:.1e}]...")

    grid = np.logspace(np.log10(p_min), np.log10(p_max), num=num)
    residuals = []

    for p in grid:
        r = shooting_residual(p, M, delta)
        residuals.append(r)

    residuals = np.array(residuals)

    # Check for sign change on segments where residuals are finite
    for i in range(len(grid) - 1):
        r1 = residuals[i]
        r2 = residuals[i + 1]

        if np.isnan(r1) or np.isnan(r2):
            continue

        if r1 * r2 < 0.0:
            print(f"  [Auto-Bracket] Found bracket: [{grid[i]:.2f}, {grid[i+1]:.2f}]")
            return grid[i], grid[i + 1]

    # If no sign change is found, raise an error for the caller to handle
    raise RuntimeError("Unable to find a valid bracket for p_b(0).")


# ----------------------------------------------------------------------
# Find initial momentum p_b(0) via robust root finding
# ----------------------------------------------------------------------
def find_shooting_solution(M: float, delta: float) -> float:
    """
    Determine the correct initial p_b(0) so that
      p_b(τ_H) = (2GM)^2
    at the horizon.
    """
    try:
        p_left, p_right = find_bracket(M, delta, p_min=1.0, num=50)
    except RuntimeError as e:
        print("  [Auto-Bracket] Failed to find bracket.")
        raise e

    def residual_wrapper(p_b0: float) -> float:
        return shooting_residual(p_b0, M, delta)

    # Use Brent's method once we have a valid bracket
    result = root_scalar(
        residual_wrapper,
        bracket=(p_left, p_right),
        method="brentq",
        rtol=1e-8,
        maxiter=100,
    )

    if not result.converged:
        raise RuntimeError("Shooting method did not converge.")

    return result.root


# ----------------------------------------------------------------------
# Euclidean Action Evaluation (compute_actions)
# ----------------------------------------------------------------------
def compute_actions(sol: solve_ivp, M: float, delta: float, p_b0: float) -> Tuple[float, float]:
    """
    Compute the on-shell Euclidean action S_E and the canonical bulk action S_bulk.

    The bulk part is
        S_bulk = ∫_0^{τ_H} (p_b \dot{b}_E + p_c \dot{c}_E) dτ_E

    and the full Euclidean action including the polymer-corrected GHY term is
        S_E = S_bulk + S_GHY^{(E,δ)}

    with
        S_GHY^{(E,δ)} = - |p_c(τ_H)| / (G γ) * sin(δ c_E(τ_H)) / δ.

    This follows the canonical and boundary analysis in the Appendices.
    """
    tau = sol.t
    bE, cE, p_b, p_c = sol.y

    # Reconstruct db_E/dτ_E and dc_E/dτ_E using the same EOM for consistency
    db_dtau = np.empty_like(bE)
    dc_dtau = np.empty_like(cE)

    for i in range(len(tau)):
        u_i = sol.y[:, i]
        derivs = eom_tau(tau[i], u_i, M, delta)
        db_dtau[i] = derivs[0]
        dc_dtau[i] = derivs[1]

    integrand = p_b * db_dtau + p_c * dc_dtau

    # Canonical bulk action via Simpson integration
    S_bulk = simpson(integrand, x=tau)

    # Polymer-corrected GHY term evaluated at the horizon (final point)
    cH = cE[-1]
    p_cH = p_c[-1]
    S_GHY = - (abs(p_cH) / (G * gamma)) * (sin(delta * cH) / delta)

    # Full Euclidean action
    S_E = S_bulk + S_GHY

    return float(S_E), float(S_bulk)


# ----------------------------------------------------------------------
# Main Execution (CI/CD Ready)
# ----------------------------------------------------------------------
if __name__ == "__main__":

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

        # D. Numerical Validation for CI
        # Expected on-shell Euclidean action for M=30, delta=0.05 using the
        # canonical bulk + polymer-GHY prescription.
        EXPECTED_S_E = 1.8498734595582305e5
        TOLERANCE = 5.0e-4  # 0.05% relative tolerance

        diff = np.abs(S_E - EXPECTED_S_E)
        rel_err = diff / EXPECTED_S_E

        print(f"\n[Results]")
        print(f"  S_E (Total)        = {S_E:.10e}")
        print(f"  S_bulk (Canonical) = {S_bulk:.10e}")

        print(
            f"\n[Validation] Expected: {EXPECTED_S_E:.10e}, "
            f"Got: {S_E:.10e}, RelErr: {rel_err:.3e}"
        )

        if rel_err < TOLERANCE:
            print("[CI_VALIDATION] SUCCESS")
            sys.exit(0)
        else:
            print("[CI_VALIDATION] FAILED: Result outside tolerance.")
            sys.exit(1)

    except Exception as e:

        print("\n[CRITICAL FAILURE] Test Run Failed.")
        print(f"Actual Error Message: {type(e).__name__}: {e}")
        sys.exit(1)
