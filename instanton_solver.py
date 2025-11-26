import sys
from typing import Dict, Tuple, List

import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp, simps
from scipy.optimize import root_scalar

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

    Parameters
    ----------
    tau : float
        Euclidean time parameter.
    u : array_like, shape (4,)
        Phase space variables [b_E, c_E, p_b, p_c].
    M : float
        Black-hole mass (enters only through boundary conditions; it is
        included in the signature so that solve_ivp can pass args=(M, delta)
        consistently to both the RHS and event function).
    delta : float
        Polymer scale (delta_b = delta_c = delta).

    Returns
    -------
    du_dtau : ndarray, shape (4,)
        Time derivatives of [b_E, c_E, p_b, p_c].
    """
    bE, cE, p_b, p_c = u

    # Holonomy-modified equations (same structure as in the paper;
    # see Sec. III–IV).
    dbE = (1.0 / G) * (sin(delta * cE) / delta)
    dcE = (1.0 / G) * (bE * cos(delta * cE))
    dp_b = -(1.0 / (2.0 * G)) * sin(2.0 * delta * bE) / delta

    # p_c > 0 along the instanton trajectory; guard against division by zero.
    if p_c <= 0.0:
        dp_c = 0.0
    else:
        dp_c = (1.0 / (2.0 * G)) * (p_b ** 2 / (p_c ** 2))

    return np.array([dbE, dcE, dp_b, dp_c], dtype=float)


# ----------------------------------------------------------------------
# Hamiltonian constraint (for diagnostics only)
# ----------------------------------------------------------------------


def H_E(u: np.ndarray, delta: float) -> float:
    """
    Euclidean Hamiltonian constraint used as a diagnostic.

    This matches the simplified form used in the code accompanying the paper.
    """
    bE, cE, p_b, p_c = u

    term1 = 2.0 * bE * sin(delta * cE) / delta
    term2 = (sin(delta * bE) / delta) ** 2
    # Effective potential (KS interior)
    U = 1.0 - p_b ** 2 / abs(p_c)

    return -(term1 + term2 + U) / (2.0 * G)


# ----------------------------------------------------------------------
# Horizon event (b_E = 2 G M)
# ----------------------------------------------------------------------


def horizon_event(tau: float, u: np.ndarray, M: float, delta: float) -> float:
    """
    Event function locating the matching hypersurface (the "horizon").

    The event is triggered when b_E reaches 2 G M.
    """
    bE = u[0]
    return bE - 2.0 * G * M


# ----------------------------------------------------------------------
# Instanton solver
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
    Integrate the Euclidean equations from the bounce to the horizon.

    Parameters
    ----------
    M : float
        Black-hole mass.
    delta : float
        Polymer scale.
    p_b0 : float
        Initial value p_b(0) at the bounce.
    tau_max : float, optional
        Maximum Euclidean time for the integration.
    rtol, atol : float, optional
        Relative and absolute tolerances for solve_ivp.

    Returns
    -------
    sol : OdeResult
        Solution object returned by scipy.integrate.solve_ivp.
    """
    # Bounce initial data (Sec. IV)
    b0 = gamma
    c0 = 0.0
    p_c0 = gamma ** 2
    u0 = np.array([b0, c0, p_b0, p_c0], dtype=float)

    def event(tau, u, M_arg, delta_arg):
        return horizon_event(tau, u, M_arg, delta_arg)

    event.terminal = True
    event.direction = +1.0

    sol = solve_ivp(
        fun=eom_tau,
        t_span=(0.0, tau_max),
        y0=u0,
        method="Radau",
        rtol=rtol,
        atol=atol,
        events=event,
        args=(M, delta),
    )

    if sol.status < 0:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol


# ----------------------------------------------------------------------
# Shooting: determine p_b(0) from horizon boundary condition
# ----------------------------------------------------------------------


def shooting_target(p_b0: float, M: float, delta: float) -> float:
    """
    Residual used in the shooting method.

    We impose p_b(tau_H) = (2 G M)^2 at the horizon.
    """
    try:
        sol = solve_instanton(M, delta, p_b0)
    except Exception:
        # If the ODE solver fails badly for this guess, return a large
        # residual with fixed sign so that the bracket search can move on.
        return 1e30

    # If no horizon event was found, use the deviation in b_E as a proxy.
    if len(sol.t_events[0]) == 0:
        bH = sol.y[0, -1]
        b_target = 2.0 * G * M
        return bH - b_target

    # Otherwise, use the p_b boundary condition at the horizon.
    p_bH = sol.y[2, -1]
    target = (2.0 * G * M) ** 2
    return p_bH - target


def find_bracket(
    M: float,
    delta: float,
    p_min: float = 1e-3,
    p_max: float = None,
    num: int = 40,
) -> Tuple[float, float]:
    """
    Automatically search for a bracket [p_left, p_right] with a sign change
    in the shooting residual.

    This is used to provide a robust starting interval for root_scalar.
    """
    if p_max is None:
        # A conservative upper scale based on the horizon momentum.
        p_max = 5.0 * (2.0 * G * M) ** 2

    # Use a logarithmic scan to cover several orders of magnitude.
    grid = np.logspace(np.log10(p_min), np.log10(p_max), num=num)
    residuals = []

    for p in grid:
        r = shooting_target(p, M, delta)
        if not np.isfinite(r):
            residuals.append(np.nan)
        else:
            residuals.append(r)

    residuals = np.array(residuals, dtype=float)

    # Look for neighbouring points with opposite sign.
    for i in range(len(grid) - 1):
        r1, r2 = residuals[i], residuals[i + 1]
        if np.isnan(r1) or np.isnan(r2):
            continue
        if r1 == 0.0:
            return grid[i], grid[i]
        if r1 * r2 < 0.0:
            return grid[i], grid[i + 1]

    raise RuntimeError(
        "Unable to find a sign-changing bracket for p_b0. "
        "Try adjusting p_min/p_max or inspecting the residual manually."
    )


def find_shooting_solution(
    M: float,
    delta: float,
    bracket: Tuple[float, float] = None,
) -> float:
    """
    Find the correct initial momentum p_b(0) by solving shooting_target = 0.
    """
    if bracket is None:
        bracket = find_bracket(M, delta)

    root = root_scalar(
        shooting_target,
        args=(M, delta),
        bracket=bracket,
        method="brentq",
        xtol=1e-12,
        rtol=1e-10,
        maxiter=100,
    )

    if not root.converged:
        raise RuntimeError("Shooting method did not converge.")

    return float(root.root)


# ----------------------------------------------------------------------
# Euclidean action evaluation
# ----------------------------------------------------------------------


def compute_actions(
    sol,
    M: float,
    delta: float,
    p_b0: float,
) -> Tuple[float, float]:
    """
    Compute the total Euclidean action S_E and the canonical bulk action.

    S_E is evaluated using the boundary expression
        S_E = (1 / (G * gamma)) [p_b b_E]_{bounce}^{horizon},
    which matches the derivation in Sec. V and Appendix D.

    The bulk action S_bulk is computed independently using a Simpson
    quadrature over tau as a numerical cross-check.
    """
    tau = sol.t
    bE, cE, p_b, p_c = sol.y

    # Total Euclidean action from boundary terms (bulk + GHY on-shell)
    b0 = gamma
    p_bH = p_b[-1]
    bH = bE[-1]

    S_E = (1.0 / (G * gamma)) * (p_bH * bH - p_b0 * b0)

    # Canonical bulk action for comparison: integral of p dq.
    # Use Simpson quadrature and finite-difference derivatives.
    db_dtau = np.gradient(bE, tau)
    dc_dtau = np.gradient(cE, tau)
    integrand = p_b * db_dtau + p_c * dc_dtau
    S_bulk = simps(integrand, tau)

    return float(S_E), float(S_bulk)


# ----------------------------------------------------------------------
# Parameter scan (mass and polymer scale)
# ----------------------------------------------------------------------


def scan_parameters(
    M_list: List[float],
    delta_list: List[float],
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """
    Scan over a list of masses and polymer scales.

    Returns
    -------
    results : dict
        Dictionary keyed by (M, delta).  Each entry contains:

        {
            "status": "ok" or "failed",
            "p_b0": ...,
            "S_E": ...,
            "S_bulk": ...,
            "bH": ...,
            "residual": ...,
        }
    """
    results: Dict[Tuple[float, float], Dict[str, float]] = {}

    for M in M_list:
        for delta in delta_list:
            key = (float(M), float(delta))
            try:
                p_b0 = find_shooting_solution(M, delta)
                sol = solve_instanton(M, delta, p_b0)
                S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

                if len(sol.t_events[0]) == 0:
                    bH = sol.y[0, -1]
                else:
                    bH = sol.y[0, -1]

                target = (2.0 * G * M) ** 2
                res = sol.y[2, -1] - target

                results[key] = {
                    "status": "ok",
                    "p_b0": float(p_b0),
                    "S_E": float(S_E),
                    "S_bulk": float(S_bulk),
                    "bH": float(bH),
                    "residual": float(res),
                }
            except Exception as exc:  # noqa: BLE001
                results[key] = {
                    "status": "failed",
                    "error": str(exc),
                }

    return results


# ----------------------------------------------------------------------
# Example run and simple CI-style validation
# ----------------------------------------------------------------------


if __name__ == "__main__":
    import sys as _sys

    M = 30.0
    delta = 0.05

    print(f"--- Euclidean instanton solver: M={M}, delta={delta} ---")

    try:
        p_b0 = find_shooting_solution(M, delta)
        print(f"[shooting] p_b(0) = {p_b0:.10e}")

        sol = solve_instanton(M, delta, p_b0)
        S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

        if len(sol.t_events[0]) == 0:
            print("Warning: horizon event was not detected.")
        else:
            print(f"Horizon reached at tau_H = {sol.t_events[0][0]:.10e}")

        bH = sol.y[0, -1]
        p_bH = sol.y[2, -1]
        target = (2.0 * G * M) ** 2
        residual = p_bH - target

        print("\n[Shooting summary]")
        print(f"  b_E(tau_H)   = {bH:.10e}")
        print(f"  p_b(tau_H)   = {p_bH:.10e}")
        print(f"  target p_bH  = {target:.10e}")
        print(f"  residual     = {residual:.3e}")

        print("\n[Action summary]")
        print(f"  S_E (total)  = {S_E:.10e}")
        print(f"  S_bulk       = {S_bulk:.10e}")
        print(f"  |S_E-S_bulk| = {abs(S_E - S_bulk):.3e}")

        # Simple CI-style numerical check against the reference value
        # quoted in the paper for (M, delta) = (30, 0.05).
        EXPECTED_S_E = 11333.0
        TOL = 5.0e-3  # 0.5 per cent relative tolerance

        rel_err = abs(S_E - EXPECTED_S_E) / EXPECTED_S_E
        print(f"\n[Validation] relative error vs. reference = {rel_err:.3e}")

        if rel_err < TOL:
            print("[Validation] SUCCESS: S_E is within tolerance.")
            _sys.exit(0)
        else:
            print("[Validation] FAILED: S_E is outside tolerance.")
            _sys.exit(1)

    except Exception as exc:  # noqa: BLE001
        print("\n[CRITICAL FAILURE] Test run failed.")
        print(f"Error: {type(exc).__name__}: {exc}")
        _sys.exit(1)
