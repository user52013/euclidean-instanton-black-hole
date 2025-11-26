import numpy as np
from numpy import sin, cos

from scipy.integrate import solve_ivp
try:
    # Newer SciPy: simpson is the supported 1D Simpson integrator
    from scipy.integrate import simpson as _simpson
except ImportError:
    # Fallback: use numpy.trapz if simpson is not available
    def _simpson(y, x):
        return np.trapz(y, x)

from scipy.optimize import root_scalar

# -----------------------------------------------------
# Physical constants
# -----------------------------------------------------
G = 1.0          # Gravitational constant (in units where G = 1)
gamma = 0.2375   # Barbero–Immirzi parameter


# -----------------------------------------------------
# Euclidean equations of motion
# -----------------------------------------------------
def eom_tau(tau, u, M, delta):
    """
    Euclidean equations of motion (EOM) for the polymer-corrected
    Kantowski–Sachs interior.

    Parameters
    ----------
    tau : float
        Euclidean time parameter τ_E.
    u : array_like, shape (4,)
        State vector [b_E, c_E, p_b, p_c].
    M : float
        Black hole mass.
    delta : float
        Polymer scale (same for b_E and c_E in this implementation).

    Returns
    -------
    du_dtau : list[float]
        Time derivatives [db_E/dτ, dc_E/dτ, dp_b/dτ, dp_c/dτ].
    """
    b_E, c_E, p_b, p_c = u

    # Polymer-substituted connections
    fb = sin(delta * b_E) / delta
    fc = sin(delta * c_E) / delta

    # The specific form here matches the minisuperspace Hamiltonian
    # used in Sec. III–V.  We keep the same structure as in the
    # previous versions of this file so as not to change the physics.
    #
    # db_E / dτ_E
    db_dtau = (1.0 / G) * fc

    # dc_E / dτ_E
    dc_dtau = (1.0 / G) * b_E * cos(delta * c_E)

    # dp_b / dτ_E
    dpb_dtau = -(1.0 / (2.0 * G)) * sin(2.0 * delta * b_E) / delta

    # dp_c / dτ_E
    # (same structure as older version; depends only on p_b, p_c)
    if p_c != 0.0:
        dpc_dtau = (1.0 / (2.0 * G)) * (p_b ** 2) / (p_c ** 2)
    else:
        # Avoid division by zero in pathological cases
        dpc_dtau = 0.0

    return [db_dtau, dc_dtau, dpb_dtau, dpc_dtau]


# -----------------------------------------------------
# Horizon event: b_E = 2 G M
# -----------------------------------------------------
def _make_horizon_event(M):
    """
    Factory for the horizon event function b_E - 2 G M = 0.
    The returned function has the signature required by solve_ivp.
    """

    def horizon_event(tau, u, M_arg, delta_arg):
        # We ignore M_arg, delta_arg here because we close over M.
        b_E = u[0]
        return b_E - 2.0 * G * M

    horizon_event.terminal = True
    horizon_event.direction = +1.0
    return horizon_event


# -----------------------------------------------------
# Core ODE integration: solve_instanton
# -----------------------------------------------------
def solve_instanton(M, delta, p_b0,
                    tau_max=200.0,
                    rtol=1e-10,
                    atol=1e-12):
    """
    Integrate the Euclidean EOM from the bounce to the horizon.

    Parameters
    ----------
    M : float
        Black hole mass.
    delta : float
        Polymer scale.
    p_b0 : float
        Initial momentum p_b(0) at the bounce.
    tau_max : float, optional
        Maximum Euclidean integration time.
    rtol, atol : float, optional
        Relative and absolute tolerances for solve_ivp.

    Returns
    -------
    sol : OdeResult
        SciPy solve_ivp solution object.
    """
    # Bounce initial data
    b0 = gamma          # minimal radius scale
    c0 = 0.0
    p_c0 = gamma ** 2
    u0 = [b0, c0, p_b0, p_c0]

    horizon_event = _make_horizon_event(M)

    # Integrate Euclidean EOM
    sol = solve_ivp(
        fun=eom_tau,
        t_span=(0.0, tau_max),
        y0=u0,
        method="Radau",         # stiff, A-stable
        events=horizon_event,
        args=(M, delta),
        rtol=rtol,
        atol=atol,
    )

    return sol


# -----------------------------------------------------
# Shooting residual for horizon matching
# -----------------------------------------------------
def shooting_residual(p_b0, M, delta):
    """
    Residual function used in the shooting method.

    We require p_b(τ_H) = (2 G M)^2 at the horizon τ_H.
    """

    try:
        sol = solve_instanton(M, delta, p_b0)

        # If horizon event not reached, penalize strongly
        if (sol.t_events is None or len(sol.t_events) == 0 or
                len(sol.t_events[0]) == 0):
            return 1e6

        uH = sol.y[:, -1]
        p_bH = uH[2]
        target = (2.0 * G * M) ** 2

        return p_bH - target

    except Exception:
        # Any numerical failure is treated as very bad residual
        return 1e6


# -----------------------------------------------------
# Find initial momentum p_b(0) via robust root finding
# -----------------------------------------------------
def find_shooting_solution(M,
                           delta,
                           bracket=(10.0, 500.0),
                           rtol=1e-8,
                           maxiter=50):
    """
    Determine the correct initial p_b(0) such that the
    horizon condition p_b(τ_H) = (2 G M)^2 is satisfied.

    Parameters
    ----------
    M : float
        Black hole mass.
    delta : float
        Polymer scale.
    bracket : tuple(float, float), optional
        Initial bracketing interval for root_scalar.
    rtol : float, optional
        Relative tolerance for the root.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    p_b0 : float
        Converged initial momentum at the bounce.
    """
    def f(p_b0):
        return shooting_residual(p_b0, M, delta)

    root_result = root_scalar(
        f,
        bracket=bracket,
        method="brentq",
        rtol=rtol,
        maxiter=maxiter
    )

    if not root_result.converged:
        raise RuntimeError(
            f"Shooting did not converge for M={M}, delta={delta} "
            f"within bracket={bracket}"
        )

    return root_result.root


# -----------------------------------------------------
# Euclidean action: bulk + boundary (GHY) consistency
# -----------------------------------------------------
def compute_actions(sol, M, delta, p_b0):
    """
    Compute the bulk canonical action S_bulk and the full Euclidean
    action S_E (including the polymer-corrected GHY boundary term).

    Parameters
    ----------
    sol : OdeResult
        Solution object from solve_instanton.
    M : float
        Black hole mass.
    delta : float
        Polymer parameter.
    p_b0 : float
        Initial momentum at the bounce.

    Returns
    -------
    S_E : float
        Full on-shell Euclidean action.
    S_bulk : float
        Canonical bulk action ∫ (p_b db_E + p_c dc_E) dτ_E.
    """
    tau = sol.t
    b_E = sol.y[0]
    c_E = sol.y[1]
    p_b = sol.y[2]
    p_c = sol.y[3]

    # Numerical derivatives db_E/dτ_E, dc_E/dτ_E
    db_dtau = np.gradient(b_E, tau)
    dc_dtau = np.gradient(c_E, tau)

    # Canonical bulk action via 1D high-order quadrature
    integrand = p_b * db_dtau + p_c * dc_dtau
    S_bulk = _simpson(integrand, tau)

    # Polymer-corrected GHY term at the horizon
    uH = sol.y[:, -1]
    c_EH = uH[1]
    p_cH = uH[3]

    # K_E^(delta) ~ (2/gamma) sin(delta c_E) / delta
    S_GHY = -(np.abs(p_cH) / (G * gamma)) * (sin(delta * c_EH) / delta)

    # Canonical boundary term at the bounce:
    # p_b(0) b_E(0) with b_E(0) = gamma
    S_bounce = (p_b0 * gamma) / (G * gamma)

    # Consistent with the derivation in Appendix D:
    # full Euclidean action combines bulk + boundary pieces.
    S_E = S_bulk + S_GHY - S_bounce

    return S_E, S_bulk


# Backwards compatibility wrapper
def compute_full_action(sol, M, delta, p_b0):
    """
    Wrapper for older code that expected compute_full_action.
    """
    return compute_actions(sol, M, delta, p_b0)


# -----------------------------------------------------
# Parameter scan over (M, delta)
# -----------------------------------------------------
def scan_parameters(M_list, delta_list,
                    bracket=(10.0, 500.0)):
    """
    Systematically scan the (M, delta) parameter space and
    return a list of (M, delta, S_E) dictionaries.

    This is meant to support the fits S_E ∝ M^{2+δ_eff}
    discussed in Sec. V.D.

    Parameters
    ----------
    M_list : iterable of float
        List of black hole masses.
    delta_list : iterable of float
        List of polymer scales.
    bracket : tuple(float, float), optional
        Initial bracket for shooting in each (M, delta) pair.

    Returns
    -------
    results : list of dict
        Each element has keys "M", "delta", "S_E", "S_bulk".
    """
    results = []

    for M in M_list:
        for delta in delta_list:
            p_b0 = find_shooting_solution(M, delta, bracket=bracket)
            sol = solve_instanton(M, delta, p_b0)
            S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

            results.append({
                "M": M,
                "delta": delta,
                "S_E": S_E,
                "S_bulk": S_bulk,
            })

    return results


# -----------------------------------------------------
# Example run + CI validation
# -----------------------------------------------------
if __name__ == "__main__":
    import sys

    # Example parameters matching Sec. V.E
    M = 30.0
    delta = 0.05

    print(f"--- Euclidean instanton solver: M={M}, delta={delta} ---")

    p_b0 = None

    try:
        # A. Shooting to find p_b(0)
        print("\n[Shooting] Attempting to find p_b(0)...")
        p_b0 = find_shooting_solution(M, delta, bracket=(10.0, 500.0))
        print(f"  Found p_b(0) = {p_b0:.10e}")

        # B. Integrate EOM
        sol = solve_instanton(M, delta, p_b0)

        # C. Compute actions
        S_E, S_bulk = compute_actions(sol, M, delta, p_b0)

        # D. Summary
        if sol.t_events is None or len(sol.t_events) == 0 or len(sol.t_events[0]) == 0:
            print("Warning: horizon event was not detected.")
        else:
            print(f"Horizon reached at τ_H = {sol.t_events[0][0]:.10e}")

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

        # E. CI validation: compare with expected S_E ≈ 11333
        EXPECTED_S_E = 11333.0
        TOLERANCE = 5.0e-5  # 0.005 relative tolerance

        rel_err = np.abs(S_E - EXPECTED_S_E) / EXPECTED_S_E

        if rel_err < TOLERANCE:
            print("\n[CI_VALIDATION] SUCCESS: "
                  "Computed S_E matches expected value (within tolerance).")
            sys.exit(0)
        else:
            print("\n[CI_VALIDATION] FAILED:")
            print(f"  Computed S_E = {S_E:.10e}")
            print(f"  Expected S_E = {EXPECTED_S_E:.10e}")
            print(f"  Relative error = {rel_err:.3e}")
            sys.exit(1)

    except Exception as e:
        print("\n[CRITICAL FAILURE] Test Run Failed.")
        print(f"Actual Error Message: {type(e).__name__}: {e}")
        sys.exit(1)
