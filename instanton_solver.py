import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar

# =========================================================
# Physical constants and global polymer parameters
# =========================================================

G = 1.0        # Gravitational constant (units with G = 1)
gamma = 0.2375 # Barbero–Immirzi parameter (fixed from BH entropy)

# Default polymer scales (can be varied in scans)
delta_b_default = 0.03
delta_c_default = 0.03


# =========================================================
# Euclidean polymer Hamiltonian for diagnostics
# =========================================================

def H_E(u, M, delta_b=delta_b_default, delta_c=delta_c_default):
    """
    Euclidean Hamiltonian constraint H_E for the polymer-corrected
    Kantowski–Sachs interior (used only for diagnostics).

    Parameters
    ----------
    u : array-like
        Phase-space vector [b_E, c_E, p_b, p_c].
    M : float
        Black-hole mass (enters only through horizon scale, but we
        keep it here for completeness).
    delta_b, delta_c : float
        Polymer parameters for the b- and c-sectors.

    Returns
    -------
    float
        Value of the Euclidean Hamiltonian constraint.
    """
    bE, cE, p_b, p_c = u

    # Polymer holonomy functions
    fb = sin(delta_b * bE) / delta_b
    fc = sin(delta_c * cE) / delta_c

    # Effective potential-like term
    U = 1.0 - p_b**2 / abs(p_c)

    # Hamiltonian (up to overall lapse factor)
    term1 = 2.0 * bE * fc
    term2 = fb**2

    return - (term1 + term2 + U) / (2.0 * G)


# =========================================================
# Euclidean equations of motion
# =========================================================

def eom_tau(tau, u, delta_b=delta_b_default, delta_c=delta_c_default):
    """
    Euclidean equations of motion du/dτ = {u, H_E} for the
    polymer-corrected Kantowski–Sachs interior.

    Parameters
    ----------
    tau : float
        Euclidean "time" parameter τ_E.
    u : array-like
        Phase-space vector [b_E, c_E, p_b, p_c].
    delta_b, delta_c : float
        Polymer parameters (holonomy scales) for b and c.

    Returns
    -------
    list
        Time derivatives [db_E/dτ, dc_E/dτ, dp_b/dτ, dp_c/dτ].
    """
    bE, cE, p_b, p_c = u

    # db_E/dτ
    dbE = (1.0 / G) * (sin(delta_c * cE) / delta_c)

    # dc_E/dτ
    dcE = (1.0 / G) * (bE * cos(delta_c * cE))

    # dp_b/dτ
    dpb = - (1.0 / (2.0 * G)) * sin(2.0 * delta_b * bE) / delta_b

    # dp_c/dτ
    dpc = (1.0 / (2.0 * G)) * (p_b**2 / p_c**2)

    return [dbE, dcE, dpb, dpc]


# =========================================================
# Horizon event: b_E = 2 G M  (matching surface)
# =========================================================

def horizon_event(tau, u, M):
    """
    Event function for the horizon matching surface b_E = 2 G M.

    Parameters
    ----------
    tau : float
        Euclidean time.
    u : array-like
        Phase-space vector [b_E, c_E, p_b, p_c].
    M : float
        Black-hole mass.

    Returns
    -------
    float
        Zero when b_E reaches the (polymer-corrected) horizon radius.
    """
    bE = u[0]
    return bE - 2.0 * G * M

# Stop integration when the horizon is reached, from below
horizon_event.terminal = True
horizon_event.direction = +1.0


# =========================================================
# Shooting and ODE integration
# =========================================================

def initial_data_bounce(M, p_b0, delta_b=delta_b_default, delta_c=delta_c_default):
    """
    Regularity conditions at the Euclidean bounce surface τ_E = 0.

    Parameters
    ----------
    M : float
        Black-hole mass.
    p_b0 : float
        Shooting parameter: initial value of p_b at the bounce.
    delta_b, delta_c : float
        Polymer parameters (currently not entering the initial data
        explicitly, but kept for future generalizations).

    Returns
    -------
    list
        Initial state vector [b_E(0), c_E(0), p_b(0), p_c(0)].
    """
    b0 = gamma
    c0 = 0.0
    p_c0 = gamma**2

    return [b0, c0, p_b0, p_c0]


def solve_instanton(M,
                    p_b0,
                    tau_max=80.0,
                    delta_b=delta_b_default,
                    delta_c=delta_c_default,
                    rtol=1e-10,
                    atol=1e-12):
    """
    Integrate the Euclidean equations of motion from the bounce to the
    horizon for a given shooting parameter p_b0.

    Parameters
    ----------
    M : float
        Black-hole mass.
    p_b0 : float
        Initial momentum p_b at the bounce (shooting parameter).
    tau_max : float
        Maximum Euclidean time to integrate if the horizon is not reached.
    delta_b, delta_c : float
        Polymer parameters.
    rtol, atol : float
        Relative and absolute tolerances for the ODE solver.

    Returns
    -------
    OdeResult
        SciPy OdeResult object containing the solution up to the horizon
        (if reached) or up to tau_max otherwise.
    """
    u0 = initial_data_bounce(M, p_b0, delta_b=delta_b, delta_c=delta_c)

    # Define RHS with fixed polymer parameters
    rhs = lambda tau, u: eom_tau(tau, u, delta_b=delta_b, delta_c=delta_c)

    # Event function capturing M through a lambda (no extra args to eom_tau)
    event = lambda tau, u: horizon_event(tau, u, M)

    sol = solve_ivp(
        rhs,
        (0.0, tau_max),
        u0,
        method="Radau",        # implicit, stiff-robust
        rtol=rtol,
        atol=atol,
        events=event,
        dense_output=False
    )

    return sol


def shooting_residual(p_b0,
                      M,
                      tau_max=80.0,
                      delta_b=delta_b_default,
                      delta_c=delta_c_default):
    """
    Shooting residual enforcing the boundary condition
    p_b(τ_H) = (2 G M)^2 at the horizon.

    Parameters
    ----------
    p_b0 : float
        Trial value for p_b at the bounce.
    M : float
        Black-hole mass.
    tau_max : float
        Maximum Euclidean time if horizon is not reached.
    delta_b, delta_c : float
        Polymer parameters.

    Returns
    -------
    float
        Residual p_b(τ_H) - (2 G M)^2.  The root of this function
        corresponds to a correctly matched instanton trajectory.
    """
    sol = solve_instanton(M, p_b0, tau_max=tau_max,
                          delta_b=delta_b, delta_c=delta_c)

    # If the horizon event did not trigger, penalize strongly
    if sol.t_events is None or len(sol.t_events[0]) == 0:
        # Take last point as "proxy" and return a large residual
        uH = sol.y[:, -1]
        p_bH = uH[2]
        target = (2.0 * G * M)**2
        return 10.0 * (p_bH - target)

    # Otherwise, evaluate p_b at the horizon
    uH = sol.y[:, -1]
    p_bH = uH[2]
    target = (2.0 * G * M)**2

    return p_bH - target


def find_shooting_data(M,
                       p_b0_bracket,
                       tau_max=80.0,
                       delta_b=delta_b_default,
                       delta_c=delta_c_default):
    """
    Use a robust root finder (bracketing method) to determine the
    correct p_b(0) that satisfies the horizon matching condition.

    Parameters
    ----------
    M : float
        Black-hole mass.
    p_b0_bracket : tuple(float, float)
        Bracketing interval [p_b0_min, p_b0_max] for the root of the
        shooting residual.
    tau_max : float
        Maximum Euclidean time to integrate.
    delta_b, delta_c : float
        Polymer parameters.

    Returns
    -------
    p_b0_star : float
        Converged shooting parameter.
    sol_star : OdeResult
        ODE solution corresponding to p_b0_star.
    """
    def residual_local(p_b0):
        return shooting_residual(p_b0, M, tau_max=tau_max,
                                 delta_b=delta_b, delta_c=delta_c)

    root = root_scalar(
        residual_local,
        bracket=p_b0_bracket,
        method="brentq",
        xtol=1e-10,
        rtol=1e-10,
        maxiter=80
    )

    if not root.converged:
        raise RuntimeError("Shooting root finder failed to converge.")

    p_b0_star = root.root
    sol_star = solve_instanton(M, p_b0_star, tau_max=tau_max,
                               delta_b=delta_b, delta_c=delta_c)

    return p_b0_star, sol_star


# =========================================================
# Euclidean action: bulk integral and boundary terms
# =========================================================

def compute_bulk_action(sol):
    """
    Compute the canonical bulk contribution to the Euclidean action:
        S_bulk = ∫ (p_b db_E/dτ + p_c dc_E/dτ) dτ
    using Simpson's rule (scipy.integrate.simpson) and a
    non-uniform time grid if necessary.

    Parameters
    ----------
    sol : OdeResult
        ODE solution along the instanton trajectory.

    Returns
    -------
    float
        Bulk contribution to the Euclidean action.
    """
    bE = sol.y[0, :]
    cE = sol.y[1, :]
    p_b = sol.y[2, :]
    p_c = sol.y[3, :]
    tau = sol.t

    # Numerical derivatives db_E/dτ and dc_E/dτ
    db_dtau = np.gradient(bE, tau)
    dc_dtau = np.gradient(cE, tau)

    integrand = p_b * db_dtau + p_c * dc_dtau

    S_bulk = simpson(integrand, tau)

    return S_bulk


def compute_boundary_terms(M, p_b0_star, sol):
    """
    Compute the boundary contribution to the Euclidean action from the
    polymer-corrected Gibbons–Hawking–York term.  In the canonical
    representation used here, this is encoded in a difference of
    p_b * b_E evaluated between bounce and horizon.

    Parameters
    ----------
    M : float
        Black-hole mass.
    p_b0_star : float
        Converged shooting parameter p_b(0) at the bounce.
    sol : OdeResult
        ODE solution corresponding to the instanton trajectory.

    Returns
    -------
    float
        Boundary contribution S_boundary to the Euclidean action.
    """
    # Horizon values
    bE_H = sol.y[0, -1]
    p_bH = sol.y[2, -1]

    # Bounce values
    bE_0 = gamma
    p_b_0 = p_b0_star

    norm = 1.0 / (G * gamma)

    S_H = norm * (p_bH * bE_H)
    S_0 = norm * (p_b_0 * bE_0)

    return S_H - S_0


def compute_full_action(M,
                        p_b0_bracket,
                        tau_max=80.0,
                        delta_b=delta_b_default,
                        delta_c=delta_c_default,
                        return_solution=False):
    """
    High-level driver to compute the on-shell Euclidean action S_E(M, δ)
    for given mass M and polymer parameters (delta_b, delta_c).

    Parameters
    ----------
    M : float
        Black-hole mass.
    p_b0_bracket : tuple(float, float)
        Initial bracketing interval for the shooting parameter p_b(0).
    tau_max : float
        Maximum Euclidean time for ODE integration.
    delta_b, delta_c : float
        Polymer parameters.
    return_solution : bool
        If True, also return the ODE solution and shooting data.

    Returns
    -------
    S_E : float
        Total on-shell Euclidean action.
    (optional) dict
        Additional data: {'M', 'delta_b', 'delta_c', 'p_b0', 'S_bulk',
        'S_boundary', 'solution'}.
    """
    p_b0_star, sol_star = find_shooting_data(
        M,
        p_b0_bracket,
        tau_max=tau_max,
        delta_b=delta_b,
        delta_c=delta_c
    )

    S_bulk = compute_bulk_action(sol_star)
    S_boundary = compute_boundary_terms(M, p_b0_star, sol_star)

    S_E = S_bulk + S_boundary

    if not return_solution:
        return S_E

    data = {
        "M": M,
        "delta_b": delta_b,
        "delta_c": delta_c,
        "p_b0": p_b0_star,
        "S_bulk": S_bulk,
        "S_boundary": S_boundary,
        "solution": sol_star,
    }
    return S_E, data


# =========================================================
# Parameter scans: mass and polymer dependence
# =========================================================

def scan_over_mass(masses,
                   p_b0_bracket_func,
                   tau_max=80.0,
                   delta_b=delta_b_default,
                   delta_c=delta_c_default):
    """
    Scan the Euclidean action S_E(M, δ) over a list/array of masses M.

    Parameters
    ----------
    masses : array-like
        Sequence of black-hole masses to evaluate.
    p_b0_bracket_func : callable
        Function M -> (p_b0_min, p_b0_max) providing a suitable
        bracketing interval for the shooting parameter as a function
        of mass.
    tau_max : float
        Maximum Euclidean time for integration.
    delta_b, delta_c : float
        Polymer parameters.

    Returns
    -------
    dict
        {'M_array', 'S_E_array'} with numpy arrays for further analysis.
    """
    M_array = np.array(masses, dtype=float)
    S_E_array = np.zeros_like(M_array)

    for i, M in enumerate(M_array):
        bracket = p_b0_bracket_func(M)
        S_E_array[i] = compute_full_action(
            M,
            bracket,
            tau_max=tau_max,
            delta_b=delta_b,
            delta_c=delta_c,
            return_solution=False
        )

    return {"M_array": M_array, "S_E_array": S_E_array}


def scan_over_polymer_delta(M,
                            deltas,
                            p_b0_bracket,
                            tau_max=80.0):
    """
    Scan the Euclidean action S_E(M, δ) as a function of a single
    polymer parameter δ (here taken to be delta_b = delta_c = δ for
    simplicity).

    Parameters
    ----------
    M : float
        Fixed black-hole mass.
    deltas : array-like
        Sequence of δ values to explore.
    p_b0_bracket : tuple(float, float)
        Bracketing interval for the shooting parameter at this mass.
    tau_max : float
        Maximum Euclidean time for integration.

    Returns
    -------
    dict
        {'delta_array', 'S_E_array'} with numpy arrays.
    """
    delta_array = np.array(deltas, dtype=float)
    S_E_array = np.zeros_like(delta_array)

    for i, delta in enumerate(delta_array):
        S_E_array[i] = compute_full_action(
            M,
            p_b0_bracket,
            tau_max=tau_max,
            delta_b=delta,
            delta_c=delta,
            return_solution=False
        )

    return {"delta_array": delta_array, "S_E_array": S_E_array}


# =========================================================
# Example driver
# =========================================================

if __name__ == "__main__":
    # Example usage: single mass, moderate δ
    M_example = 30.0
    delta_example = 0.03

    # Simple heuristic bracket for p_b0 as a function of M
    def bracket_func(M):
        # Center around (2GM)^2 with a modest ± factor
        p_central = (2.0 * G * M)**2
        return (0.5 * p_central, 1.5 * p_central)

    bracket_example = bracket_func(M_example)

    print("======================================================")
    print(" Euclidean instanton solver for black-hole interior")
    print("------------------------------------------------------")
    print(f" M        = {M_example:.6f}")
    print(f" delta    = {delta_example:.6f}")
    print(f" bracket  = [{bracket_example[0]:.6f}, {bracket_example[1]:.6f}]")
    print("======================================================")

    S_E_val, data = compute_full_action(
        M_example,
        bracket_example,
        tau_max=80.0,
        delta_b=delta_example,
        delta_c=delta_example,
        return_solution=True
    )

    print("\n[Instanton data]")
    print(f" p_b0*    = {data['p_b0']:.10e}")
    print(f" S_bulk   = {data['S_bulk']:.10e}")
    print(f" S_bound  = {data['S_boundary']:.10e}")
    print(f" S_E      = {S_E_val:.10e}")

    # Basic constraint-violation check along the trajectory
    sol = data["solution"]
    H_vals = np.array([H_E(sol.y[:, i], M_example,
                           delta_b=delta_example,
                           delta_c=delta_example)
                       for i in range(sol.y.shape[1])])
    max_H = np.max(np.abs(H_vals))
    print(f"\n[Diagnostics]")
    print(f" max |H_E| along trajectory  = {max_H:.3e}")
