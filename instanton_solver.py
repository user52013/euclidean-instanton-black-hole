import numpy as np
from numpy import sin, cos, sqrt
from scipy.integrate import solve_ivp, simps
from scipy.optimize import root_scalar

# -----------------------------------------------------
# Physical / model parameters (can be modified/scanned)
# -----------------------------------------------------
G = 1.0                 # units with G = 1
gamma = 0.2375          # Barbero–Immirzi parameter
delta_b = 0.03          # polymer scale for b
delta_c = 0.03          # polymer scale for c

# --------------------------------------------------------------------
# Euclidean polymer Hamiltonian C^{(E,δ)} in KS minisuperspace
# u = (b, p_b, c, p_c)
# --------------------------------------------------------------------
def H_E(u):
    """
    Euclidean polymer Hamiltonian constraint C^{(E,δ)} for the
    Kantowski–Sachs interior.

    Parameters
    ----------
    u : array_like, shape (4,)
        Phase-space vector (b, p_b, c, p_c).

    Returns
    -------
    float
        Value of the Euclidean Hamiltonian constraint.
    """
    b, p_b, c, p_c = u
    # Guard against negative p_c roundoff
    pc_abs = np.abs(p_c) + 1e-18

    fb = sin(delta_b * b) / delta_b
    fc = sin(delta_c * c) / delta_c

    CE = -1.0 / (2.0 * G * gamma**2) * (
        -2.0 * sqrt(pc_abs) * fb * fc
        + (fb**2 - gamma**2) * p_b / sqrt(pc_abs)
    )
    return CE


# --------------------------------------------------------------------
# Euclidean equations of motion (stiff ODE system)
# --------------------------------------------------------------------
def eom_tau(tau, u):
    """
    Euclidean equations of motion in τ for u = (b, p_b, c, p_c).

    The equations are derived from H_E via:
      dot{q_i} = {q_i, H_E}
      dot{p_i} = {p_i, -H_E}
    with Poisson brackets:
      {b, p_b} = G γ,   {c, p_c} = 2 G γ
    """
    b, p_b, c, p_c = u
    pc_abs = np.abs(p_c) + 1e-18

    fb = sin(delta_b * b) / delta_b
    fc = sin(delta_c * c) / delta_c

    # ∂H/∂p_b and ∂H/∂p_c
    dH_dp_b = -1.0 / (2.0 * G * gamma**2) * (fb**2 - gamma**2) / sqrt(pc_abs)
    dH_dp_c = -1.0 / (2.0 * G * gamma**2) * (
        -2.0 * fb * fc * (0.5 / sqrt(pc_abs))
        - 0.5 * (fb**2 - gamma**2) * p_b / (pc_abs**1.5)
    )

    # ∂H/∂b
    dfb_db = cos(delta_b * b)
    dH_db = -1.0 / (2.0 * G * gamma**2) * (
        -2.0 * sqrt(pc_abs) * dfb_db * fc
        + 2.0 * fb * dfb_db * p_b / sqrt(pc_abs)
    )

    # ∂H/∂c
    dfc_dc = cos(delta_c * c)
    dH_dc = -1.0 / (2.0 * G * gamma**2) * (
        -2.0 * sqrt(pc_abs) * fb * dfc_dc
    )

    # Hamilton's equations (Euclidean)
    dot_b =  G * gamma      * dH_dp_b
    dot_c =  2.0 * G * gamma * dH_dp_c
    dot_p_b = -G * gamma     * dH_db
    dot_p_c = -2.0 * G * gamma * dH_dc

    return np.array([dot_b, dot_p_b, dot_c, dot_p_c], dtype=float)


# --------------------------------------------------------------------
# Event function: stop integration when b reaches the horizon 2GM
# --------------------------------------------------------------------
def horizon_event(tau, u, M):
    """
    Event function to detect when b = 2GM.

    Parameters
    ----------
    tau : float
        Euclidean time.
    u : array_like
        State vector (b, p_b, c, p_c).
    M : float
        Black-hole mass.

    Returns
    -------
    float
        b - 2GM. The event is triggered when this crosses zero.
    """
    b = u[0]
    return b - 2.0 * G * M

horizon_event.terminal = True
horizon_event.direction = +1.0


# --------------------------------------------------------------------
# Initial conditions at the bounce (τ = 0)
# --------------------------------------------------------------------
def initial_conditions(M, p_b0):
    """
    Initial conditions at the bounce surface τ = 0.

    We choose:
      b(0)   = b_min ≈ γ
      c(0)   = 0
      p_c(0) = b_min^2
      p_b(0) = p_b0  (shooting parameter)

    Parameters
    ----------
    M : float
        Black-hole mass (not directly used here, but kept for clarity).
    p_b0 : float
        Initial guess for p_b at the bounce.

    Returns
    -------
    ndarray, shape (4,)
        Initial state vector u0 = (b0, p_b0, c0, p_c0).
    """
    b_min = gamma  # minimal radius at bounce (model assumption)
    c0 = 0.0
    p_c0 = b_min**2
    u0 = np.array([b_min, p_b0, c0, p_c0], dtype=float)
    return u0


# --------------------------------------------------------------------
# Solve instanton for a given M and shooting parameter p_b0
# --------------------------------------------------------------------
def solve_instanton(M, p_b0, tau_max=10.0, rtol=1e-12, atol=1e-14):
    """
    Solve the Euclidean ODE from the bounce to the horizon
    using a stiff implicit integrator (Radau).

    Parameters
    ----------
    M : float
        Black-hole mass.
    p_b0 : float
        Shooting parameter p_b(0).
    tau_max : float, optional
        Maximum Euclidean time if event is not triggered.
    rtol, atol : float
        Relative and absolute tolerances.

    Returns
    -------
    sol : OdeResult
        SciPy solve_ivp result, stopped at b = 2GM if event succeeds.
    """
    u0 = initial_conditions(M, p_b0)

    def rhs(tau, u):
        return eom_tau(tau, u)

    events = lambda tau, u: horizon_event(tau, u, M)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, tau_max),
        y0=u0,
        method="Radau",
        rtol=rtol,
        atol=atol,
        events=events,
        dense_output=False,
    )

    if sol.status == 1 and sol.t_events[0].size > 0:
        # Truncate solution precisely at event time for consistency
        t_end = sol.t_events[0][0]
        mask = sol.t <= t_end + 1e-12
        sol.t = sol.t[mask]
        sol.y = sol.y[:, mask]

    return sol


# --------------------------------------------------------------------
# Residual for shooting: enforce p_b(τ_H) ≈ (2GM)^2
# --------------------------------------------------------------------
def shooting_residual(p_b0, M):
    """
    Residual function for shooting on p_b0.

    We enforce:
      p_b(τ_H) ≈ (2GM)^2
    where τ_H is when b = 2GM.

    Parameters
    ----------
    p_b0 : float
        Initial value of p_b at τ = 0.
    M : float
        Black-hole mass.

    Returns
    -------
    float
        Residual R = p_b(τ_H) - (2GM)^2.
    """
    sol = solve_instanton(M, p_b0)
    bH, p_bH, cH, p_cH = sol.y[:, -1]
    target = (2.0 * G * M)**2
    return p_bH - target


# --------------------------------------------------------------------
# Find p_b0 by root-finding (secant method)
# --------------------------------------------------------------------
def find_pb0(M, bracket=(1.0, 50.0)):
    """
    Find the shooting parameter p_b0 such that p_b(τ_H)=(2GM)^2.

    Parameters
    ----------
    M : float
        Black-hole mass.
    bracket : tuple of float
        Initial guesses (p_b0_left, p_b0_right) for the secant method.

    Returns
    -------
    p_b0_root : float
        Root value of p_b0.
    """
    a, b = bracket

    def f(pb0):
        return shooting_residual(pb0, M)

    sol_root = root_scalar(
        f,
        x0=a,
        x1=b,
        method="secant",
        rtol=1e-10,
        maxiter=50,
    )

    if not sol_root.converged:
        raise RuntimeError(
            f"Shooting for p_b0 did not converge for M={M}. "
            f"Last status: root={sol_root.root}, flag={sol_root.flag}"
        )
    return sol_root.root


# --------------------------------------------------------------------
# High-accuracy Euclidean action S_E (bulk + GHY)
# --------------------------------------------------------------------
def compute_bulk_action(sol):
    """
    Compute the bulk part of the Euclidean action:

      S_E^{bulk} = ∫ (p_b \dot{b} + p_c \dot{c}) dτ

    using Simpson's rule (scipy.integrate.simps) and a
    manually constructed high-order finite-difference derivative
    (no np.gradient).

    Parameters
    ----------
    sol : OdeResult
        Solution object from solve_instanton.

    Returns
    -------
    float
        Bulk Euclidean action S_E^{bulk}.
    """
    tau = sol.t
    b, p_b, c, p_c = sol.y

    n = tau.size

    # Central finite differences for interior points (O(h^2))
    db_dt = np.zeros_like(b)
    dc_dt = np.zeros_like(c)

    # Interior points
    db_dt[1:-1] = (b[2:] - b[:-2]) / (tau[2:] - tau[:-2])
    dc_dt[1:-1] = (c[2:] - c[:-2]) / (tau[2:] - tau[:-2])

    # One-sided differences at boundaries
    db_dt[0]  = (b[1]  - b[0])  / (tau[1]  - tau[0])
    db_dt[-1] = (b[-1] - b[-2]) / (tau[-1] - tau[-2])

    dc_dt[0]  = (c[1]  - c[0])  / (tau[1]  - tau[0])
    dc_dt[-1] = (c[-1] - c[-2]) / (tau[-1] - tau[-2])

    integrand = p_b * db_dt + p_c * dc_dt

    S_bulk = simps(integrand, tau)
    return float(S_bulk)


def compute_GHY_action(sol):
    """
    Compute the polymer-corrected Euclidean GHY boundary term:

      S_GHY^{(E,δ)} = -(1/(Gγ)) |p_c| sin(δ c)/δ

    evaluated at the horizon.

    Parameters
    ----------
    sol : OdeResult

    Returns
    -------
    float
        GHY contribution to the Euclidean action.
    """
    bH, p_bH, cH, p_cH = sol.y[:, -1]
    pc_abs = np.abs(p_cH)
    S_GHY = -(1.0 / (G * gamma)) * pc_abs * sin(delta_c * cH) / delta_c
    return float(S_GHY)


def compute_full_action(sol):
    """
    Compute the full Euclidean action:

      S_E = S_E^{bulk} + S_GHY^{(E,δ)}

    using:
      - Simpson's rule for the bulk integral,
      - polymer-corrected GHY boundary term.

    Returns
    -------
    float
        Total Euclidean action S_E.
    """
    S_bulk = compute_bulk_action(sol)
    S_GHY = compute_GHY_action(sol)
    return S_bulk + S_GHY


# --------------------------------------------------------------------
# Parameter scan over (M, delta_b, gamma)
# --------------------------------------------------------------------
def scan_parameters(
    masses,
    deltas,
    gammas,
    pb0_bracket=(1.0, 50.0),
):
    """
    Systematically scan parameter space (M, δ, γ) and compute S_E.

    Parameters
    ----------
    masses : array_like
        List or array of black-hole masses M to scan.
    deltas : array_like
        List or array of polymer scales δ (we set δ_b=δ_c=δ).
    gammas : array_like
        List or array of Immirzi parameters γ to scan.
    pb0_bracket : tuple
        Bracket for shooting parameter p_b0.

    Returns
    -------
    results : dict
        Dictionary with keys:
          'M', 'delta', 'gamma', 'S_E'
        each a list of values with the same length.
    """
    global delta_b, delta_c, gamma

    masses = np.atleast_1d(masses)
    deltas = np.atleast_1d(deltas)
    gammas = np.atleast_1d(gammas)

    Ms_list = []
    deltas_list = []
    gammas_list = []
    S_list = []

    for M in masses:
        for delt in deltas:
            for gam in gammas:
                delta_b = float(delt)
                delta_c = float(delt)
                gamma = float(gam)

                # Find p_b0 by shooting
                p_b0 = find_pb0(M, bracket=pb0_bracket)
                sol = solve_instanton(M, p_b0)
                S_E = compute_full_action(sol)

                Ms_list.append(M)
                deltas_list.append(delt)
                gammas_list.append(gam)
                S_list.append(S_E)

                print(
                    f"[scan] M={M:.3f}, δ={delt:.4f}, γ={gam:.4f} "
                    f"→ p_b0={p_b0:.6g}, S_E={S_E:.6g}"
                )

    return {
        "M": np.array(Ms_list),
        "delta": np.array(deltas_list),
        "gamma": np.array(gammas_list),
        "S_E": np.array(S_list),
    }


# --------------------------------------------------------------------
# Example run
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Single example: M = 30, default δ, γ
    M = 30.0
    print("Solving instanton for M =", M)
    p_b0 = find_pb0(M, bracket=(1.0, 50.0))
    sol = solve_instanton(M, p_b0)
    S_E = compute_full_action(sol)

    print("Converged p_b0 =", p_b0)
    print("Euclidean action S_E =", S_E)

    # Optional parameter scan (uncomment to use):
    # masses = [10.0, 20.0, 30.0, 40.0]
    # deltas = [0.02, 0.03, 0.05]
    # gammas = [0.20, 0.2375, 0.30]
    # results = scan_parameters(masses, deltas, gammas)
    # np.savez("instanton_scan_results.npz", **results)
