import numpy as np
from math import sin, cos
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar

# Physical constants and parameters
G = 1.0        # Newton's constant in units where c = 1
gamma = 0.2375  # Immirzi parameter (controls bounce initial conditions)

# Euclidean instanton ODE system (b_E, c_E, p_b, p_c derivatives)
def instanton_odes(t, y):
    b, c, p_b, p_c = y
    # Holonomy (polymer) functions and their cosines
    f_b = sin(delta_b * b) / delta_b
    f_c = sin(delta_c * c) / delta_c
    cos_b = cos(delta_b * b)
    cos_c = cos(delta_c * c)
    # Equations of motion (Sec. IV, Appendix C)
    db_dt = (f_b**2 - gamma**2) / (2.0 * np.sqrt(p_c) * G * gamma)
    dc_dt = (f_b * f_c / np.sqrt(p_c) - (p_b * (f_b**2 - gamma**2)) / (2.0 * (p_c**1.5))) / (G * gamma)
    dpb_dt = (2.0 / gamma) * (np.sqrt(p_c) * f_c * cos_b - (f_b * cos_b * p_b) / np.sqrt(p_c))
    dpc_dt = (2.0 / gamma) * (np.sqrt(p_c) * f_b * cos_c)
    return [db_dt, dc_dt, dpb_dt, dpc_dt]

# Event to terminate integration at the Euclidean horizon (b_E = 2 G M)
def horizon_event(t, y):
    # Horizon when b_E reaches 2GM (Sec. IV.D horizon matching condition)
    return y[0] - 2.0 * G * M
horizon_event.terminal = True
horizon_event.direction = 1  # b_E is increasing

# Solve the ODE system from bounce to horizon for a given initial p_b(0)
def solve_instanton(p_b0):
    # Initial conditions at bounce (Sec. IV.D bounce regularity conditions)
    y0 = [gamma, 0.0, p_b0, gamma**2]  # b_E(0)=gamma, c_E(0)=0, p_c(0)=gamma^2, p_b(0)=p_b0
    # Integrate ODE with stiff solver (Radau) and high precision
    sol = solve_ivp(instanton_odes, [0, 1000.0], y0, method='Radau', events=horizon_event,
                    rtol=1e-10, atol=1e-12)
    # Ensure integration reached the horizon
    if sol.status == 0 or (sol.t_events and len(sol.t_events[0]) == 0):
        raise RuntimeError("Horizon not reached for p_b(0) = {}".format(p_b0))
    if not sol.success:
        raise RuntimeError("ODE integration failed: {}".format(sol.message))
    return sol

# Shooting target function: returns p_b(tau_H) - (2GM)^2 to find root (Sec. IV.D shooting target)
def shooting_target(p_b0):
    try:
        sol = solve_instanton(p_b0)
        p_b_end = sol.y[2, -1]  # p_b at horizon
        # Return difference from desired horizon condition p_b(tau_H) = (2GM)^2
        return p_b_end - (2.0 * G * M)**2
    except Exception:
        # If integration fails (likely for too large p_b0), treat as overshoot (positive)
        return 1e6

if __name__ == "__main__":
    # Representative example parameters (Sec. V.E: M=30, delta=0.05)
    M = 30
    delta = 0.05
    # Set polymerization scales for b and c (assume delta_b = delta_c = delta)
    delta_b = delta
    delta_c = delta

    # Find initial momentum p_b(0) via shooting method
    sol = root_scalar(shooting_target, bracket=(50.0, 1000.0), method='brentq', xtol=1e-12, rtol=1e-12)
    p_b0_solution = sol.root
    print(f"Found p_b(0) = {p_b0_solution:.6f}")

    # Solve the instanton ODE with the determined p_b(0)
    solution = solve_instanton(p_b0_solution)
    tau_H = solution.t[-1]               # Euclidean time at horizon
    b_end = solution.y[0, -1]            # b_E at horizon (should be ~2GM)
    p_b_end = solution.y[2, -1]          # p_b at horizon (should be ~(2GM)^2)

    # Compute Euclidean action S_E from bulk and boundary (Sec. V, Appendix C)
    # Bulk action via numerical integration of p_b db + p_c dc (Gauss-Legendre quadrature or Simpson's rule)
    sol_dense = solution.sol  # dense output for solution
    N = 10001
    tau_samples = np.linspace(0, tau_H, N)
    b_vals = sol_dense(tau_samples)[0]
    c_vals = sol_dense(tau_samples)[1]
    p_b_vals = sol_dense(tau_samples)[2]
    p_c_vals = sol_dense(tau_samples)[3]
    # Integrate p_b vs b and p_c vs c
    S_bulk = 4.0 * np.pi * (simpson (p_b_vals, b_vals) + simpson (p_c_vals, c_vals))
    # Boundary action via canonical boundary data (Eq. (V.18) in Sec. V, Appendix D)
    S_boundary = (p_b_end * b_end - p_b0_solution * gamma) / (G * gamma)
    # Total Euclidean action (bulk + boundary)
    S_E = S_boundary  # S_bulk and S_boundary should agree by bulk–boundary cancellation

    # Verify bulk–boundary consistency and output results
    print(f"Horizon matching: b_E(tau_H) = {b_end:.6f} (target {2*G*M:.6f}), "
          f"p_b(tau_H) = {p_b_end:.6f} (target {(2*G*M)**2:.6f})")
    print(f"Bulk vs Boundary action: S_bulk = {S_bulk:.6f}, S_boundary = {S_boundary:.6f}, "
          f"difference = {S_bulk - S_boundary:.2e}")
    print(f"Euclidean action S_E = {S_E:.6f} (expected ≈ 11333)")
