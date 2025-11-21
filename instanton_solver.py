import numpy as np
from scipy.integrate import solve_ivp
from numpy import sin, cos
from scipy.optimize import newton

# -----------------------------------------------------
# Physical constants
# -----------------------------------------------------
G = 1.0                     # Gravitational constant (set to 1 for simplicity)
gamma = 0.2375              # Immirzi parameter (regulator for b_min)
λ = 0.03                    # Polymer scale (lambda)

# -----------------------------------------------------
# Euclidean Hamiltonian (polymerized)
# -----------------------------------------------------
def H_E(u):
    """Euclidean Hamiltonian constraint H_E(bE, cE, p_b, p_c)."""
    bE, cE, p_b, p_c = u
    # H_E is proportional to: term1 + term2 + U
    term1 = 2 * bE * sin(λ * cE) / λ
    term2 = (sin(λ * bE) / λ)**2
    # U = 1 - p_b**2 / |p_c|. p_c is positive at bounce.
    U = 1 - p_b**2 / abs(p_c) 
    return - (term1 + term2 + U) / (2 * G)

# -----------------------------------------------------
# Euclidean equations of motion
# -----------------------------------------------------
def eom(t, u):
    """Euclidean EOMs: du/dτ = {u, H_E}"""
    bE, cE, p_b, p_c = u
    
    # dbE/dτ = {bE, H_E}
    dbE = (1/G) * (sin(λ * cE) / λ)
    # dcE/dτ = {cE, H_E}
    dcE = (1/G) * (bE * cos(λ * cE))
    # dpb/dτ = -{bE, H_E}
    dpb = -(1/(2*G)) * sin(2*λ*bE) / λ
    # dpc/dτ = -{cE, H_E}
    dpc =  (1/(2*G)) * (p_b**2 / p_c**2)
    
    return [dbE, dcE, dpb, dpc]

# -----------------------------------------------------
# Horizon event: bE = 2GM (r=2M)
# -----------------------------------------------------
def horizon_event(t, u, M):
    """Integration event for the Horizon matching surface."""
    return u[0] - 2*G*M
horizon_event.terminal = True # Stop integration when triggered
horizon_event.direction = +1 # Trigger when u[0] is increasing

# -----------------------------------------------------
# Shooting method: one parameter pb0
# -----------------------------------------------------
def solve_instanton(M, pb0):
    """Solves the Euclidean EOMs from bounce to horizon."""
    # Bounce regularity conditions at tau=0
    b0 = gamma
    c0 = 0.0
    p_c0 = gamma**2
    u0 = [b0, c0, pb0, p_c0]

    # Time span: [tau_0, tau_max]
    sol = solve_ivp(
        eom,
        [0, 200],                  # max integration time
        u0,
        events=lambda t,u: horizon_event(t,u,M),
        rtol=1e-10, atol=1e-12,    # high precision for stiff ODEs
        method="Radau"             # implicit method for stiff problems
    )
    # Check if the event was found
    if len(sol.t_events[0]) == 0:
        # If the horizon was not reached, return a solution with a large residual
        # This prevents the Newton solver from being trapped by unsuccessful runs.
        uH = sol.y[:, -1]
        p_bH = uH[2]
        # Return a non-converged solution object
        sol.y = np.array([[uH[0]], [uH[1]], [p_bH], [uH[3]]])
        sol.t = np.array([sol.t[-1]])
    
    return sol

# -----------------------------------------------------
# Residual for shooting: match p_b(τ_H) = (2GM)^2
# -----------------------------------------------------
def residual(pb0, M):
    """Calculates the residual (mismatch) at the Horizon."""
    sol = solve_instanton(M, pb0)
    
    # Check if the Horizon event was detected
    if len(sol.t) < 2 and len(sol.t_events[0]) == 0:
        # Penalize solutions that fail to reach the horizon strongly
        return 1e5
        
    uH = sol.y[:, -1]
    p_bH = uH[2]
    
    # Boundary condition: p_b(τ_H) = (2GM)^2
    return p_bH - (2*G*M)**2

# -----------------------------------------------------
# Newton shooting (using scipy.optimize.newton for robustness)
# -----------------------------------------------------
def find_solution(M, pb0_guess):
    """Finds the correct initial momentum pb0 using Newton's method."""
    
    # Use scipy's robust Newton solver
    pb0_converged = newton(
        func=residual,
        x0=pb0_guess,
        args=(M,),
        tol=1e-10,
        maxiter=30
    )
    return pb0_converged

# -----------------------------------------------------
# Compute Euclidean action S_E (Total Action including GHY)
# -----------------------------------------------------
def compute_full_action(sol, pb0):
    """
    Computes the total on-shell Euclidean action S_E.
    
    S_E is computed via the difference of canonical boundary terms:
    S_E = 1/(G*gamma) * [p_b * b_E]_(tau=0)^(tau=tau_H)
    This form is equivalent to the canonical bulk integral S_bulk + S_GHY on-shell.
    """
    # ----------------------------------------------------------------------
    # 1. Canonical Bulk Action (S_bulk) - retained for check/comparison
    # ----------------------------------------------------------------------
    bE = sol.y[0]
    cE = sol.y[1]
    pb = sol.y[2]
    pc = sol.y[3]
    τ  = sol.t
    
    # Numerical derivatives: db/dτ and dc/dτ
    # np.gradient handles non-uniform time steps (sol.t)
    db = np.gradient(bE, τ)
    dc = np.gradient(cE, τ)
    
    integrand = pb * db + pc * dc
    S_bulk = np.trapz(integrand, τ) # Canonical Bulk Action S_bulk = ∫(p*dq) dτ

    # ----------------------------------------------------------------------
    # 2. Total On-shell Action S_E (including GHY boundary term)
    # ----------------------------------------------------------------------
    # Factor 1/(G*gamma) comes from the canonical phase space normalization
    norm_factor = 1.0 / (G * gamma)

    # Horizon Term (tau_H)
    uH = sol.y[:, -1]
    bE_H = uH[0]
    pb_H = uH[2]
    S_H_term = norm_factor * (pb_H * bE_H)

    # Bounce Term (tau=0)
    # At bounce: bE(0) = gamma, pb(0) = pb0
    bE_0 = gamma
    pb_0 = pb0
    S_0_term = norm_factor * (pb_0 * bE_0)
    
    S_E_Total = S_H_term - S_0_term

    return S_E_Total, S_bulk

# -----------------------------------------------------
# Example run
# -----------------------------------------------------
if __name__ == "__main__":
    M = 30.0
    pb0_guess = (2.0 * G * M)**2 * 0.05 # Initial guess for M=30.0
    
    print(f"--- Euclidean Instanton Solver (M={M}) ---")

    try:
        # 1. Find the initial momentum pb0 using the shooting method
        pb0_converged = find_solution(M, pb0_guess)
        
        # 2. Re-solve the ODE with the converged pb0
        sol = solve_instanton(M, pb0_converged)

        # 3. Compute the action
        S_E_Total, S_bulk = compute_full_action(sol, pb0_converged)

        print("\n[Shooting Results]")
        print(f"Converged initial momentum (pb0): {pb0_converged:.8f}")
        
        # Check Horizon Matching: p_b(τ_H) should be (2GM)^2
        p_bH = sol.y[2, -1]
        target_pbH = (2*G*M)**2
        match_residual = p_bH - target_pbH
        print(f"Horizon Radius (bE_H):           {sol.y[0, -1]:.8f}")
        print(f"p_b(τ_H) mismatch (Residual):   {match_residual:.2e}")
        
        print("\n[Action Results]")
        # S_E_Total is the physical tunneling action
        print(f"Total On-shell Action (S_E):      {S_E_Total:.8f}")
        # S_bulk is the canonical bulk integral (for comparison)
        print(f"Canonical Bulk Action (S_bulk):   {S_bulk:.8f}")
        
        # The two values should be close but not identical due to numerical integration vs. boundary evaluation.
        
    except RuntimeError as e:
        print(f"\nERROR: {e}")
    except ValueError as e:
        print(f"\nERROR: Shooting failed due to non-convergence or ODE failure. Details: {e}")
