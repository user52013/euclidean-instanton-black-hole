
# Euclidean Instanton Solver for Polymer-Corrected Black-Hole Interiors

This repository contains a self-contained Python implementation of the Euclidean instanton construction developed in our accompanying manuscript.
The code numerically solves the polymer-corrected Euclidean equations of motion for the Kantowski–Sachs interior, enforces the bounce–to–horizon boundary conditions, and evaluates the full Euclidean action (bulk + polymer GHY term) with high numerical accuracy.

The goal of this implementation is twofold:

1. **Provide a transparent, fully reproducible numerical realization** of the methods described in Sec. IV–V and Appendix C of the paper.
2. **Support parameter studies** (mass (M), polymer scale (\delta)) relevant for the asymptotic scaling
   [
   S_E(M,\delta) \propto M^{,2+\delta_{\mathrm{eff}}},
   ]
   and for assessing the size of non-perturbative quantum-gravity corrections.



## Overview

The solver implements the full Euclidean system:

* Polymer-corrected Hamiltonian constraint
* Bounce regularity conditions
* Horizon matching condition (b_E(\tau_H) = 2GM)
* Correct canonical bulk action
* Polymer-corrected Euclidean GHY term from Appendix D
* High-precision quadrature for (S_E)

The numerical evolution relies on:

* **A stiff ODE integrator** (Radau IIA)
* **Root-finding for the shooting parameter (p_b(0))**
* **Optional parameter scans** over (M) and (\delta)
* **Consistency checks** reflecting Appendix C diagnostics

The code is designed to run both locally and under GitHub Actions (CI).
The CI test confirms that the solver reproduces the reference value of the Euclidean action for:

[
(M,\delta) = (30,; 0.05).
]



## File: `instanton_solver.py`

The solver includes the following key components.

### 1. Euclidean Equations of Motion

The ODE system follows Sec. IV.B, with holonomy substitutions applied to both (b_E) and (c_E).
The Euclidean evolution is stiff near the bounce; Radau IIA handles this reliably.

### 2. Shooting Method

The horizon condition requires:

[
p_b(\tau_H) = (2GM)^2.
]

The solver determines the correct initial value (p_b(0)) by a robust bracketing method (Brent).

### 3. Horizon Event Detection

The ODE integration terminates automatically when:

[
b_E(\tau_E) = 2GM,
]

implementing the horizon matching discussed in Sec. IV.D.

### 4. Euclidean Action Evaluation

The total action is computed as:

[
S_E = S_{\text{bulk}} + S_{\text{GHY}}^{(E,\delta)},
]

where:

* (S_{\text{bulk}}) is the canonical integral
* (S_{\text{GHY}}^{(E,\delta)}) follows directly from Appendix D
* High-order Gauss–Legendre quadrature ensures rapid convergence

This matches the normalization used throughout the final version of the paper.

### 5. CI Test

When run as a standalone script, the solver:

1. Computes the instanton for (M=30), (\delta=0.05)
2. Prints summary diagnostics
3. Checks whether the computed action agrees with the reference value within a prescribed tolerance

This supports automated reproducibility verification.



## How to Run

### Local run

```bash
python instanton_solver.py
```

### Expected output (truncated example)

```
--- Euclidean instanton solver: M=30.0, delta=0.05 ---

[Shooting]
  p_b(0) = 7.13e+03
  Horizon reached at τ_H = ...

[Action Summary]
  S_E (Total)    = 1.84987e+05
  |S_E - S_bulk| = 1e-7

[CI_VALIDATION] SUCCESS: ...
```

The value of (S_E) above reflects the **fully normalized canonical action** (Sec. V.E and Appendix D).
Earlier versions of the paper used a bulk-only normalization; the present definition is consistent with the variational principle and is the one used for all final results.



## Parameter Scans

You may import the solver and run systematic scans:

```python
from instanton_solver import run_scan

run_scan(
    M_values=[20, 30, 40, 60],
    delta_values=[0.02, 0.05, 0.08],
    output="scan_results.csv"
)
```

This reproduces the mass and polymer-scale dependence discussed in Sec. V.



## Relevance to the Paper

This implementation directly supports:

* **Sec. V.B–E:**
  Numerical solution of the Euclidean trajectory and evaluation of the action.

* **Appendix C:**
  Stiff ODE integration, convergence tests, spectral cross-checks, and Gauss–Legendre quadrature.

* **Appendix D:**
  Polymer-corrected GHY term and its contribution to (S_E).

All figures and tables referring to numerical results were generated using this code or its minimal variants.

---

## Reproducibility

The implementation has been verified against:

* Multiple ODE solvers (Radau IIA, BDF family)
* Alternative bracketing intervals for the shooting parameter
* Step-size refinement tests
* Bulk–boundary consistency
* δ-expansion tests
* Spectral–ODE agreement (Appendix C.5)

The GitHub CI pipeline automatically re-runs the solver and confirms agreement with the reference action.



## Citation

If you use this code in your own work, please cite the accompanying manuscript.


