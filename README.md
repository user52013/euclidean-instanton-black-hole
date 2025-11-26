________________________________________
README.md — Clean GitHub Markdown Version (No Formatting Issues)
________________________________________
Euclidean Instanton for Black-to-White Hole Tunneling
High-precision numerical implementation accompanying:
“A Euclidean Instanton Connecting Black- and White-Hole Geometries:
Existence, Regularity, and Action in a Reduced Quantum-Gravitational Framework.”
This repository contains all numerical tools required to reproduce the instanton construction, including the boundary-value solution, the Euclidean action, and parameter scans used for the results in Sec. V–VIII and Appendix C of the paper.
The numerical implementation matches the procedures described in Sec. IV and the accuracy tests in Appendix C.
________________________________________
1. Overview
The solver computes a Euclidean trajectory interpolating between:
•	a regular “bounce’’ surface inside the Kantowski–Sachs interior, and
•	the matching hypersurface defined by b_E = 2GM.
The evolution equations follow from the polymer-corrected Euclidean Hamiltonian (Appendix A).
A tunable polymer parameter delta controls the holonomy modification.
The Euclidean action is evaluated using:
•	Radau IIA implicit integration (stiff ODEs)
•	high-order Gauss–Legendre quadrature for the bulk action
•	the polymer-corrected Gibbons–Hawking–York boundary term (Appendix D)
All diagnostics—constraint violation, horizon matching, and spectral convergence—follow the reproducibility checklist in Sec. IV.G and Appendix C.5.
________________________________________
2. Code layout
```bash
instanton_solver.py   # Main solver and action evaluation
README.md             # This file
```
Core functions
Function	Purpose
solve_instanton(M, delta, p_b0)	Integrates the Euclidean equations up to the horizon.
shooting_residual(p_b0, M, delta)	Boundary mismatch at b_E = 2GM.
find_shooting_solution(M, delta)	Shooting method to determine the correct p_b(0).
compute_full_action(sol, M, delta, p_b0)	Evaluates the Euclidean action from bulk + boundary terms.
scan_parameters(M_list, delta_list)	Produces tables for mass and polymer-parameter scans.
________________________________________
3. Requirements
Only standard scientific Python packages are required:
```nginx
numpy
scipy
```
Install with:
pip install numpy scipy
________________________________________
4. Example: computing the instanton for M = 30
```python
from instanton_solver import (
    find_shooting_solution,
    solve_instanton,
    compute_full_action,
)

M = 30.0
delta = 0.05

p_b0 = find_shooting_solution(M, delta)
sol = solve_instanton(M, delta, p_b0)

S_E, S_bulk = compute_full_action(sol, M, delta, p_b0)

print("Initial momentum p_b(0):", p_b0)
print("Total on-shell action S_E:", S_E)
```
Expected checks (Sec. V and Appendix C):
•	shooting residual below 1e-10
•	constraint violation below 1e-9
•	Gauss–Legendre vs. canonical bulk action consistent to better than 1e-6
•	matches the polynomial fit in Sec. V.D
________________________________________
5. Parameter scan (mass and polymer scale)
```python
from instanton_solver import scan_parameters

M_list = [10, 20, 30, 40]
delta_list = [0.00, 0.02, 0.05, 0.10]

results = scan_parameters(M_list, delta_list)

for item in results:
    print(item["M"], item["delta"], item["S_E"])
```
This reproduces the mass and delta scaling
discussed in Sec. V.D:
```mathematica
S_E(M,delta) =
alpha_0 M^2 * (1 + kappa_1 delta + kappa_2 delta^2 + O(delta^3))
```
________________________________________
6. Numerical accuracy and diagnostics
The solver implements all tests in the reproducibility checklist
(Sec. IV.G and Appendix C.5):
•	constraint violation
max|C_E(sol)| < 1e-9
•	horizon matching
|p_b(tau_H) − (2GM)^2| < 1e-10
•	Gauss–Legendre vs. canonical bulk action
agreement to better than 1e-6
•	stiff integrator tolerance
rtol = 1e-12, atol = 1e-12
All instanton trajectories used in Sec. V and Sec. VII satisfy these accuracy thresholds.
________________________________________
7. Relation to the paper
This repository implements the numerical procedures associated with:
•	Sec. IV — Euclidean instanton construction
•	Sec. V — Euclidean action and delta-expansion
•	Sec. VII — phenomenological applications
•	Appendix A — constraint algebra
•	Appendix C — numerical methods and convergence
•	Appendix D — GHY boundary term
All figures and tables in Sec. V can be regenerated from the scripts here.
________________________________________
8. Citation
If you use this code, please cite the associated paper:
“A Euclidean Instanton Connecting Black- and White-Hole Geometries:
Existence, Regularity, and Action in a Reduced Quantum-Gravitational Framework.”
________________________________________
9. License
MIT License (or another license of your choice).
________________________________________

