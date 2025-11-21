READM  for the Euclidean Instanton Black Hole Repository

This repository implements the computational framework for constructing Euclidean instantons in a reduced minisuperspace model inspired by loop quantum gravity (LQG). These instantons mediate quantum tunneling transitions from black hole interiors to white hole geometries, resolving classical singularities through polymerization corrections. The code directly supports the calculations in the paper "A Euclidean Instanton Connecting Black and White Hole Geometries: Existence, Regularity, and Action in a Reduced Quantum Gravitational Framework" (Paper I of a four-part series). By solving the boundary-value problem (BVP) for the Euclidean equations of motion, the code computes the instanton trajectory and the Euclidean action \(S_E(M)\), which determines the tunneling rate and black hole lifetime scaling.

This setup draws on fundamental concepts in black hole thermodynamics (e.g., Wick rotation for path integrals) and cosmology (e.g., bounce mechanisms avoiding big bang singularities), ensuring mathematical rigor through stiff ODE solvers and shooting methods.

Repository Structure
- `instanton_solver.py`: Python implementation using SciPy for ODE integration and shooting.
- (Optional) Data outputs: CSV files generated from runs (e.g., trajectory data for \(b_E(\tau_E)\), \(S_E\)).

 Physical Background
In the reduced Kantowski-Sachs minisuperspace, the Lorentzian black hole interior is Wick-rotated to a Euclidean sector with polymerization (holonomy corrections inspired by LQG). The code solves the first-order system:
\[
\frac{db_E}{d\tau} = \frac{1}{G} \frac{\sin(\lambda c_E)}{\lambda}, \quad \frac{dc_E}{d\tau} = \frac{1}{G} b_E \cos(\lambda c_E), \quad \dots
\]
with boundary conditions at the bounce (\(b_E = \gamma\), \(c_E = 0\), \(p_c = \gamma^2\)) and horizon (\(b_E = 2GM\), \(p_b = (2GM)^2\)). The Euclidean action \(S_E\) is computed via trapezoidal integration of the canonical terms, yielding the mass scaling \(S_E \propto M^{2+\delta}\) (with \(\delta \approx 0.1-0.18\)), crucial for tunneling lifetimes in quantum cosmology.

This model avoids singularities, providing a mathematically consistent saddle point for the gravitational path integral, with implications for black hole evaporation and gravitational-wave echoes.

Requirements
- **Python Version**: 3.8+ (tested on 3.12.3 as per available environments).
  - Libraries: NumPy, SciPy (for `solve_ivp` and numerical integration).

No additional installations are needed beyond these, as the code avoids external dependencies like pip installs for internet-restricted environments.

Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/euclidean-instanton-black-hole.git
   cd euclidean-instanton-black-hole
   ```
2. Ensure NumPy and SciPy are installed (pre-installed in many scientific environments).

Usage
Run the code to solve for the instanton and compute \(S_E\) for a given black hole mass \(M\).

- **Python Example**:
  ```python
  # Run the main block for M=30.0
  python instanton_solver.py
  ```
  Output (example):
  ```
  Converged pb0 = 178.245  # Approximate value; depends on guess
  Action S_E = 350.12     # Bulk action for M=30, λ=0.03
  ```
  Adjust `pb0_guess` in `find_solution` for convergence (e.g., start with `(2*G*M)**2 * 0.05`).

Key parameters:
- `M`: Black hole mass (e.g., 30.0 for testing).
- `λ`: Polymerization scale (default 0.03; tune for different quantum corrections).
- `γ`: Immirzi parameter (default 0.2375, fixed from black hole entropy in LQG).
- Numerical tolerances: `rtol=1e-10`, `atol=1e-12` for stiff ODEs.

For custom runs, modify `if __name__ == "__main__":` to sweep \(M\) values and plot \(S_E(M)\) scaling (use Matplotlib for visualization).

Numerical Diagnostics
- Constraint violation: Monitored via equivalent checks in Python; ensure \(|\mathcal{H}_E| < 10^{-9}\).
- Convergence: Shooting uses Newton iteration with finite-difference Jacobian; test with varying `eps` for stability.
- Potential issues: For small \(\lambda\), equations stiffen—use adaptive solvers like Radau. If horizon event not triggered, increase `τmax`.

Contributing
Contributions are welcome! Fork the repo, create a branch, and submit a pull request. Focus on extensions like integrating RWZ equations for gravitational-wave echoes (as in Paper III) or cosmological parameter sweeps (Paper IV). Ensure changes maintain mathematical consistency with the Euclidean Hamiltonian.

License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Acknowledgments
Inspired by works in loop quantum gravity (e.g., Ashtekar-Olmedo-Singh 2018) and black hole phenomenology. 

If you use this code in your research, please cite the associated paper. Contact the author for questions on black hole bounces or quantum cosmology applications.
