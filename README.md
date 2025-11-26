________________________________________
README — Euclidean Instanton Solver for the Polymer-Corrected KS Interior
This repository contains the Python implementation used in our work on Euclidean instantons in the polymer-corrected Kantowski–Sachs (KS) interior.
The code integrates the Euclidean equations of motion, applies the horizon matching condition through a shooting procedure, and evaluates the on-shell Euclidean action (including the polymer-corrected GHY term). The numerics follow the scheme described in Sec. V and Appendix C of the paper.
The goal is to provide a clean and reproducible reference implementation. The script was written to be readable rather than optimized, and it avoids external numerical frameworks beyond numpy and scipy.
________________________________________
1. Getting Started
The code runs on a standard Python scientific stack.
A minimal environment is:
python3 -m pip install numpy scipy
After that, the script can be executed directly:
python3 instanton_solver.py
The default run solves the instanton for ( M=30 ) with a reasonable initial guess for ( p_b(0) ).
The solver automatically finds the correct shooting value and prints the matching accuracy and the resulting Euclidean action.
________________________________________
2. What the Script Does
The main stages are:
1.	Integrate the Euclidean EOM using an implicit Radau method.
This avoids stiffness issues that appear for small polymer scales.
2.	Locate the matching surface ( b_E = 2GM ).
This is implemented as a zero-crossing event.
3.	Solve the shooting problem to determine the initial momentum ( p_b(0) ).
The script uses a standard one-dimensional root finder.
4.	Evaluate the Euclidean action from the canonical boundary expression
derived in Sec. V and Appendix D.
5.	(Optional) Parameter scans.
The script includes a helper routine that can sweep over masses and polymer
scales. This was used to check the scaling
[
S_E(M,\delta) \sim \alpha_0,M^2\bigl[1 + \kappa_1 \delta + \kappa_2 \delta^2 + \cdots\bigr]
]
reported in Sec. V.
________________________________________
3. Reproducing Figures and Tables in the Paper
The output of the code can be used to reproduce:
•	the mass dependence of the action (( S_E ) vs. ( M )),
•	the small-( \delta ) scaling discussed in Sec. V.D,
•	the numerical diagnostics summarized in Appendix C.5.
For scans, the suggested pattern is:
from instanton_solver import scan_parameters

M_list     = [10, 20, 30, 40, 60]
delta_list = [0.01, 0.03, 0.05]

results = scan_parameters(M_list, delta_list)
The returned dictionary contains, for each pair ((M,\delta)):
•	the converged shooting value ( p_b(0) ),
•	the horizon radius,
•	the Euclidean action,
•	the bulk contribution (useful for cross-checks).
This is the same information used to generate the plots that appear in Sec. V.
________________________________________
4. Notes on Numerical Accuracy
The solver was written to follow the convergence tests described in Appendix C.
A few practical comments may be helpful for anyone extending or modifying the code:
•	The Radau method is noticeably more stable than RK45 for this system.
•	The horizon event becomes sensitive when ( \delta ) is small; tightening tolerances can help.
•	The action is evaluated from boundary data rather than a bulk numerical integral, which avoids the noise associated with differentiating the trajectory.
•	The shooting root finder usually converges quickly if the mass is not too small; for Planck-scale masses the dynamics become sharper and sometimes require a better initial guess.
In general, the code should reproduce our published results to within the quoted precision.
________________________________________
5. Citing the Code
If you use this implementation in your own work, please cite the accompanying paper:
[Your citation block here — same as in the repository bib file]
________________________________________
6. Contact
Questions, discussions, or pull requests are welcome.
This repository is meant to serve as a transparent and reproducible reference for the instanton construction described in the paper.
________________________________________

