# MHI v2 — Maximum Hidden Irreversibility![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-blue.svg)Reference-quality Python implementation for:Irreversibility-based order parameter (Maximum Hidden Irreversibility, MHI) 
Rotational (antisymmetric) dynamics 
Mixture-of-Experts (MoE) extension 
 
This code is intentionally simple, explicit, readable, and numerically stable. It provides a toy model simulation for demonstration and can be extended for custom dynamics.FeaturesComputes instantaneous and sliding-window Maximum Hidden Irreversibility (MHI) using Gaussian approximations. 
Supports proper forward and backward models for accurate irreversibility measures. 
Calculates rotational magnitude from the antisymmetric component of dynamics. 
Extends MHI with Mixture-of-Experts via routing entropy. 
Includes a stable toy linear dynamics simulation with optional state normalization. 
Numerical stability enhancements: eigenvalue checks, invertible matrix validation, and covariance matching. 
 
What's New in v2Compared to the original version:Stabilized toy model: Normalized antisymmetric matrix, reduced default epsilon for bounded eigenvalues. 
Proper backward model: Uses matrix inverse for mean and adjusted covariance in linear cases. 
Vectorized MoE extension: Applies to full time-series with time-varying routing probabilities. 
Numerical checks: Warnings for unstable dynamics, positive-definite covariance validation. 
Optional trajectory normalization to prevent explosion in norms. 
Improved readability: Type hints, more comments, and fixed covariance mismatch between simulation noise and model. 
Demo enhancements: Time-varying routing probs example, mean trajectory norm output. 
 
InstallationRequires Python 3.8+ and NumPy.pip install numpyOptional for plotting or visualization (not used in core script):pip install matplotlib seabornUsageClone the repository and run the demo:git clone https://github.com/your-repo/mhi-v2.git
cd mhi-v2
python mhi_v2.pyWhat the Demo DoesSimulates a linear system with symmetric (dissipative) and antisymmetric (rotational) components + Gaussian noise. 
Computes the MHI time-series using forward/backward Gaussian log-probabilities. 
Calculates the rotational magnitude of the dynamics matrix. 
Applies MoE extension with example time-varying routing probabilities. 
Outputs key metrics to the console. 
 
You can customize parameters in simulate_toy_dynamics (e.g., T, dim, eta, epsilon, noise, normalize_traj).For your own data:Provide a trajectory array traj and define F_fwd, F_bwd, cov_fwd, cov_bwd. 
Call maximum_hidden_irreversibility and moe_mhi as needed. 
 
Example OutputFinal MHI: 0.1234
Final MoE-extended MHI: 1.4567
Rotational magnitude |A|: 0.0100
Mean trajectory norm: 1.0000(Actual values depend on random seed; with normalize_traj=True, norms stay around 1.)LicenseThis project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). You are free to share and adapt the material, provided you give appropriate credit, distribute under the same license, and indicate changes.