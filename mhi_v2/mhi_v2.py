"""
MHI v2 — Maximum Hidden Irreversibility
=====================================

Reference implementation for:
- Irreversibility-based order parameter
- Rotational (antisymmetric) dynamics
- Mixture-of-Experts (MoE) extension

This code is intentionally simple, explicit, and readable.
Improved version: 
- Fixed covariance mismatch between simulation noise and model cov.
- Added proper backward model for linear dynamics (inverse mean and cov).
- Stabilized toy model: normalized antisymmetric matrix, reduced epsilon, optional normalization.
- Vectorized MoE extension for full time-series.
- Added numerical stability checks (e.g., eigenvalue warnings, invertible checks).
- Improved readability with type hints and more comments.
"""

import numpy as np
import warnings


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def gaussian_log_prob(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Log-probability of multivariate Gaussian.
    Handles numerical stability with checks.
    """
    d = x.shape[0]
    try:
        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        if det_cov <= 0:
            raise ValueError("Covariance matrix is not positive definite.")
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular or not invertible.")

    diff = x - mean
    quad_form = diff.T @ cov_inv @ diff

    return -0.5 * (
        d * np.log(2 * np.pi)
        + np.log(det_cov)
        + quad_form
    )


# --------------------------------------------------
# Irreversibility (MHI)
# --------------------------------------------------

def instantaneous_mhi(
    h_t: np.ndarray,
    h_tp1: np.ndarray,
    F_fwd: callable,
    F_bwd: callable,
    cov_fwd: np.ndarray,
    cov_bwd: np.ndarray
) -> float:
    """
    Instantaneous Hidden Irreversibility:

        I_t = log p(h_{t+1} | h_t) - log p(h_t | h_{t+1})

    Using proper forward and backward Gaussian models.
    """
    mean_fwd = F_fwd(h_t)
    mean_bwd = F_bwd(h_tp1)

    log_fwd = gaussian_log_prob(h_tp1, mean_fwd, cov_fwd)
    log_bwd = gaussian_log_prob(h_t, mean_bwd, cov_bwd)

    return log_fwd - log_bwd


def maximum_hidden_irreversibility(
    h_traj: np.ndarray,
    F_fwd: callable,
    F_bwd: callable,
    cov_fwd: np.ndarray,
    cov_bwd: np.ndarray,
    window: int = 100
) -> np.ndarray:
    """
    Sliding-window Maximum Hidden Irreversibility (MHI).
    Vectorized for efficiency where possible.
    """
    T = len(h_traj)
    I_vals = np.zeros(T - 1)

    for t in range(T - 1):
        I_vals[t] = instantaneous_mhi(
            h_traj[t],
            h_traj[t + 1],
            F_fwd,
            F_bwd,
            cov_fwd,
            cov_bwd
        )

    mhi = np.zeros_like(I_vals)

    for t in range(len(I_vals)):
        start = max(0, t - window + 1)
        mhi[t] = np.max(I_vals[start:t + 1])

    return mhi


# --------------------------------------------------
# Rotational dynamics
# --------------------------------------------------

def antisymmetric_part(J: np.ndarray) -> np.ndarray:
    """
    A = (J - J^T) / 2
    """
    return 0.5 * (J - J.T)


def rotational_magnitude(J: np.ndarray) -> float:
    """
    Frobenius norm of antisymmetric component.
    """
    A = antisymmetric_part(J)
    return np.linalg.norm(A, ord="fro")


# --------------------------------------------------
# Mixture-of-Experts (MoE)
# --------------------------------------------------

def routing_entropy(p: np.ndarray) -> float:
    """
    Entropy of routing distribution.
    """
    p = np.clip(p, 1e-8, 1.0)
    return -np.sum(p * np.log(p))


def moe_mhi(base_mhi: np.ndarray, routing_probs: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    MoE-extended MHI:
        MHI_MoE = MHI + alpha * routing entropy
    Assumes routing_probs is array of shape (T-1, num_experts) for time-varying.
    If scalar or 1D, broadcasts to all times.
    """
    if routing_probs.ndim == 1:
        routing_probs = np.tile(routing_probs, (len(base_mhi), 1))
    
    entropies = np.array([routing_entropy(p) for p in routing_probs])
    return base_mhi + alpha * entropies


# --------------------------------------------------
# Toy model simulation (for testing)
# --------------------------------------------------

def simulate_toy_dynamics(
    T: int = 2000,
    dim: int = 16,
    eta: float = 0.01,
    epsilon: float = 0.01,  # Reduced for stability
    noise: float = 0.1,    # Increased to match reasonable cov
    normalize_traj: bool = False  # Option to normalize states for bounded norms
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear system with symmetric + antisymmetric components.
    Improved: Normalized A for spectral radius <1, optional trajectory normalization.
    """
    # Symmetric part: dissipative
    S = np.eye(dim)

    # Antisymmetric part: rotational
    A = np.random.randn(dim, dim)
    A = antisymmetric_part(A)
    if np.linalg.norm(A, ord=2) > 0:
        A /= np.linalg.norm(A, ord=2)  # Normalize to unit spectral norm

    # Dynamics matrix
    M = np.eye(dim) - eta * S + epsilon * A

    # Check stability
    eigvals = np.linalg.eigvals(M)
    max_eig = np.max(np.abs(eigvals))
    if max_eig > 1:
        warnings.warn(f"Dynamics unstable: max |eig| = {max_eig:.2f} > 1. Trajectories may explode.")

    # Initial state
    h = np.random.randn(dim)
    traj = [h.copy()]

    for _ in range(T):
        h = M @ h + noise * np.random.randn(dim)
        if normalize_traj:
            norm = np.linalg.norm(h)
            if norm > 0:
                h /= norm  # Keep on unit sphere for bounded behavior
        traj.append(h.copy())

    return np.array(traj), M


# --------------------------------------------------
# Main (demo run)
# --------------------------------------------------

if __name__ == "__main__":
    traj, M = simulate_toy_dynamics(normalize_traj=True)  # Enable normalization for stability

    noise = 0.1  # From simulation
    dim = traj.shape[1]
    cov_fwd = noise**2 * np.eye(dim)  # Correct variance match

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        raise ValueError("Dynamics matrix M is not invertible.")

    cov_bwd = M_inv @ cov_fwd @ M_inv.T

    F_fwd = lambda h: M @ h
    F_bwd = lambda h: M_inv @ h

    mhi = maximum_hidden_irreversibility(
        traj,
        F_fwd,
        F_bwd,
        cov_fwd,
        cov_bwd,
        window=100
    )

    rot = rotational_magnitude(M)

    # Example time-varying routing probs (for demo: softening uniform over time)
    num_experts = 4
    routing_probs = np.ones((len(mhi), num_experts)) / num_experts
    for t in range(len(mhi)):
        routing_probs[t] += 0.1 * np.sin(t / 100) * np.random.randn(num_experts)
        routing_probs[t] /= routing_probs[t].sum()  # Normalize

    mhi_moe = moe_mhi(mhi, routing_probs)

    print("Final MHI:", mhi[-1])
    print("Final MoE-extended MHI:", mhi_moe[-1])
    print("Rotational magnitude |A|:", rot)
    print("Mean trajectory norm:", np.mean([np.linalg.norm(h) for h in traj]))