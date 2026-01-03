"""
mhi-demo.py

Minimal reproducible demonstration of Maximum Hidden Irreversibility (MHI)
in a toy high-order Markov system.

This code shows hidden rotational dynamics invisible to standard metrics.
"""

import numpy as np
import itertools

# Toy model parameters
N = 4       # number of states
n = 2       # Markov chain order
k_max = 2   # HO-TPF levels
epsilon = 0.05
states = list(range(N))

# Generate all tuples of given length
def all_tuples(order):
    return list(itertools.product(states, repeat=order))

# Create irreversible transition table T_B
def create_T_B(N, n, k_max, epsilon):
    T = {}
    for order in range(n+1, n+k_max+2):
        for t in all_tuples(order):
            T[t] = 1/N
        # Simple cyclic perturbation to induce hidden irreversibility
        cycle = tuple(range(order))
        T[cycle] += epsilon
        T[tuple(reversed(cycle))] -= epsilon
    return T

# Stationary distribution
def create_Pi(N, n):
    Pi = {t[:n+1]: 1/N**(n+1) for t in all_tuples(n+2)}
    return Pi

# Compute HO-TPF and MHI
def compute_HO_TPF(T_B, Pi, n, k_max):
    MHI_dict = {}
    for k in range(1, k_max+1):
        order = n+k+1
        J = {}
        for t in itertools.product(states, repeat=order):
            prefix = t[:order-1]
            suffix = t[1:]
            pi_prefix = Pi.get(prefix, 0)
            pi_suffix = Pi.get(suffix, 0)
            T_forward = T_B.get(t, 0)
            T_backward = T_B.get(tuple(list(suffix)+[prefix[0]]), 0)
            J[t] = pi_prefix*T_forward - pi_suffix*T_backward
        # Recursive decomposition
        F = {}
        for t in itertools.product(states, repeat=order):
            sum_H = sum(J.get(t[:order-1]+(t[i],), 0) for i in range(order-1))/(order-1)
            F[t] = J[t] - sum_H
        MHI_dict[k] = sum(abs(v) for v in F.values())
    return MHI_dict

# Run demo
if name == "__main__":
    T_B = create_T_B(N, n, k_max, epsilon)
    Pi = create_Pi(N, n)
    MHI_result = compute_HO_TPF(T_B, Pi, n, k_max)
    print("Demo MHI^(n+k) for toy system:", MHI_result)
