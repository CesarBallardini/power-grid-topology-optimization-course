"""The discretized Gaussian that ToOp samples PST tap steps from (Chapter 5, Exercise 3).

``_build_discrete_pst_step_distribution`` in ``optimizer/dc/genetic_functions/mutation/mutate_nodal_inj.py`` takes
the integers k in [−⌈4σ⌉, ⌈4σ⌉] with weights exp(−k² / (2σ²)) and normalizes them. ToOp computes the weights in
float32; the differences from float64 are far below the digits printed here.
"""

import math

import numpy as np
import pandas as pd

SIGMA = 1.0
SELECTION_PROBABILITY = 0.2  # default pst_mutation_probability in mutate_psts


def step_distribution(sigma: float = SIGMA) -> pd.Series:
    """Probabilities of the integer steps in [−⌈4σ⌉, ⌈4σ⌉], indexed by step."""
    radius = max(1, math.ceil(4 * sigma))
    steps = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (steps / sigma) ** 2)
    return pd.Series(weights / weights.sum(), index=steps, name='P(step)')


def probability_no_move(sigma: float = SIGMA) -> float:
    """P(step = 0): a PST selected for mutation stays on its tap."""
    return float(step_distribution(sigma)[0])


def fraction_changing(sigma: float = SIGMA, selection: float = SELECTION_PROBABILITY) -> float:
    """Fraction of PSTs that change tap in one mutation: selected and drawing a non-zero step."""
    return selection * (1 - probability_no_move(sigma))


def main() -> None:
    """Print the step table for σ = 1 and the two probabilities of Exercise 3."""
    dist = step_distribution()
    print(dist.round(4).to_string())
    print(f'P(step = 0) = {probability_no_move():.4f}')
    print(f'fraction of PSTs that change tap = 0.2 x (1 - {probability_no_move():.4f}) = {fraction_changing():.4f}')
    continuous = 1 / math.sqrt(2 * math.pi)
    print(f'continuous Gaussian density at 0 for comparison: {continuous:.4f}')


if __name__ == '__main__':
    main()
