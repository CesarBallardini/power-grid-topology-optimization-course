"""ToOp's loadflow-based limit rule for border lines and DSO transformers (Chapter 22, Exercise 3).

The new current limit is ``clip(f * I_lf, min(I_old, I_lf + m * I_old), I_old)``: a factor times the base-case
current, never above the old limit and never below the base-case current plus a small share of the old limit.
"""

import numpy as np

F_N0, F_N1, M = 1.2, 1.4, 0.05
I_OLD = 1000.0  # A


def new_limit(i_lf: np.ndarray | float, i_old: float = I_OLD, f: float = F_N0, m: float = M) -> np.ndarray | float:
    """The adjusted limit for a base-case current ``i_lf``."""
    lower = np.minimum(i_old, i_lf + m * i_old)
    return np.clip(f * i_lf, lower, i_old)


def breakpoints(i_old: float = I_OLD, f: float = F_N0, m: float = M) -> tuple[float, float]:
    """Base-case currents where the lower clip stops acting, m I_old / (f - 1), and the upper one starts, I_old / f."""
    return m * i_old / (f - 1.0), i_old / f


def main() -> None:
    """The three branches of the rule for a 1 000 A limit, with the N-0 and the N-1 factor."""
    for f in (F_N0, F_N1):
        low, high = breakpoints(f=f)
        print(
            f'f = {f}: I_lf + {M * I_OLD:.0f} A below {low:.1f} A, {f} I_lf between, {I_OLD:.0f} A above {high:.1f} A'
        )
    print('unloaded border line (I_lf = 0):', new_limit(0.0), 'A')
    for i in (100.0, 250.0, 500.0, 833.3, 900.0, 1200.0):
        print(f'I_lf = {i:6.1f} A -> I_new = {new_limit(i):6.1f} A (N-0), {new_limit(i, f=F_N1):6.1f} A (N-1)')


if __name__ == '__main__':
    main()
