"""Two-component coupler states of a ring of busbars, by brute force and in closed form (Chapter 13, Exercise 3).

A ring of n busbars joined by n couplers falls into exactly two connected groups when exactly two couplers are
open, so the count is n choose 2. The brute force of ``coupler_splits.two_node_configurations`` visits all 2**n
coupler states, which is what bounds ToOp's enumeration to 20 couplers.
"""

import time
from math import comb

from coupler_splits import two_node_configurations


def ring_couplers(n: int) -> list[tuple[int, int]]:
    """The n couplers of a ring of n busbars: busbar i joined to busbar (i + 1) mod n."""
    return [(i, (i + 1) % n) for i in range(n)]


def count_two_component_states(n: int) -> int:
    """Brute-force count of the coupler states of a ring of n busbars that leave exactly two components."""
    return len(two_node_configurations(ring_couplers(n), n))


def closed_form(n: int) -> int:
    """The binomial coefficient n choose 2: the two-component states open exactly two of the n couplers of the ring."""
    return comb(n, 2)


def timed_count(n: int) -> tuple[int, float]:
    """The brute-force count for a ring of n busbars and the wall-clock seconds it took."""
    start = time.perf_counter()
    count = count_two_component_states(n)
    return count, time.perf_counter() - start


def main(n_max: int = 16) -> None:
    """Print count, closed form and time for n = 3 .. n_max, then extrapolate the time to ToOp's limit of 20."""
    print(' n  states  count  C(n,2)  seconds')
    last = 0.0
    for n in range(3, n_max + 1):
        count, seconds = timed_count(n)
        print(f'{n:2d}  {2**n:6d}  {count:5d}  {closed_form(n):6d}  {seconds:7.3f}')
        last = seconds
    print(f'doubling per coupler: about {last * 2 ** (20 - n_max):.0f} s for n = 20 on this machine')


if __name__ == '__main__':
    main()
