# Chapter 24. Heuristics and evolutionary algorithms

**Weeks:** 2 · **Semester:** 4 · **Prerequisites:** [Chapter 5](../part-1-math/ch05-probability.md), [Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md), [Chapter 23](ch23-optimization-modeling.md)

!!! info "Why it matters for ToOp"

    The DC optimizer is a genetic algorithm over a discrete genotype: a topology is a set of station actions,
    disconnected branches and PST setpoints (`optimizer/dc/genetic_functions/genotype.py`). Unused disconnection
    slots are padded with a sentinel integer, and duplicate genotypes are removed (`deduplicate_genotypes`).
    Crossover samples unique actions from two parents (`optimizer/dc/genetic_functions/crossover.py`). Mutations
    (`optimizer/dc/genetic_functions/mutation/`, defaults in `optimizer/interfaces/messages/dc_params.py`) are
    discrete throughout:

    - **Substations.** The number of substation mutation steps is drawn from a Poisson distribution with
      $\lambda = 2$ (`n_subs_mutated_lambda`) and clipped to the range from 1 to `max_num_splits`; each step adds,
      changes or removes a split, or does nothing with the remaining probability (`mutate.py`,
      `mutate_substations.py`). Disconnections are mutated afterwards (`mutate_disconnections.py`).
    - **PST taps** (only with `enable_nodal_inj_optim=True`). Each PST is selected with probability
      `pst_mutation_probability` (default 0.2) and moved by an integer step drawn from a discretized Gaussian on
      $\{-\lceil 4\sigma \rceil, \dots, \lceil 4\sigma \rceil\}$ with $\sigma$ = `pst_mutation_sigma` (default 1), then
      reset to its starting tap with probability `pst_reset_probability` (default 0.0) and clipped to the tap
      range (`mutate_nodal_inj.py`). A validator requires all three to be zero, or both $\sigma$ and the mutation
      probability to be positive.

    A brute-force optimizer enumerates every combination of up to `max_num_splits` split substations and up to
    `max_num_disconnections` disconnections lazily with `itertools` (`optimizer/dc_bruteforce/generator.py`). On a
    small grid it gives the exact optimum against which a heuristic can be judged, which is exactly what the lab
    of this chapter does. Note that line switching is off by default: `max_num_disconnections` is `0` in
    `optimizer/interfaces/messages/dc_params.py`.

## Topics

- Local search, hill climbing, simulated annealing, tabu search.
- Genetic algorithms: representation, variation operators, selection, elitism, diversity.
- Exploration vs exploitation; restarts; the No Free Lunch theorem.
- Designing a genotype and operators for a constrained discrete problem: fixed-length padded sets, sorted
  indices, repair vs penalty for infeasible candidates (islanding), deduplication.
- Exhaustive enumeration as a baseline: counting the search space and knowing when it is still affordable.
- Randomness in heuristics: seeds, variance between runs (treated rigorously in
  [Chapter 26](ch26-experimental-methodology.md)).

## Learning outcomes

- Implement hill climbing, simulated annealing and a genetic algorithm for a subset-selection problem in NumPy,
  with explicit, seeded random number generators.
- Design a genotype and mutation and crossover operators for "switch off at most $k$ lines" that never produce
  duplicate indices, and explain how infeasible (islanding) candidates are handled.
- Compare a heuristic with exact enumeration on a small grid by solution quality, number of fitness
  evaluations and wall-clock time, and explain when enumeration stops being feasible.

## Resources

- Free texts:
  - Luke, *Essentials of Metaheuristics*, 2nd ed.: Ch. 2 single-state methods, Ch. 3 population methods,
    Ch. 4 representation. FREE <https://people.cs.gmu.edu/~sean/book/metaheuristics/>.
  - Brownlee, *Clever Algorithms: Nature-Inspired Programming Recipes*: FREE <https://cleveralgorithms.com/>.
  - Libraries: DEAP <https://deap.readthedocs.io/>, pymoo <https://pymoo.org/>.
- Textbooks:
  - Eiben & Smith, *Introduction to Evolutionary Computing* (5 syllabi, the most cited). 1st ed. 2003
    **BORROW** [`introductiontoev0000eibe`](https://archive.org/details/introductiontoev0000eibe) (2nd ed. 2015
    current).
  - Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning* (1989; 3 syllabi). **BORROW**
    [`geneticalgorithm0000gold`](https://archive.org/details/geneticalgorithm0000gold).
  - Mitchell, *An Introduction to Genetic Algorithms* (1996). **BORROW**
    [`introductiontoge00mitc`](https://archive.org/details/introductiontoge00mitc).
  - Michalewicz & Fogel, *How to Solve It: Modern Heuristics* (2000). **BORROW**
    [`howtosolveitmode0000mich`](https://archive.org/details/howtosolveitmode0000mich).
- Papers: Kirkpatrick, Gelatt & Vecchi, "Optimization by Simulated Annealing", *Science* 220, 1983;
  Wolpert & Macready, "No free lunch theorems for optimization", *IEEE TEC* 1(1), 1997.

## Worked example

A genotype for "switch off at most $k$ of $n$ lines" can be a sorted integer vector of length $k$ padded with the
sentinel $n$ (ToOp pads with the largest integer instead). Mutation changes one slot, and the candidate is
normalized so that equal sets have equal genotypes:

```python
import numpy as np

def normalize(g, n):
    """Remove duplicates, sort, pad with sentinel n."""
    u = np.unique(g[g < n])
    return np.concatenate([u, np.full(len(g) - len(u), n)])

def mutate(g, n, rng, p_remove=0.2):
    g = g.copy()
    slot = rng.integers(len(g))
    g[slot] = n if rng.random() < p_remove else rng.integers(n)
    return normalize(g, n)

rng = np.random.default_rng(seed=1)
n, k = 41, 3
g = normalize(np.array([7, 7, 41]), n)   # -> [7, 41, 41]
print(g, mutate(g, n, rng))
```

The fitness function receives the set `g[g < n]`, rebuilds the network without those lines and returns the
negated N-1 overload energy, or $-\infty$ when the switched network is disconnected.

## Lab

!!! example "Lab: a genetic algorithm for line switching, checked against brute force"

    **Tools:** Jupyter with NumPy, pandas and matplotlib; your DC power flow
    ([Chapter 16](../part-3-power-systems/ch16-dc-power-flow.md)) and PTDF/LODF code
    ([Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md)).

    1. **Fitness evaluator.** Load IEEE 30 (MATPOWER `case30` includes branch ratings). Scale all loads and
       generation by a common factor until the base topology has N-1 overloads. For a set $D$ of switched-off
       lines, rebuild the PTDF and LODF of the reduced network with your Chapter 17 code and compute the N-1
       overload energy

        $$
        E(D) = \sum_{c}\sum_{\ell} \max\bigl(0,\ |f_\ell^{(c)}(D)| - \bar f_\ell\bigr),
        $$

        summing over non-bridge contingencies $c$ and monitored lines $\ell$. Return $-\infty$ if removing $D$
        islands the grid. Validate a few values with a full DC power flow.
    2. **Brute force.** Enumerate all sets with $|D| \le 2$ and then $|D| \le 3$ using `itertools.combinations`,
       as ToOp's `optimizer/dc_bruteforce/generator.py` does. Record the exact optimum, the number of evaluations
       and the time. Predict the time for $|D| \le 4$ from the count.
    3. **Genetic algorithm.** Implement a GA with the genotype of the worked example, tournament selection,
       elitism, the mutation above and a crossover that samples $k$ unique indices from the union of the two
       parents. Use a fixed budget of fitness evaluations (for example 10% of the $|D| \le 3$ enumeration).
    4. **Compare.** Run the GA with 10 different seeds. In a pandas table report the best value found, the
       evaluations needed to first reach the brute-force optimum (or "not reached") and the time. Plot best
       fitness against evaluations for all seeds in one matplotlib figure. Explain the spread between seeds; you
       will analyze it statistically in Chapter 26.

## Exercises

1. For hill climbing with the "change one slot" neighborhood on the $k$-subset problem, how many neighbors does
   a solution have? Give a two-line toy example where hill climbing gets stuck in a local optimum that simulated
   annealing can leave. *(pen and paper)*
2. Implement simulated annealing for the lab problem with a geometric cooling schedule and compare the final
   value after the same evaluation budget as the GA, for 10 seeds. *(Jupyter: NumPy + matplotlib)*
3. Show that switching benefits are not additive: find two lines $a, b$ in IEEE 30 with
   $E(\{a,b\}) - E(\emptyset) \ne \bigl(E(\{a\}) - E(\emptyset)\bigr) + \bigl(E(\{b\}) - E(\emptyset)\bigr)$ and
   explain with LODFs why. Why does this make greedy selection unreliable? *(Jupyter: NumPy)*
4. Compare "penalty" ($-\infty$ fitness) with "repair" (drop lines from $D$ until the grid is connected) for
   islanding candidates: which one wastes fewer evaluations in your runs? *(Jupyter: NumPy + pandas)*
5. The No Free Lunch theorem says all algorithms perform equally on average over all problems. Explain in one
   paragraph why this does not make the choice of operators for line switching irrelevant. *(pen and paper)*
