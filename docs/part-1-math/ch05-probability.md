# Chapter 5. Probability for stochastic search

**Weeks:** 1 · **Semester:** 1 · **Prerequisites:** none

!!! info "Why it matters for ToOp"

    The evolutionary operators sample. The number of substations to mutate is drawn from a Poisson distribution
    (default $\lambda = 2$, `n_subs_mutated_lambda` in `optimizer/interfaces/messages/dc_params.py`) and then
    **clipped to** $[1, m]$ with $m$ = `max_num_splits` (`jnp.clip` in
    `optimizer/dc/genetic_functions/mutation/mutate.py`), so the draw is never zero. PST taps move by an integer
    step drawn from a discretized Gaussian (`optimizer/dc/genetic_functions/mutation/mutate_nodal_inj.py`).
    Parents are drawn uniformly among the occupied cells of the repertoire
    (`optimizer/dc/repertoire/discrete_me_repertoire.py`). Randomness is controlled with JAX PRNG keys that are
    split before use. Results are summarized with maxima, medians and the top-k worst contingencies
    (`get_median_flow_n_1_matrix` and `get_worst_k_contingencies` in `dc_solver/jax/aggregate_results.py`).

## Topics

- Discrete and continuous random variables: uniform, categorical, Bernoulli, Poisson, normal.
- Functions of a random variable: clipping, rounding and discretizing a distribution.
- Expectation, variance; sampling; pseudo-random generators and seeds; independent streams.
- Order statistics (max, median, top-k); comparing stochastic algorithms over repeated runs (preview of
  [Chapter 26](../part-4-optimization/ch26-experimental-methodology.md)).

## Learning outcomes

- Derive the probability mass function and mean of a clipped Poisson variable and confirm them by
  simulation.
- Simulate a random operator reproducibly with explicit seeds and independent streams.
- Summarize repeated stochastic runs with max, median, quantiles and an empirical CDF, and explain what each
  statistic hides.

## Resources

- Courses: MIT OCW 18.05 <https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/>.
- Free texts: Grinstead & Snell, *Introduction to Probability* <https://math.dartmouth.edu/~prob/prob/prob.pdf>;
  *OpenIntro Statistics* <https://www.openintro.org/book/os/>.
- Textbook: Ross, *A First Course in Probability*. **PRINT-DISABLED**.
- Documentation and tools: NumPy random `Generator` (seeding, `poisson`, `normal`, `choice`, `spawn`)
  <https://numpy.org/doc/stable/reference/random/generator.html>.

## Lab

!!! example "Lab: simulating ToOp's mutation counts"

    Tools: Jupyter + NumPy + pandas + matplotlib (uses
    [Chapter 30](../part-5-computing/ch30-scientific-python.md) basics).

    1. With `rng = np.random.default_rng(seed)`, draw 10 000 values $N \sim \text{Poisson}(2)$ and then compute
       $K = \min(\max(N, 1), m)$ for $m = 3$ (the equivalent of `jnp.clip(n, 1, max_num_splits)`). Tabulate
       the frequencies of $N$ and of $K$ with `pd.Series.value_counts` and plot both as bar charts.
    2. Overlay the exact probabilities of $K$ (Exercise 1) on the bar chart. Note that the bar at $K = 1$
       absorbs the $N = 0$ mass, $e^{-2} \approx 13.5\%$ of all draws.
    3. Repeat for $m = 1, \dots, 6$ and plot the sample mean of $K$ against $m$, with a horizontal line at
       $\lambda + e^{-\lambda}$. Explain which clip (lower or upper) dominates for small and for large $m$.
    4. Run the simulation twice with the same seed and once with a different seed; then create independent
       child generators with `rng.spawn(4)`. Compare with how `mutate.py` splits a JAX key into
       `substation_key` and `disconnection_key` before sampling.

## Exercises

1. Let $N \sim \text{Poisson}(\lambda)$ and $K = \min(\max(N, 1), m)$. Show that $P(K = 1) = e^{-\lambda}(1 + \lambda)$,
   $P(K = k) = e^{-\lambda}\lambda^k / k!$ for $1 < k < m$ and $P(K = m) = P(N \ge m)$. For $\lambda = 2$, $m = 3$
   compute the three probabilities and $E[K]$. *(Pen and paper)* *Answer:* $0.406$, $0.271$, $0.323$;
   $E[K] \approx 1.917$, below the unclipped mean 2.
2. Show that $E[\max(N, 1)] = \lambda + e^{-\lambda}$ and explain why clipping at 1 alone raises the mean while
   clipping at $m$ lowers it. *(Pen and paper)*
3. ToOp samples PST tap steps from integers $k \in [-\lceil 4\sigma \rceil, \lceil 4\sigma \rceil]$ with weights
   $\propto e^{-k^2 / (2\sigma^2)}$ (`_build_discrete_pst_step_distribution` in
   `optimizer/dc/genetic_functions/mutation/mutate_nodal_inj.py`). For $\sigma = 1$ tabulate the probabilities,
   and compute the probability that a PST selected for mutation does not move. With a selection probability
   of 0.2 per PST (the default), what fraction of PSTs actually change tap in one mutation, ignoring reset
   and clipping? *(Jupyter: NumPy + pandas)* *Answer:* $P(k = 0) \approx 0.399$; about 12%.
4. For $n$ independent Uniform(0, 1) values, show that $E[\max] = n/(n+1)$ and check it by simulation for
   $n = 1, 10, 100$. Relate it to "the worst of many contingencies is almost always high". *(Pen and paper,
   then Jupyter: NumPy)*
5. A stochastic optimizer returns a best cost $C = 100 + 10\,Z$ with $Z \sim \mathcal{N}(0, 1)$ per run. Simulate
   30 runs for two variants whose means differ by 3. Plot both empirical CDFs and report mean, median and
   the 10% and 90% quantiles. Would one run per variant have told them apart? *(Jupyter: NumPy + pandas +
   matplotlib)*
