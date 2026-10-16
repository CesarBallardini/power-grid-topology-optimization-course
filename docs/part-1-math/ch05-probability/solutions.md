# Chapter 5 – Solutions

Solutions of the Lab and the Exercises of [Chapter 5](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`clipped_poisson.py`](src/clipped_poisson.py) (lab steps 1–3, Exercises 1
and 2), [`rng_streams.py`](src/rng_streams.py) (lab step 4), [`pst_steps.py`](src/pst_steps.py) (Exercise 3),
[`uniform_max.py`](src/uniform_max.py) (Exercise 4) and [`two_variants.py`](src/two_variants.py) (Exercise 5). Each
module's `main()` prints every number quoted below; the simulations use `np.random.default_rng(0)` unless another
seed is named, so they reproduce exactly. The worked examples run in [examples.ipynb](src/examples.ipynb).

## Lab

1. **Draws of $N$ and $K$.** `simulate()` draws 10 000 values of $N \sim \text{Poisson}(2)$ with
   `rng.poisson(2, size=10_000)` and clips them with `np.clip(n, 1, 3)`. The relative frequencies
   (`frequencies`, built on `pd.Series.value_counts(normalize=True)`):

    | $N$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | frequency | 0.1348 | 0.2698 | 0.2722 | 0.1823 | 0.0897 | 0.0340 | 0.0123 | 0.0037 | 0.0011 | 0.0001 |

    | $K$ | 1 | 2 | 3 |
    |---|---|---|---|
    | frequency | 0.4046 | 0.2722 | 0.3232 |

    The sample mean of $N$ is 1.9993 and that of $K$ is 1.9186. `plot_frequencies` draws the two bar charts
    side by side: the Poisson bars with their long right tail, and three bars for $K$ with the tail folded onto
    $K = 3$ and the $N = 0$ bar folded onto $K = 1$ ($0.1348 + 0.2698 = 0.4046$).
2. **Exact probabilities.** [Exercise 1](#exercise-1) gives $P(K = 1) = 0.4060$, $P(K = 2) = 0.2707$ and
   $P(K = 3) = 0.3233$ (`clipped_pmf(2.0, 3)`), drawn as black dots on the bar chart; each simulated frequency is
   within 0.002 of its exact value, well inside the sampling error of about $\sqrt{p(1-p)/10\,000} \approx 0.005$.
   The bar at $K = 1$ absorbs the $N = 0$ mass, $e^{-2} = 0.1353$ of all draws (0.1348 in this sample): in ToOp
   terms, one draw in seven would have mutated no substation at all, and the clip turns every one of them into a
   single-substation mutation.
3. **Mean of $K$ against $m$.** With the same 10 000 draws of $N$ (`sample_mean_vs_m`):

    | $m$ | 1 | 2 | 3 | 4 | 5 | 6 |
    |---|---|---|---|---|---|---|
    | sample mean of $K$ | 1.0000 | 1.5954 | 1.9186 | 2.0595 | 2.1107 | 2.1279 |
    | exact $E[K]$ | 1.0000 | 1.5940 | 1.9173 | 2.0602 | 2.1128 | 2.1294 |

    `plot_mean_vs_m` shows the curve rising from 1 towards the horizontal line $\lambda + e^{-\lambda} = 2.1353$,
    the mean with the lower clip alone ([Exercise 2](#exercise-2)), and crossing $\lambda = 2$ between $m = 3$ and
    $m = 4$. For small $m$ the upper clip dominates: it cuts the whole right tail of the Poisson down to $m$, and
    the mean is below 2. For large $m$ the upper clip bites only on the rare draws beyond $m$ and the lower clip
    dominates: it adds $P(N = 0) = 0.135$ to the mean, which therefore ends up above 2. With ToOp's default
    $\lambda = 2$, a `max_num_splits` of 3 or 4 sits right at the crossing, so the average mutation touches about
    two substations, as the parameter name suggests, with the two clipping effects nearly cancelling.
4. **Seeds and streams.** `numpy_draws(42)` twice gives the same eight Poisson draws, `[4 3 3 2 1 5 2 0]`; seed 43
   gives `[1 0 2 2 3 1 0 3]`. `np.random.default_rng(42).spawn(4)` gives four children whose streams,
   `[5 2 2 2 0 1 2 3]`, `[1 1 3 1 2 1 2 3]`, `[0 1 2 3 3 3 2 0]` and `[3 1 2 1 3 2 2 0]`, differ from each other
   and from the parent's own stream, and are reproduced exactly on a second call: independent streams from a
   single seed, with no shared state between them. `mutate.py` does the same thing without a mutable generator:
   `random_key, substation_key, disconnection_key = jax.random.split(random_key, 3)` turns one key into three
   fresh ones; the Poisson count is drawn from `substation_key` (`jax_mutation_count(0)` runs that line with
   `PRNGKey(0)` and gets $N = 0$, clipped to $K = 1$, the same twice), the substation loop also starts from
   `substation_key`, the disconnection mutation gets `disconnection_key`, and the function returns the new
   `random_key` for the caller to split again. A key is never reused for two different draws, which is the JAX
   equivalent of `spawn`: the two mutation operators cannot accidentally consume each other's random numbers, and
   adding a draw to one of them does not change the other's results. In NumPy the same discipline means one
   spawned child per operator rather than one shared `rng`.

## Exercise 1

$K = \min(\max(N, 1), m)$ takes the values $1, \dots, m$. $K = 1$ exactly when $N \in \{0, 1\}$, so

$$
P(K = 1) = P(N = 0) + P(N = 1) = e^{-\lambda} + e^{-\lambda} \lambda = e^{-\lambda}(1 + \lambda).
$$

For $1 < k < m$ neither clip acts and $K = k$ exactly when $N = k$: $P(K = k) = e^{-\lambda} \lambda^k / k!$.
$K = m$ exactly when $N \ge m$: $P(K = m) = P(N \ge m) = 1 - \sum_{j < m} e^{-\lambda} \lambda^j / j!$.

For $\lambda = 2$, $m = 3$ (`clipped_pmf(2.0, 3)` and `clipped_mean`): $P(K = 1) = 3 e^{-2} = 0.4060$,
$P(K = 2) = 2 e^{-2} = 0.2707$, $P(K = 3) = 1 - 5 e^{-2} = 0.3233$, which sum to 1, and

$$
E[K] = 1 \times 0.4060 + 2 \times 0.2707 + 3 \times 0.3233 = 1.9173 ,
$$

below the unclipped mean 2, as the chapter states. The simulation of lab step 1 gave 0.4046, 0.2722, 0.3232 and
a sample mean of 1.9186.

## Exercise 2

Split the expectation at $N = 0$, the only value the lower clip changes:

$$
E[\max(N, 1)] = 1 \cdot P(N = 0) + \sum_{n \ge 1} n\, P(N = n) = e^{-\lambda} + \sum_{n \ge 0} n\, P(N = n)
= e^{-\lambda} + \lambda ,
$$

because the $n = 0$ term of the full sum is zero and the full sum is $E[N] = \lambda$ (`mean_of_max_with_one(2.0)`
returns 2.1353; `clipped_mean(2.0, 60)`, where the upper clip never acts in practice, returns the same value to
$10^{-12}$).

The lower clip replaces the value 0 by 1 on an event of probability $e^{-\lambda}$ and changes nothing else, so
it adds exactly $e^{-\lambda}$ to the mean: clipping below can only raise a random variable. The upper clip
replaces every $N > m$ by the smaller value $m$ and changes nothing else, so it lowers the mean by
$E[(N - m)^+] = \sum_{n > m} (n - m) P(N = n) > 0$: clipping above can only lower it. Both at once give
$E[K] = \lambda + e^{-\lambda} - E[(N - m)^+]$; for $\lambda = 2$, $m = 3$ the tail term is $2.1353 - 1.9173 =
0.218$, larger than the $0.135$ added below, which is why $E[K] < \lambda$ there, while for $m \ge 4$ the tail
term is smaller than $0.135$ and $E[K] > \lambda$ (lab step 3).

## Exercise 3

For $\sigma = 1$ the support is $k \in [-4, 4]$ ($\lceil 4\sigma \rceil = 4$) and the weights $e^{-k^2/2}$
normalized give (`step_distribution`):

| $k$ | $\pm 4$ | $\pm 3$ | $\pm 2$ | $\pm 1$ | 0 |
|---|---|---|---|---|---|
| $P(k)$ | 0.0001 | 0.0044 | 0.0540 | 0.2420 | 0.3989 |

$P(k = 0) = 0.3989$: two PSTs in five that are selected for mutation keep their tap. It equals the continuous
Gaussian density at 0, $1/\sqrt{2\pi} = 0.3989$, to six decimals, because for $\sigma \ge 1$ the lattice sum
$\sum_k e^{-k^2/(2\sigma^2)}$ is $\sqrt{2\pi}\,\sigma$ to high accuracy and the truncation at $4\sigma$ drops less
than $10^{-4}$ of the mass. With the default selection probability 0.2 per PST, the fraction of PSTs that change
tap in one mutation is

$$
0.2 \times (1 - 0.3989) = 0.1202 ,
$$

about 12 %, ignoring the reset step (which puts a PST back to its starting tap with probability 0.1) and the
clipping to $[0, n_{\text{taps}} - 1]$ (which can turn a non-zero step into no move at the end of the range).
ToOp computes the weights in `float32`; its probabilities differ from the `float64` ones above by less than
$2 \times 10^{-8}$.

## Exercise 4

Let $U_1, \dots, U_n$ be independent Uniform(0, 1) and $M = \max_i U_i$. For $x \in [0, 1]$,
$P(M \le x) = P(\text{all } U_i \le x) = x^n$, so $M$ has density $n x^{n-1}$ and

$$
E[M] = \int_0^1 x \cdot n x^{n-1}\, dx = \frac{n}{n + 1},
\qquad\text{or}\qquad
E[M] = \int_0^1 P(M > x)\, dx = \int_0^1 (1 - x^n)\, dx = 1 - \frac{1}{n + 1}.
$$

Simulation with 100 000 repetitions (`comparison`):

| $n$ | exact $n/(n+1)$ | simulated | standard error | $P(M > 0.9)$ |
|---|---|---|---|---|
| 1 | 0.50000 | 0.49957 | 0.00091 | 0.099 |
| 10 | 0.90909 | 0.90944 | 0.00026 | 0.653 |
| 100 | 0.99010 | 0.99004 | 0.00003 | 0.99995 |

Each simulated mean is within one standard error of the exact value. The last column is the point of the
exercise: with 100 draws the maximum exceeds 0.9 essentially always ($1 - 0.9^{100}$), even though each draw
exceeds it with probability 0.1 only. An N-1 analysis evaluates hundreds of contingencies and reports the worst
loading; if their loadings scatter at all, the worst one sits near the top of its range in every topology, so the
maximum is a stable but pessimistic summary that changes little between topologies that differ in the bulk. That
is why ToOp also keeps the median (`get_median_flow_n_1_matrix`) and the top-$k$ list (`get_worst_k_contingencies`)
in `dc_solver/jax/aggregate_results.py`: the max tells you how bad the worst case is, the median how the typical
contingency fares, and the top-$k$ which few contingencies drive the max.

## Exercise 5

`simulate_runs(0)` draws 30 costs $C = 100 + 10 Z$ for variant A and $C = 103 + 10 Z$ for variant B
(`summary`):

| Variant | mean | median | 10 % quantile | 90 % quantile | min | max |
|---|---|---|---|---|---|---|
| A | 98.79 | 98.70 | 90.46 | 109.57 | 76.75 | 113.66 |
| B | 105.76 | 106.56 | 91.13 | 117.95 | 90.12 | 122.60 |

`plot_ecdfs` draws the two empirical CDFs: B's curve lies to the right of A's, but the two overlap over most of
their range, from about 90 to 114. Mean, median and the two quantiles all point the same way for this seed, and
the sample gap in means, 7.0, is more than twice the true gap of 3: with a run-to-run standard deviation of 10,
the standard error of a difference of two means of 30 runs is $10\sqrt{2/30} = 2.58$, so a sample gap anywhere
between $-2$ and $8$ is unremarkable.

One run per variant would not have told them apart. The difference of two single runs is
$\mathcal{N}(-3, (10\sqrt{2})^2)$, so A looks better than B with probability $\Phi(3 / (10\sqrt{2})) = \Phi(0.212)
= 0.584$ (`probability_single_run_orders_correctly`; 0.586 in a simulation of 100 000 pairs): barely better
than a coin toss, and 42 % of the time a single comparison names the wrong winner. Thirty runs each are better but
not decisive: Welch's test on seed 0 gives $t = -3.04$, $p = 0.004$, a lucky draw, while seeds 1 to 5 give
$p = 0.145, 0.022, 0.578, 0.114, 0.225$ (`welch_test`). Over 200 seeds the average gap is 3.0, so the estimate is
unbiased, but at 30 runs per variant the 3-point difference is 1.16 standard errors and most seeds do not reach
the 5 % level; `runs_needed()` gives 174 runs per variant for 80 % power. This is the lesson
[Chapter 26](../../part-4-optimization/ch26-experimental-methodology/index.md) builds on: report distributions over
many seeds, not one number per variant, and decide the number of runs from the effect you want to detect before
running the experiment.
