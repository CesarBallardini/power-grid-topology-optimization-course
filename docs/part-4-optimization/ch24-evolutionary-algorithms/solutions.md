# Chapter 24 – Solutions

Solutions of the Lab and the Exercises of [Chapter 24](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`genome_mutation.py`](src/genome_mutation.py) (the genotype of the
worked example), [`line_switching.py`](src/line_switching.py) (IEEE 30, the fitness and the brute force),
[`switching_ga.py`](src/switching_ga.py) (the genetic algorithm, the seed table, penalty against repair) and
[`annealing.py`](src/annealing.py) (hill climbing and simulated annealing). `python line_switching.py` prints the
numbers of Lab steps 1 and 2 and of Exercise 3, `python switching_ga.py` those of Lab steps 3 and 4 and of
Exercise 4, and `python annealing.py` those of Exercises 1 and 2. Every random number generator is seeded, so the
values below are reproducible; the times are those of one laptop CPU and will differ on yours.

## Lab

1. **Fitness evaluator.** pandapower ships MATPOWER's `case30`; `to_ppc` returns the MATPOWER matrices, whose
   branch column 5 (`rateA`) holds the ratings (16 to 130 MVA, read as MW in the DC model). The unscaled grid is
   N-1 secure: $E(\emptyset) = 0.000$ MW (one line sits exactly on its limit). Scaling every load and generator by
   1.2 gives $E(\emptyset) = 16.972$ MW, by 1.4 gives $145.351$ MW with 51 post-contingency overloads and a worst
   loading of 140.0 %; the lab uses 1.4. For a set $D$ the evaluator removes the lines, checks connectivity with
   `scipy.sparse.csgraph.connected_components`, rebuilds the PTDF $H$ of the reduced grid and the whole LODF
   matrix at once, $L_{\ell k} = (H A^\top)_{\ell k} / (1 - (H A^\top)_{kk})$ with $L_{kk} = -1$, and sums
   $\max(0, |f_\ell + L_{\ell k} f_k| - \bar f_\ell)$ over the non-bridge outages $k$ (a bridge has
   $(H A^\top)_{kk} = 1$; the unswitched grid has 3: the radial lines L12-13, L9-11 and L25-26). The LODF flows
   agree with a fresh DC power flow of the grid without the outaged branch to $10^{-13}$ MW
   (`dc_power_flow_check`, three cases checked). Each evaluation takes about 0.25 ms.
2. **Brute force.** $|D| \le 2$: $1 + 41 + \binom{41}{2} = 862$ sets in 0.2 s, 146 of them islanding; the
   optimum is $E = 27.343$ MW at $\{$L24-25, L6-28$\}$. $|D| \le 3$: 11 522 sets in 2.9 s, 3 302 islanding; the
   optimum is $E = 14.391$ MW at $\{$L22-24, L24-25, L6-28$\}$. $|D| \le 4$ has $112\,792$ sets, so at the
   measured rate it would take about 29 s, and $|D| \le 5$ (862 190 sets) about 4 minutes: on IEEE 30 the
   enumeration is still affordable, which is what makes it a usable baseline. On a 1 000-branch grid $|D| \le 3$
   alone is $1.7 \times 10^8$ sets.
3. **Genetic algorithm.** Population 32, tournament of size 3, 2 elites copied unchanged, crossover with
   probability 0.7 followed by the mutation of the worked example, a budget of $1\,152$ evaluations (10 % of the
   $|D| \le 3$ enumeration). Every call of the fitness is counted, including those that return $-\infty$. The
   crossover of two parents samples $k$ unique indices from the union of their lines (fewer when the union is
   smaller), so a child never has a duplicate index, and `normalize` keeps equal sets equal.
4. **Compare.** Ten seeds with the penalty for islanding candidates:

    | seed | best $E$ [MW] | lines | evaluations to the optimum | seconds | islanding evaluations |
    |---|---|---|---|---|---|
    | 0 | 14.391 | L22-24, L24-25, L6-28 | 503 | 0.48 | 177 |
    | 1 | 14.391 | L22-24, L24-25, L6-28 | 434 | 0.52 | 145 |
    | 2 | 14.391 | L22-24, L24-25, L6-28 | 357 | 0.44 | 163 |
    | 3 | 14.391 | L22-24, L24-25, L6-28 | 979 | 0.44 | 138 |
    | 4 | 22.943 | L24-25, L29-30, L6-28 | not reached | 0.44 | 138 |
    | 5 | 14.391 | L22-24, L24-25, L6-28 | 318 | 0.44 | 147 |
    | 6 | 14.391 | L22-24, L24-25, L6-28 | 967 | 0.54 | 167 |
    | 7 | 14.391 | L22-24, L24-25, L6-28 | 497 | 0.48 | 170 |
    | 8 | 14.391 | L22-24, L24-25, L6-28 | 387 | 0.50 | 161 |
    | 9 | 14.391 | L22-24, L24-25, L6-28 | 45 | 0.49 | 163 |

    Nine seeds reach the brute-force optimum, after 45 to 979 evaluations; seed 4 ends at the second-best set of
    the whole enumeration ($22.943$ MW). The mean best value is $15.246$ MW. The figure drawn by
    `plot_convergence` has one line per seed, best-so-far $E$ on a log axis against evaluations: every curve
    drops from a few hundred MW to below 30 MW within the first 200 evaluations, because
    $\{$L24-25, L6-28$\}$ and its 3-line extensions are easy to hit, and then shows a flat step whose length is
    the time the population needs to find L22-24 as the third line. The spread comes from the random
    initialization and the random order of mutations: once a population is full of sets containing L24-25 and
    L6-28, finding the third line is a draw of one slot among 39 candidates per child, so the waiting time is
    roughly geometric, which is why the "evaluations to the optimum" column spans a factor of 20. Seed 4 is a
    population that converged on $\{$L24-25, L29-30, L6-28$\}$, a local optimum of the "change one slot"
    neighborhood (14 sets of the enumeration are below 30 MW, and 13 of them contain L24-25 and L6-28), and the two elites kept
    it there until the budget ran out. Ten runs are too few to say more; Chapter 26 does the statistics.

Solutions: `main()` of `line_switching.py` and `switching_ga.py`; the notebook [lab.ipynb](src/lab.ipynb) has the
steps with empty cells.

```python title="line_switching.py"
--8<-- "docs/part-4-optimization/ch24-evolutionary-algorithms/src/line_switching.py"
```

```python title="switching_ga.py"
--8<-- "docs/part-4-optimization/ch24-evolutionary-algorithms/src/switching_ga.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch24-evolutionary-algorithms/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

A genome has $k$ slots; "change one slot" overwrites one slot with one of the $n$ line indices or the sentinel,
so there are $k(n + 1)$ raw moves, $3 \times 42 = 126$ for the lab. After normalization several moves give the
same set, so count sets instead. If the genome holds $j \le k$ distinct lines, a real slot can be swapped for one
of the $n - j$ absent lines ($j(n - j)$ neighbors), overwritten with the sentinel or with a line already present
($j$ neighbors, each dropping one line), and, if $j < k$, a sentinel slot can take one of the $n - j$ absent lines
($n - j$ neighbors):

$$
N(j) = j(n - j) + j + [\,j < k\,](n - j).
$$

With $n = 41$, $k = 3$: a full genome has $3 \times 38 + 3 = 117$ neighbors, a two-line genome
$2 \times 39 + 2 + 39 = 119$ and the empty genome 41. `annealing.neighbors` enumerates the sets and finds the
same counts. The two-line toy example (`toy_fitness`): four lines, $k = 2$, fitness

| set | $\{0, 1\}$ | $\{2, 3\}$ | any mixed pair | any single line | $\emptyset$ |
|---|---|---|---|---|---|
| fitness | 10 | 20 | 0 | 5 | 0 |

From $\{0, 1\}$ every neighbor ($\{0, 2\}$, $\{0, 3\}$, $\{1, 2\}$, $\{1, 3\}$, $\{0\}$, $\{1\}$) scores at most
5, so hill climbing evaluates the 7 candidates and stops with 10. Simulated annealing at temperature $T$ accepts
the step down to a mixed pair with probability $e^{-10/T}$, and from $\{0, 2\}$ one more slot change reaches
$\{2, 3\}$; with $T_0 = 10$, $\alpha = 0.97$ and 200 steps, 18 of 20 seeded runs end at 20.

```python title="annealing.py (toy landscape and hill climbing)"
--8<-- "docs/part-4-optimization/ch24-evolutionary-algorithms/src/annealing.py:toy"
```

## Exercise 2

`anneal_case` runs one mutation per step, accepts a worse candidate with probability $e^{\Delta / T}$ and cools
geometrically, $T_i = 20 \times 0.995^i$, over the same 1 152 evaluations as the GA ($T$ falls to about 0.06 at
the end, so the last steps are pure hill climbing). Ten seeds:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| best $E$ [MW] | 14.391 | 14.391 | 14.391 | 14.391 | 14.391 | 14.391 | 14.391 | 14.391 | 14.391 | 14.391 |
| evaluations to the optimum | 408 | 803 | 174 | 231 | 497 | 418 | 168 | 174 | 333 | 531 |

All ten runs reach the brute-force optimum (mean and median $14.391$ MW), against 9 of 10 for the GA (mean
$15.246$ MW, median $14.391$). On this small problem a single-state method with a few hundred accepted downhill
moves explores enough; the GA's elitism is what trapped seed 4. Do not conclude that annealing is better in
general: Chapter 26 shows that ten runs separate the two only if the effect is large, and here the medians are
equal.

```python title="annealing.py"
--8<-- "docs/part-4-optimization/ch24-evolutionary-algorithms/src/annealing.py"
```

## Exercise 3

Take $a = $ L24-25 and $b = $ L6-28, the best pair of the enumeration: $E(\emptyset) = 145.351$,
$E(\{a\}) = 94.869$, $E(\{b\}) = 138.910$ and $E(\{a, b\}) = 27.343$ MW. The single gains are $50.5$ and
$6.4$ MW, so additivity would predict $E(\{a, b\}) = 88.427$ MW; the pair removes $118.0$ MW, more than twice the
sum. Two mechanisms, both visible in the LODFs (`per_line_energy` prints the contributions per branch):

- **The LODFs of the reduced grid are not those of the original.** After removing $a$, the PTDF changes, and
  with it the LODF column of every outage, including the one that overloads L6-8 (32 MW limit), the branch that
  carries 102.07 of the 145.35 MW of the base energy. Removing $b$ alone shifts flow onto L25-27 and L24-25 and
  creates new overloads there (13.47 and 11.43 MW); removing $a$ as well deletes one of those branches and
  removes the other overload, so the sum of two single effects counts damage that the pair never does.
- **The energy is a sum of hinges $\max(0, \cdot)$.** Each switch alone lowers L6-8's post-contingency flows but
  leaves them above 32 MW in many outages (its contribution drops to 57.53 MW with $a$, to 65.79 MW with $b$);
  together they push the flows below the limit and the contribution falls to zero. A reduction that stays above
  the limit is credited in full, one that crosses it is credited only down to the limit, so gains superadd near
  the limit and subadd far from it.

Greedy selection adds the single best line at each step: `greedy` picks L25-27 ($69.449$ MW), then L6-28
($33.283$), then L29-30 ($28.883$), and stops there. It never looks at L24-25, whose single gain is smaller than
L25-27's, so it ends at twice the true $|D| \le 3$ optimum of $14.391$ MW ($\{$L22-24, L24-25, L6-28$\}$), and
it cannot backtrack. A heuristic that mutates sets and keeps a population can hold L24-25 and L6-28 together even
when neither is the best single move, which is the whole argument for population-based search on this problem.

## Exercise 4

The same ten seeds with `repaired=True`, where a child that islands the grid loses lines in random order until it
is connected (a connectivity check, not a fitness evaluation), against the penalty of Lab step 4:

| | penalty | repair |
|---|---|---|
| islanding evaluations per run (of 1 152) | 156.9 (13.6 %) | 0.0 |
| mean best $E$ [MW] | 15.246 | 16.101 |
| optimum reached | 9 of 10 | 8 of 10 |

Repair wastes no evaluations; with the penalty 138 to 177 evaluations per run return $-\infty$ and teach the
population nothing. Yet repair did not improve the outcome here: it reached the optimum in 8 seeds instead of 9
and its mean is worse. Dropping lines biases the children towards smaller sets, and the optimum needs all three
slots; the repaired children also take about 0.8 s per run instead of 0.5 s because of the connectivity checks.
On a grid with many radial lines (3 302 of the 11 522 sets island IEEE 30) repair is the better default, but it
has to be combined with an operator that adds lines back, which is what ToOp's `add_disconnection_prob` does.
The honest conclusion from ten runs is "repair wastes fewer evaluations, and the final quality is not
distinguishable"; Chapter 26 gives the test for the second half.

## Exercise 5

The No Free Lunch theorems average over *all* fitness functions on a finite search space, including every
permutation of the values: over that set, knowing the fitness of some sets says nothing about the fitness of
others, so no algorithm can exploit anything and all do equally well. Line switching is not drawn from that set.
Its fitness has structure: the flows are a smooth function of the admittance matrix, the LODF of a set changes
little when one far-away line is added, sets sharing two lines of a good triple tend to be good (14 of the 11 522
sets are below 30 MW and 13 of them contain L24-25 and L6-28), and islanding sets can be recognized without a
power flow. Operators that respect this structure, a mutation that changes one line, a crossover that keeps the
common lines of two good parents, repair that keeps the grid connected, deduplication that avoids paying twice
for the same set, exploit the correlation between neighboring sets that NFL assumes away. The theorem is a warning
against claiming a universally best optimizer, not a license to ignore the problem: the choice of operators is
exactly the place where knowledge about the problem class enters, and it is why ToOp's mutation acts on
substations and disconnections rather than on random bit strings.
