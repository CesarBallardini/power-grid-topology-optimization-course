# Chapter 29 – Solutions

Solutions of the Lab and the Exercises of [Chapter 29](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`grid_env.py`](src/grid_env.py) (a five-bus topology-control environment in
NumPy with the do-nothing, random, greedy-by-simulation and Q-learning agents), [`surrogate_mlp.py`](src/surrogate_mlp.py)
(the JAX surrogate of the alternative track), [`value_iteration.py`](src/value_iteration.py),
[`action_space.py`](src/action_space.py), [`jax_susceptance.py`](src/jax_susceptance.py) (which extends the worked
example's [`jax_ptdf.py`](src/jax_ptdf.py)) and [`surrogate_ga.py`](src/surrogate_ga.py). The IEEE 30 data come
from Chapter 27's `ch27_ieee30.py`. Each module prints the numbers quoted below when run as a script. Grid2Op,
LightSim2Grid, Gymnasium and Stable-Baselines3 are not installed in the course environment: the Grid2Op parts of
the lab run in the labs container (`docker/labs`, Grid2Op 1.12.6, LightSim2Grid 1.2.0, Stable-Baselines3 2.9.0 with
CPU-only PyTorch 2.14.1) with [`grid2op_agents_labs.py`](src/grid2op_agents_labs.py) and
[`dqn_labs.py`](src/dqn_labs.py), on the full `l2rpn_case14_sandbox` dataset and with the container limited to one
CPU core (`--cpus=1`); the notebook is [lab_labs.ipynb](src/lab_labs.ipynb). Each step below gives the Grid2Op
result first and then the same experiment on the small NumPy environment of the fallback track.

## Lab

1. **Environment.** With Grid2Op the call is `grid2op.make("l2rpn_case14_sandbox", test=True,
   backend=LightSimBackend())`; `test=True` gives the two chronics that ship with Grid2Op, and without it the
   full dataset (1,004 chronics, a 294 MB download kept in the container's data volume) is used here. The grid has
   14 substations, 20 lines, 6 generators and 11 loads; `obs.rho` has 20 loadings, `obs.topo_vect` 57 entries
   (one per element: the busbar it sits on, −1 if disconnected) and `obs.line_status` 20.
   `env.action_space.get_all_unitary_topologies_set(env.action_space)` returns 178 single-substation
   reconfigurations, most of them at the big substations: 57 at substation 5, 31 at substation 3, 29 at
   substation 1, none at substation 7, numbering from 0 as Grid2Op does (compare the 26 of a six-element
   substation in [Exercise 2](#exercise-2)).

    ```python title="grid2op_agents_labs.py (environment)"
    --8<-- "docs/part-4-optimization/ch29-learning-based-control/src/grid2op_agents_labs.py:env"
    ```

    The stand-in `grid_env.py` of the fallback track has 5 buses, 6 lines of 15 to 130 MW, a slack generator at bus 1
    and loads at buses 3, 4 and 5 (35 %, 40 %, 25 % of a daily profile with a 150 MW evening peak, 4 % noise per
    step); the observation is the vector `rho` of the six loadings, the topology is one of 7 (every line closed, or
    one line open: none of the six is a bridge), and the 8 actions are "do nothing" and "switch to topology
    $t$". At the peak the meshed topology loads the weak tie line 2–3 (15 MW) to 114 %, two switches relieve it
    (open 2–3: 70 %; open 4–5: 80 %) and the other four are destructive (250 % to 583 %), which is the shape of
    the real action space: most topology actions are harmful. A line above 100 % for more than 2 consecutive steps
    (Grid2Op's soft-overflow allowance) or above 150 % trips and ends the episode, with a reward of −5.

    ```python title="grid_env.py (environment)"
    --8<-- "docs/part-4-optimization/ch29-learning-based-control/src/grid_env.py:env"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch29-learning-based-control/src){ target="_blank" } to see or download all the files of this chapter.

2. **Baselines.** The reward per step is the same on both tracks: minus the summed overload (in units of rating),
   minus 0.02 per topology change, minus 5 when a game over ends the day early. With Grid2Op the ten "seeds" are
   ten days: the first day (288 five-minute steps) of the ten chronics among 0–299 with the highest do-nothing
   peak loading (`chronic_scan`, 101 s for the 300 days): 248, 72, 3, 57, 154, 194, 106, 181, 220 and 240. Only
   four of the 300 first days end in a game over for do-nothing, and all four are among these ten, so the
   evaluation days are the hard ones by construction. The random and greedy agents act when the maximum loading
   passes 90 %, over all 178 unitary actions.

    | agent (Grid2Op) | survival, median (min) of 288 | mean max `rho` | reward | ms per step |
    |---|---|---|---|---|
    | do-nothing | 288 (3) | 0.748 | −2.02 | 0.85 |
    | random (acts above 90 %) | 3 (1) | 0.732 | −6.01 | 0.96 |
    | greedy by simulation | 288 (288) | 0.729 | −0.07 | 4.22 |

    Do-nothing ends three days at step 3 (days 248, 72 and 3 start with a line above its rating, which trips when
    Grid2Op's overflow allowance runs out, and the grid does not survive it) and day 57 at step 232 in the evening.
    The random agent is far worse: it acts as soon as a line passes 90 %, which happens early on these days, plays
    random topologies and dies within 6 steps on eight of the ten days. The greedy agent survives all ten days with
    the lowest mean maximum loading; it ran 6,623 simulations (`obs.simulate`, an AC power flow with LightSim2Grid
    each), 662 per day, which makes its steps five times as slow as do-nothing's (4.22 against 0.85 ms on one core).

    ```python title="grid2op_agents_labs.py (agents, Q-learning, comparison)"
    --8<-- "docs/part-4-optimization/ch29-learning-based-control/src/grid2op_agents_labs.py:agents"
    ```

    On the stand-in of the fallback track: ten seeds (one noisy day each), episodes of 288 steps.

    | agent | survival, median (min) of 287 | mean max `rho` | reward | ms per step |
    |---|---|---|---|---|
    | do-nothing | 217 (216) | 0.597 | −5.14 | 0.02 |
    | random (acts above 90 %) | 212 (210) | 0.595 | −6.65 | 0.02 |
    | greedy by simulation | 287 (287) | 0.552 | −0.02 | 0.02 |

    Do-nothing loses line 2–3 around step 217 on every seed, three consecutive steps above 100 % into the evening
    peak. The random agent acts when a line
    passes 90 %, mostly with one of the four destructive topologies, and dies a few steps earlier. The greedy
    agent simulates all 8 actions when the loading passes 90 % and plays the one with the lowest simulated
    maximum (`env.simulate`, the counterpart of Grid2Op's `obs.simulate`): it opens line 2–3 once, in the
    evening, and finishes the day with a mean maximum loading of 0.55. Wall-clock time is 0.02 ms per step for
    every agent on this grid; the greedy agent's 8 simulations per decision cost nothing here, but they are
    8 power flows per step on a real grid (hundreds of actions on case 14, each an AC solve in Grid2Op), which is
    the cost the learned agent is meant to avoid.

3. **Tabular Q-learning.** With Grid2Op the state is the bin of the maximum loading (6 bins with edges 0.6, 0.8,
   0.9, 1.0, 1.1), the index of the most loaded line and whether the grid is still in its reference topology:
   $6 \times 20 \times 2 = 240$ states. The training days are kept apart from the evaluation days: the twenty first
   days of chronics 300–1003 with the highest do-nothing peak loading (212 s to scan; 20 of those 704 days end
   in a game over for do-nothing, 6 of them at step 3). On them the greedy agent ran 14,857 simulations in 38 s
   and played only 18 distinct actions (action 137 of the unitary list 21 times, 89 13 times, 132 8 times, the
   rest 6 times or fewer), so the reduced action set is do-nothing plus those 18, two short of the 20 the lab
   asks. Training: 200 days (ten passes over the twenty), $\alpha = 0.2$, $\gamma = 0.95$, $\varepsilon$
   decaying from 0.3 to 0.01, 28 s; the table visits 56 of its 240 states.

    | agent (Grid2Op) | survival, median (min) of 288 | mean max `rho` | reward | ms per step |
    |---|---|---|---|---|
    | do-nothing | 288 (3) | 0.748 | −2.02 | 0.85 |
    | greedy by simulation | 288 (288) | 0.729 | −0.07 | 4.22 |
    | Q-learning | 288 (138) | 0.781 | −1.82 | 0.79 |

    Day by day, Q-learning survives 288, 288, 288, 237, 153, 138, 288, 288, 288 and 288 steps against
    do-nothing's 3, 3, 3, 232 and six full days. It learned what the training days taught: an overloaded line at
    the start of the day is relieved by a switch, so it survives the three step-3 crises. But it also switches on
    days 154 and 194, where doing nothing was safe, and loses them at steps 153 and 138. Paired against
    do-nothing (Chapter 26) it is better on 4 days, worse on 2 and equal on 4: a sign test gives $p = 0.69$, no
    evidence either way with ten days, although the mean reward is higher (−1.82 against −2.02). Against the
    greedy agent it is never better. It runs at the cost of do-nothing (0.79 ms per step), with no simulations.

    On the stand-in of the fallback track the state is the bin of the maximum loading (6 bins with edges 0.6, 0.8,
    0.9, 1.0, 1.1), the index of the most loaded line and the current topology: $6 \times 6 \times 7 = 252$ states.
    The action set is reduced as the lab asks: do-nothing plus the actions the greedy agent played most often on ten
    training days, which is a single one here (action 4, open line 2–3; the greedy agent never needed another), so the
    table has $252 \times 2$ entries. Training: 500 days drawn from one seed, $\alpha = 0.2$, $\gamma = 0.95$,
    $\varepsilon$-greedy with $\varepsilon$ decaying from 0.3 to 0.01, about 4 s. The decay matters: with a constant
    $\varepsilon$ the agent explores the switch at some point of the morning on almost every day, stays in the safe
    topology, and never visits the meshed grid at the peak, so the states that matter are never learned and the greedy
    policy of the table is do-nothing (this was the first attempt). The learned policy, evaluated on the ten baseline
    seeds: survival 287 (287), mean maximum loading 0.557, reward −0.04, 0.03 ms per step. It survives every day with
    a single switch, like the greedy agent, but later: the greedy agent acts when line 2–3 passes 90 % (steps 209–212
    of the ten days), the table only once the loading is in the bin above 1.0 (steps 213–216), so the Q agent
    tolerates one or two steps of a few percent overload (0.002 to 0.07 units of rating per day) that cost it the 0.02
    of reward and the slightly higher mean loading. With ten paired seeds (Chapter 26) the difference is 0 in survival
    and 0.02 in reward on every day: real, but a consequence of the state discretization, not of the learning. What
    the agent does not know: the table maps a coarse state to an action with no model of the grid; it has never seen a
    load pattern outside its training days, a second line at risk, or a day where opening 2–3 is the wrong move, while
    the exhaustive search re-checks every action against the actual next state at every step. The agent's advantage is
    that it does not need those simulations at run time; its risk is that it cannot tell when its table is wrong. The
    Grid2Op table shows the same risk at full size: it plays a switch it learned on a crisis day on a day that did not
    need one, and nothing in its 240 states tells it the difference.

    ```python title="grid_env.py (agents and Q-learning)"
    --8<-- "docs/part-4-optimization/ch29-learning-based-control/src/grid_env.py:agents"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch29-learning-based-control/src){ target="_blank" } to see or download all the files of this chapter.

4. **Optional DQN.** `Case14Gym` wraps the Grid2Op day in a `gymnasium.Env`: the observation is the 20 loadings
   (clipped at 10) plus the reference-topology flag, the action space `Discrete(19)` (the reduced set of step 3),
   the reward that of step 2, and a game over before step 288 terminates the episode. Stable-Baselines3's
   `DQN('MlpPolicy')` with two hidden layers of 64, learning from step 1,000, $\gamma = 0.95$, exploration
   decaying over the first half of the budget, trains for 20,000 environment steps (about 70 days of
   experience) cycling over the training days of step 3: 384 s on one core, 19.2 ms per step, most of it the
   network updates (an environment step costs about 0.8 ms). Exploration ends most training days early: the
   20,000 steps make 797 episodes, 25 steps each on average.

    | agent (Grid2Op) | survival, median (min) of 288 | mean max `rho` | reward | ms per step |
    |---|---|---|---|---|
    | do-nothing | 288 (3) | 0.748 | −2.02 | 0.78 |
    | greedy by simulation | 288 (288) | 0.729 | −0.07 | 4.05 |
    | DQN, 20,000 steps | 58 (3) | 0.731 | −6.13 | 1.21 |

    (This run was separate from the one of step 2, hence slightly different times.) The trained network survives
    3, 3, 3, 288, 3, 112, 288, 288, 3 and 288 steps: it fixes day 57 but none of the step-3 crises, and it
    causes game overs on days 154, 194 and 220, which do-nothing survives. With this budget the training cost is
    paid and buys nothing; the table of step 3, 28 s of training over a coarse state and the same actions, does
    better. The comparison the lab asks for: the greedy agent spends 6,623 simulations on the ten evaluation days
    (662 per day, every day, at run time) and no training; the DQN spends 20,000 training steps once (6.4 min) and
    then 1.2 ms per step. The trade pays only when the network is good, which takes far more than 70 days of
    experience on this grid, and only while the deployment days look like the training days.

    ```python title="dqn_labs.py"
    --8<-- "docs/part-4-optimization/ch29-learning-based-control/src/dqn_labs.py:dqn"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch29-learning-based-control/src){ target="_blank" } to see or download all the files of this chapter.

    On the stand-in of the fallback track the same wrapper is a `gymnasium.Env` around `GridEnv` with
    `observation_space = Box(0, 10, (6,))` (the loadings) and `action_space = Discrete(8)`; it was not trained. The
    table of step 3 already plays the single useful switch there, so a network could at best match it after
    $10^4$–$10^5$ environment steps (35 to 350 days), against zero training for the greedy agent, whose cost is 8
    simulations per decision instead.

5. **Alternative track (surrogate).** IEEE 30 with pandapower's `case30` injections scaled per bus by a uniform
   factor in $[0.5, 1.5]$ (slack balancing): 20,000 training samples, flows from the PTDF. A two-layer MLP
   (29 inputs, 64 tanh units, 41 outputs, full-batch Adam, 1,500 steps, 14 s on CPU) reaches a test error of
   0.19 % of the rating on average and 0.43 % at worst (line index 9) on 2,000 unseen injections: the map is
   linear, so a small network learns it almost exactly. Speed: the exact flows are one matrix product with the
   PTDF, 0.11 ms for 2,000 cases, the MLP 0.95 ms, a speed-up of 0.12, that is a slow-down. A surrogate of a
   *linear* DC power flow cannot beat the PTDF, which is already the cheapest possible evaluator; surrogates pay
   off against AC power flows (Newton iterations, convergence failures) and against repeated topology changes,
   where the PTDF must be rebuilt. Generalization: with line 25–27 switched off (never seen in training) the
   error grows to 6.5 % of the rating on average and 47.8 % on line index 32 (24–25, the line that takes over
   the flow), 35 times the intact error. The network learned one PTDF; a topology change is a different linear
   map, and nothing in the input tells the network that the grid changed. That is the generalization problem
   graph neural solvers address by putting the topology into the input (messages along the lines that exist),
   and the reason ToOp's AC validation recomputes rather than predicts.

    ```python title="surrogate_mlp.py (data)"
    --8<-- "docs/part-4-optimization/ch29-learning-based-control/src/surrogate_mlp.py:data"
    ```

    ```python title="surrogate_mlp.py (model)"
    --8<-- "docs/part-4-optimization/ch29-learning-based-control/src/surrogate_mlp.py:mlp"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch29-learning-based-control/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

Take the states $s_0$ = "secure" and $s_1$ = "overloaded", the actions $a_0$ = wait and $a_1$ = switch, and the
model of `value_iteration.py`: waiting keeps a secure grid secure with probability 0.9 and an overloaded one
overloaded with probability 0.7; switching lands in the secure state with probability 0.8 from either state;
the rewards are $R(s_0, a_0) = 0$, $R(s_1, a_0) = -1$ (one unit per overloaded step) and
$R(s, a_1) = -0.1$ (the switching cost). The Bellman optimality equation is

$$
V^*(s) = \max_{a} \Bigl[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s') \Bigr],
$$

that is, with $\gamma = 0.9$,

$$
\begin{aligned}
V^*(s_0) &= \max\bigl\{\, 0.9\,[0.9\,V^*(s_0) + 0.1\,V^*(s_1)],\;\; -0.1 + 0.9\,[0.8\,V^*(s_0) + 0.2\,V^*(s_1)] \,\bigr\}, \\
V^*(s_1) &= \max\bigl\{\, -1 + 0.9\,[0.3\,V^*(s_0) + 0.7\,V^*(s_1)],\;\; -0.1 + 0.9\,[0.8\,V^*(s_0) + 0.2\,V^*(s_1)] \,\bigr\}.
\end{aligned}
$$

Value iteration from $V_0 = (0, 0)$: $V_1 = (\max\{0, -0.1\}, \max\{-1, -0.1\}) = (0, -0.1)$;
$V_2 = (\max\{0.9 \cdot 0.1 \cdot (-0.1),\; -0.1 + 0.9 \cdot 0.2 \cdot (-0.1)\}, \max\{-1 + 0.9 \cdot 0.7 \cdot (-0.1),\;
-0.1 + 0.9 \cdot 0.2 \cdot (-0.1)\}) = (-0.009, -0.118)$; $V_3 = (-0.0179, -0.1277)$, and so on. The operator is a
$\gamma$-contraction, so the error shrinks by at least 0.9 per step and the iteration converges, after 177
steps for a tolerance of $10^{-10}$, to $V^* = (-0.0989, -0.2088)$ with the policy "wait when secure, switch when
overloaded". The check: solving $(I - \gamma P_\pi) V = R_\pi$ for that policy gives the same values, and the
three other deterministic policies are worse: always wait $(-1.96, -4.13)$, always switch $(-1, -1)$, switch
when secure and wait when overloaded $(-3.95, -5.58)$. The lesson for topology control is the structure of the
answer: a small switching cost is worth paying as soon as the grid is overloaded, and not before.

```python title="value_iteration.py"
--8<-- "docs/part-4-optimization/ch29-learning-based-control/src/value_iteration.py:vi"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch29-learning-based-control/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

Each of the 6 elements goes to busbar A or B: $2^6 = 64$ labelled assignments. Swapping the two busbars gives the
same electrical topology, so there are $2^5 = 32$ distinct ones (every assignment and its mirror image, and no
assignment is its own mirror). Not all are useful: an assignment that leaves one element alone on a busbar is a
disconnection in disguise, and one that leaves a busbar empty is the unsplit configuration. ToOp's rule (both
busbars keep at least two branches) leaves $2^{5} - 6 - 1 = 25$ splits ([Chapter 19](../../part-3-power-systems/ch19-operational-security/index.md),
Exercise 8), 26 configurations with the unsplit one; Grid2Op's `get_all_unitary_topologies_set` applies the same
"no busbar with a single element" rule and the enumeration in `action_space.py` gives 26 as well. For 14 such
substations acting independently the joint space is $64^{14} = 2^{84} \approx 1.9 \times 10^{25}$ labelled,
$32^{14} = 2^{70} \approx 1.2 \times 10^{21}$ up to symmetry and $26^{14} \approx 6.5 \times 10^{19}$ with the
validity rule: no table, no neural network output layer and no enumeration can cover it, which is the
action-space problem of the chapter. The reductions: *unitary actions* change one substation per step
($14 \times 25 = 350$ actions, a few hundred on case 14, the set the lab's greedy agent searches); *per-station
action sets* with validity filters, which is what ToOp enumerates in preprocessing (`n_actions_per_sub`, with the
bridge and isolation filters of Chapter 22) and then combines by evolution rather than enumeration; *pruning by
simulation*, keeping the actions that ever helped (Lab step 3 kept one of eight); and *hierarchical agents* that
first choose the substation, then the configuration.

```python title="action_space.py"
--8<-- "docs/part-4-optimization/ch29-learning-based-control/src/action_space.py:count"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch29-learning-based-control/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 3

`flows_b(p, b)` rebuilds $B_f = \mathrm{diag}(b) A$ and $B_{bus} = A^\top B_f$ from the susceptance vector, so
`jax.jacfwd(flows_b, argnums=1)` is the $3 \times 3$ matrix $\partial f_\ell / \partial b_k$. For the injections
of the worked example ($p = (-1, 0.5)$ at buses 1 and 2, flows $(0.6176, -0.3824, -0.1176)$ on lines 0–1, 1–2,
0–2):

$$
\frac{\partial f}{\partial b} =
\begin{pmatrix} 0.0145 & -0.0360 & 0.0043 \\ 0.0145 & -0.0360 & 0.0043 \\ -0.0145 & 0.0360 & -0.0043 \end{pmatrix}.
$$

The central difference $(f(b + h e_2) - f(b - h e_2)) / 2h$ with $h = 10^{-6}$ reproduces the third column to
$2 \times 10^{-11}$. Two structural facts are visible. The rows of lines 0–1 and 1–2 are equal and the row of 0–2
is their negative: in a single loop a change of any susceptance only moves flow between the two paths from bus 0
to bus 2 (through bus 1, or direct), so the nodal balances are untouched, $A^\top\, \partial f / \partial b_k = 0$.
And $\sum_k b_k\, \partial f_\ell / \partial b_k = 0$: the flows are homogeneous of degree zero in $b$, scaling
every reactance by the same factor changes nothing. For an operator the gradient says how the flows move when a
line's reactance changes: a series reactor or a phase-shifting transformer on line 1–2 (making $b_1$ smaller) pushes
flow from the path through bus 1 to the direct line at 0.036 MW per unit of susceptance, and the sign tells which
way. It is the continuous relaxation of switching: $b_k \to 0$ opens the line, and the derivative at the current
$b$ is the first-order estimate of the effect, exact for the linear DC model only along the path, since the
map $b \mapsto f$ is nonlinear (a rational function). It does not give the gradient with respect to a *binary*
status, and the linear extrapolation to $b_k = 0$ is wrong when the line carries much flow, which is why ToOp
evaluates discrete topologies exactly (LODF, BSDF) instead of following this gradient.

```python title="jax_susceptance.py"
--8<-- "docs/part-4-optimization/ch29-learning-based-control/src/jax_susceptance.py:grad"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch29-learning-based-control/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

The GA is the Chapter 24 one: genomes of at most $K = 2$ switched lines of IEEE 30 (loads scaled by 1.5 as in
Chapter 27, exact fitness `ch27_ieee30.overload_energy`, islanding penalized), population 20, tournament of two,
crossover by sampling from the union of the parents, mutation of one slot, elitist survival, 400 exact
evaluations per run. The target is the exact optimum with two lines, 44.6 MW (lines 24–25 and 6–28, Chapter 27).
The *assisted* variant generates five times as many offspring, scores them with a linear surrogate (one weight per
line plus a bias, least squares on every genome evaluated so far) and evaluates exactly only the best 20 %. Over
ten seeds:

| GA | runs reaching 44.7 MW | exact evaluations to target (median, min, max) | best value, median |
|---|---|---|---|
| plain | 6 of 10 | 75 (10, 313) | 44.6 MW |
| assisted | 3 of 10 | 23 (10, 25) | 155.8 MW |

The surrogate does what it is supposed to when it is right: the three successful assisted runs reach the target
in 23 exact evaluations against a median of 75. But it fails in seven runs out of ten, and those runs end far
from the optimum (median best value 155.8 MW): the linear model assumes that switching effects are additive, and
Chapter 24's Exercise 3 shows they are not (the pair 24–25 + 6–28 is much better than the sum of the two single
switches would suggest). The surrogate therefore ranks the offspring by the single-line effects it has seen,
steers the exact budget to combinations of individually good lines, and the population converges prematurely.
This is the model-management problem of surrogate-assisted evolution (Jin 2011): a pre-screening model needs
either a form that can represent interactions (pairwise terms, a Gaussian process, a GNN) or a fraction of the
exact budget spent on offspring the surrogate ranks low, to keep correcting it. ToOp's two-stage design is the
same idea with a physics model instead of a learned one: the DC solver is the cheap screen, exact for its own
objective, and the AC stage the expensive evaluation; a learned surrogate would sit in front of the DC stage and
inherit exactly this failure mode. A sign test on the ten paired runs gives no significant difference in
evaluations-to-target because the assisted GA has only three successes; the honest summary is "faster when it
works, and it works less often".

```python title="surrogate_ga.py"
--8<-- "docs/part-4-optimization/ch29-learning-based-control/src/surrogate_ga.py:ga"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch29-learning-based-control/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

**Brief: should the TSO deploy a reinforcement-learning topology agent?**

*Recommendation.* Not as a controller. Deploy it, if at all, as a proposal generator whose every suggestion is
simulated before a human sees it, and measure it against the exhaustive one-step search and the rule base for a
year before giving it more.

*What it offers.* A trained policy answers in milliseconds without simulating hundreds of actions (Lab step 2:
the greedy search costs one power flow per candidate action per step; the agent costs a table lookup or a
forward pass), and the L2RPN results show agents that keep case-14 and case-118 grids alive through days the
do-nothing baseline does not survive.

*Distribution shift.* The policy is only as good as the days it has seen. Training uses historical or synthetic
chronics; the next heat wave, a new wind farm, a different maintenance pattern or a market change produces states
outside that distribution, and a table or network gives a confident answer there with no warning (Lab step 3:
the agent cannot tell when its table is wrong). Retraining lags the grid; the lag is where the risk lives.

*Rare contingencies.* Reinforcement learning optimizes expected return. An N-1 event that happens once a year is
a negligible part of the return unless the reward is shaped to make it dominant, and then the agent learns a
different, more conservative behaviour that costs every day. ToOp's `overload_energy_n_1` is an N-1 metric by
construction; an RL reward that is not will leave contingency security to luck.

*Guarantees.* None: no feasibility guarantee (the proposed topology may island the grid, as four of the eight
actions of the small environment do), no optimality, no bound on how far the policy is from the best action, and
no explanation. A MILP gives bounds, ToOp gives a validated metric per proposal, the agent gives a number.

*Simulation checks.* The non-negotiable element. Every proposal is run through `obs.simulate` or an AC N-1
analysis before it reaches the operator, and rejected when it does not improve the validated metrics. This
turns the agent into a candidate generator, which is a role it can fill well, and removes the only failure mode
that matters, a wrong action executed. The cost is one or a few simulations per step, which the agent was
supposed to save; the saving is then the difference between simulating hundreds of actions and simulating one.

*How ToOp validates the proposals.* ToOp's AC stage is exactly this check: the agent's topology is applied to the
grid model, the AC N-1 is run, and `evaluate_acceptance` compares non-convergence, overload energy and critical
branches with the unsplit grid (Chapter 28; mind its zero-baseline trap). An agent's proposal can also seed ToOp's
DC repertoire, where it competes with the evolutionary candidates and is kept only if its cell has nothing
better; the operator then reads one `orao_summary.json` whatever the source of the topology.

*Interpretability and the human in the loop.* Operators act on proposals they can explain to the next shift.
Present the agent's output as "topology X, validated: overload −40 %, no new critical branch, 3 switch
operations", never as "the agent says X". Keep the rule base as the sanity check and the exhaustive search as the
reference the agent must beat on the same seeds (Chapter 26) before any scope is widened.
