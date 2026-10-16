# Chapter 6 – Solutions

Solutions of the Lab and the Exercises of [Chapter 6](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`newton.py`](src/newton.py) (the worked example),
[`newton_lab.py`](src/newton_lab.py) (the reusable solver, basins and the SciPy comparison; `python newton_lab.py`
prints every number of the Lab answers), [`newton_exercises.py`](src/newton_exercises.py) (`python
newton_exercises.py` prints those of Exercises 1–5) and [`fsolve_example.m`](src/fsolve_example.m) (Exercise 6,
Octave). The notebooks [`examples.ipynb`](src/examples.ipynb), [`lab.ipynb`](src/lab.ipynb) and
[`exercises.ipynb`](src/exercises.ipynb) import these modules.

## Lab

1. **A reusable Newton solver.** `newton_solve` below returns the solution, a success flag and the list of
   residual norms $\|F(\mathbf{x}_k)\|$ (the starting point included). Each step is one linear solve
   $J\,\Delta = -F$; `np.linalg.LinAlgError` (singular Jacobian) and a non-finite iterate end the run with
   `success = False`, and so does reaching `max_iter` above `tol`. The `damping` argument scales every step by
   a fixed fraction; `backtrack=True` halves that fraction until the residual decreases (step 5).

    ```python title="newton_lab.py: newton_solve"
    --8<-- "docs/part-1-math/ch06-newton-raphson/src/newton_lab.py:newton_solve"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-1-math/ch06-newton-raphson/src){ target="_blank" } to see or download all the files of this chapter.

2. **Residual histories and convergence order.** From $(1, -1)$ the norms are $2.1$, $8.6\cdot10^{-1}$,
   $4.0\cdot10^{-2}$, $1.0\cdot10^{-4}$, $8.4\cdot10^{-10}$, $0$: five steps to root
   A $= (1.004169, -1.729637)$. From $(-1, 1)$: $2.0$, $1.0$, $5.4\cdot10^{-2}$, $1.8\cdot10^{-4}$,
   $2.1\cdot10^{-9}$, $0$, five steps to root B $= (-1.816264, 0.837368)$. From $(2, 2)$ the residual first
   grows ($9.3$, $3.7$, $3.4$, $4.5$) and then collapses ($0.60$, $2.0\cdot10^{-2}$, $2.5\cdot10^{-5}$,
   $4.0\cdot10^{-11}$): seven steps, to root B, far from where it started. From $(0, 0)$ the Jacobian
   $\begin{bmatrix} 0 & 0 \\ 1 & 1\end{bmatrix}$ is singular and the solver returns failure at once. On
   `semilogy` the three convergent histories bend downwards, the picture of quadratic convergence. The ratio
   $\log r_{k+1} / \log r_k$ is $2.85$ then $2.27$ from $(1, -1)$, $2.95$ then $2.32$ from $(-1, 1)$, $2.71$
   then $2.26$ from $(2, 2)$: it approaches 2 from above, because
   $\log r_{k+1} / \log r_k = p + \log C / \log r_k$ for $r_{k+1} \approx C r_k^p$, and the second term only
   vanishes as $r_k \to 0$. The ratio for the first pair from $(1, -1)$ is $21.4$ and means nothing: $r_0 = 0.86$
   is so close to 1 that its logarithm is nearly zero. Estimate the order from the tail of the history only.

3. **Basins of attraction.** `basins()` iterates the $200 \times 200$ starting points together (the $2 \times 2$
   Jacobian is inverted by Cramer's rule) and labels each point by the root within $10^{-6}$ of its last
   iterate: 12 117 points reach root A, 27 673 reach root B and 210 fail (overflow or an exactly singular
   Jacobian). The map shows two broad regions, but the boundary between them is not a line: along the curve
   $\det J = 2x - 2y e^x = 0$, that is $y = x e^{-x}$, the picture is a ragged mixture of both colours, and
   139 of the 210 failures lie within $0.1$ of that curve. On the curve the Newton step is undefined; next to it
   the step is huge, so the iterate lands far away and in whichever basin that landing point belongs to. The
   boundary also follows the images of that curve under the Newton map (the points that reach the curve after
   one or more steps), which is why it looks fractal. Away from the curve the step is bounded and nearby
   starting points move together.

    ```python title="newton_lab.py: basins"
    --8<-- "docs/part-1-math/ch06-newton-raphson/src/newton_lab.py:basins"
    ```

4. **SciPy and Octave.** From $(1, -1)$ to a residual of $10^{-10}$:

    | Solver | Iterations | Function evaluations |
    |---|---|---|
    | `newton_solve` (analytic Jacobian) | 5 | 6 (plus 5 Jacobians) |
    | `optimize.root(..., method='hybr')` | not reported | 13 |
    | `optimize.root(..., method='broyden1')` | 16 | 27 |
    | Octave `fsolve` (Exercise 6) | 6 | 16 |

    `hybr` is MINPACK's hybrid Powell method: a trust-region step with a Jacobian built by finite differences
    and then updated by Broyden's formula, so its `nfev` includes the two extra evaluations of each Jacobian
    rebuild and it reports no iteration count. `broyden1` starts from the identity scaled by its `alpha`
    parameter, which is why it needs three times the iterations of Newton. Newton with an analytic Jacobian is
    hard to beat on a 2-variable problem; the comparison turns in favour of quasi-Newton methods when the
    Jacobian is expensive and the dimension large (Exercise 5).

    ```python title="newton_lab.py: scipy_comparison"
    --8<-- "docs/part-1-math/ch06-newton-raphson/src/newton_lab.py:scipy"
    ```

5. **Damping.** A fixed `damping=0.5` does not rescue the failures: 345 points fail instead of 210, because
   half a step from a point near the singular curve still lands far away, and halving every step keeps later
   iterates hovering near the curve for longer. Its cost near the solution is severe: the error is only halved
   each step (the convergence is linear with ratio $1/2$, a straight line on `semilogy`), and the run from
   $(1, -1)$ takes 34 steps instead of 5. Backtracking (start from the full step, halve it until the residual
   decreases) removes every failure: 0 of the 40 000 points fail, 13 105 reach root A and 26 895 root B. Its
   cost near the solution is nil, because the full Newton step always decreases the residual there and is
   accepted at once: the run from $(1, -1)$ still takes 5 steps. Backtracking only pays for the extra residual
   evaluations of the rejected trial steps, far from the solution. In the basin map the ragged band along
   $\det J = 0$ remains (the step is still huge there, only shortened until the residual improves), but every
   point now ends at one of the two roots.

6. **Keep the function.** `newton_solve` in `newton_lab.py` takes any `f` and `jac`; in
   [Chapter 15](../../part-3-power-systems/ch15-ac-power-flow/index.md) the residual is the power mismatch and
   the Jacobian the power-flow Jacobian. Nothing to compute here.

## Exercise 1

For $f(x) = x^2 - 2$ the Newton step is $x_{k+1} = x_k - (x_k^2 - 2)/(2x_k) = \tfrac12 (x_k + 2/x_k)$. From
$x_0 = 1$:

| $k$ | $x_k$ | $e_k = \lvert x_k - \sqrt2 \rvert$ | $e_k / e_{k-1}^2$ |
|---|---|---|---|
| 0 | $1$ | $4.1\cdot10^{-1}$ | |
| 1 | $1.5$ | $8.6\cdot10^{-2}$ | $0.500$ |
| 2 | $1.416667$ | $2.5\cdot10^{-3}$ | $0.333$ |
| 3 | $1.414216$ | $2.1\cdot10^{-6}$ | $0.353$ |

Subtracting $\sqrt2$ from the step gives exactly $x_{k+1} - \sqrt2 = (x_k - \sqrt2)^2 / (2x_k)$, so
$e_{k+1} = e_k^2 / (2x_k) \to e_k^2 / (2\sqrt2) = 0.354\, e_k^2$: the error is squared each step, with a
constant that the last ratio already reproduces to three digits. `sqrt2_iterates()` prints the table.

## Exercise 2

The Newton step on $f(x) = \arctan x$, $f'(x) = 1/(1 + x^2)$, is $x_1 = x_0 - (1 + x_0^2)\arctan x_0$, so
$x_1 = -x_0$ exactly when $2x_0 = (1 + x_0^2)\arctan x_0$. `brentq` on $[1, 2]$ gives $x_0^\ast = 1.3917$,
and from it the iteration flips sign forever. Below the threshold the sequence shrinks: from $1.3$ the
iterates are $-1.16$, $0.859$, $-0.374$, $0.034$, $-2.6\cdot10^{-5}$, $1.2\cdot10^{-14}$. Above it the sequence
grows: from $1.5$ they are $-1.69$, $2.32$, $-5.11$, $32.3$, $-1.6\cdot10^{3}$, $3.9\cdot10^{6}$, because far
from the origin $f'$ is tiny and each step $\approx x^2 \arctan x$ overshoots by more than the distance to
the root. With `damping=0.5` from $1.5$ the first half-step lands at $-0.097$, inside the convergent region,
and from there the iterates halve each step ($-0.048$, $-0.024$, $-0.012$, ...): the divergence is cured but
the convergence is now linear with ratio $1/2$, the price of a fixed damping. The backtracking rule of the
lab's `newton_solve` (shorten the step only while the residual does not decrease) converges from $1.5$ in 4
steps: it damps the first step and takes full Newton steps afterwards.

```python title="newton_exercises.py: arctan"
--8<-- "docs/part-1-math/ch06-newton-raphson/src/newton_exercises.py:arctan"
```

## Exercise 3

With $f(x) = (x - 1)^2$ and $f'(x) = 2(x - 1)$ the step is $x_{k+1} = x_k - (x_k - 1)/2 = (x_k + 1)/2$, so the
error obeys $e_{k+1} = e_k / 2$ exactly. From $x_0 = 0$: $0.5$, $0.75$, $0.875$, $0.9375$, $0.96875$,
$0.984375$, with errors $1/2$, $1/4$, $1/8$, ... (`double_root_iterates()`). The quadratic estimate
$e_{k+1} = f''(\xi)\, e_k^2 / (2 f'(x_k))$ assumes $f'(x_k) \to f'(r) \neq 0$; at a double root
$f'(x_k) = 2 e_k$ itself vanishes with the error, and the estimate degrades to $e_{k+1} = 2 e_k^2 / (4 e_k) =
e_k / 2$. In general a root of multiplicity $m$ gives linear convergence with ratio $(m - 1)/m$. The practical
sign is a residual that shrinks by a constant factor per step instead of squaring: in a power flow that is the
signature of a case at its loadability limit, where the Jacobian becomes singular (Exercise 4).

## Exercise 4

Newton on $g(\delta) = \sin\delta - PX$, $g'(\delta) = \cos\delta$, from $\delta_0 = 0$ (`angle_iterates`):

| $PX$ | Iterates (rad) | Result |
|---|---|---|
| $0.5$ | $0.5$, $0.5234$, $0.5236$ | $\delta = 30.00^\circ$ in 3 steps |
| $0.99$ | $0.99$, $1.2706$, $1.3880$, $1.4246$, $1.4292$, $1.4293$ | $\delta = 81.89^\circ$ in 6 steps |
| $1.2$ | $1.2$, $1.9395$, $1.1981$, $1.9359$, $1.1911$, $1.9228$, $1.1650$, $1.8773$, ... | no convergence |

For $PX = 0.99$ the convergence is slow at first because the root sits where $\cos\delta = 0.141$ is small:
the constant $g''/(2g') = \tan\delta / 2 \approx 3.5$ is large, and the first full step from $0$ (slope 1)
undershoots badly. For $PX = 1.2$ the equation has no real solution, since $\sin\delta \le 1 < 1.2$: the
iterates bounce between about $1.2$ and $1.9$ on either side of the maximum of $\sin\delta$, the residual never
drops below $0.2$, and after a dozen steps the sequence is still wandering (the last iterates computed are
$1.06$, $1.73$, $0.39$, $1.28$). A power flow solver stops at its iteration cap and reports non-convergence. Physically $P_{\max} = 1/X$ is the steady-state stability limit of the line; a dispatch asking
for more has no equilibrium. ToOp's AC validation runs Newton–Raphson power flows on each candidate topology;
a topology that drives a load flow past such a limit does not produce a flow to check against the ratings, so
the fact itself is counted in the `non_converging_loadflows` metric and the candidate is penalised.

## Exercise 5

Good Broyden with $B_0 = J(\mathbf{x}_0)$: the step is $\Delta = -H F(\mathbf{x})$ with $H = B^{-1}$, and
after the step, with $s = \Delta$ and $y = F(\mathbf{x} + \Delta) - F(\mathbf{x})$, the rank-1 update
$B^+ = B + (y - Bs)\, s^\top / (s^\top s)$ keeps $B^+ s = y$ (the secant condition) while changing $B$ as
little as possible. Sherman–Morrison ([Chapter 4](../ch04-numerical-linear-algebra/index.md)) turns it into
an update of the inverse without any linear solve:

$$
H^+ = H + \frac{(s - Hy)\, s^\top H}{s^\top H y}.
$$

From $(1, -1)$ Broyden takes 7 iterations against Newton's 5, with residual norms $2.1$, $8.6\cdot10^{-1}$,
$1.6\cdot10^{-1}$, $8.5\cdot10^{-3}$, $2.7\cdot10^{-5}$, $2.2\cdot10^{-6}$, $4.1\cdot10^{-8}$,
$9.6\cdot10^{-12}$: the first step is Newton's (same $B_0$), and afterwards the decrease is superlinear but
not quadratic. Started from $B_0 = I$ it needs 13 iterations. Each Broyden iteration costs one residual
evaluation and an $O(n^2)$ update instead of a Jacobian evaluation and an $O(n^3)$ solve, so it wins when
the Jacobian is expensive or unavailable, as in `hybr`, which uses the same update between rebuilds.

```python title="newton_exercises.py: broyden_good"
--8<-- "docs/part-1-math/ch06-newton-raphson/src/newton_exercises.py:broyden"
```

## Exercise 6

Octave's `fsolve` (trust-region dogleg, Jacobian by finite differences unless supplied) from $[1; -1]$ with
`TolFun = TolX = 1e-12` returns $x = (1.0042, -1.7296)$, $\|f\| = 9\cdot10^{-16}$, `info = 1` (converged) and
`output.iterations = 6`, `output.successful = 5`, `output.funcCount = 16`. The Newton code of the lab takes 5
iterations with 6 residual and 5 Jacobian evaluations; `fsolve` spends one more iteration and 16 function
evaluations, because each finite-difference Jacobian costs two extra evaluations and one of its trial steps was
rejected (`successful` counts the accepted ones). From $[-1; 1]$ it finds the other root in 6 iterations too.

```octave title="fsolve_example.m"
--8<-- "docs/part-1-math/ch06-newton-raphson/src/fsolve_example.m"
```
