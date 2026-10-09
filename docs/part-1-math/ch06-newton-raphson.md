# Chapter 6. Nonlinear equations and Newton–Raphson

**Weeks:** 1.5 · **Semester:** 2 · **Prerequisites:** [Chapter 4](ch04-numerical-linear-algebra.md)

!!! info "Why it matters for ToOp"

    AC validation runs Newton–Raphson power flows in pandapower (`algorithm: "nr"`,
    `tolerance_mva`) and PowSyBl OpenLoadFlow (`maxNewtonRaphsonIterations`)
    (`grid_helpers/pandapower/loadflow_parameters.py`, `grid_helpers/powsybl/loadflow_parameters.py`). During
    import ToOp tries three voltage initializations to find converging parameters
    (`docs/dc_solver/loadflow_parameters.md`), and non-convergence is reported as a metric
    (`non_converging_loadflows` in `interfaces/types.py`). The power-flow equations themselves come in
    [Chapter 15](../part-3-power-systems/ch15-ac-power-flow.md); this chapter is about the method.

## Topics

- Newton's method in one variable; fixed-point iteration; convergence order.
- Newton for systems; Jacobian; quadratic convergence and its preconditions.
- Initialization, damping, divergence; quasi-Newton (Broyden, itself a rank-1 update, see
  [Chapter 4](ch04-numerical-linear-algebra.md)).

## Learning outcomes

- Implement Newton–Raphson for a 2–3 variable system and diagnose non-convergence.
- Recognize quadratic convergence in a residual history and explain when it degrades to linear.
- Explain why a good starting point (flat start, DC angles, previous solution) matters.

## Resources

- Courses: Coursera *Numerical Methods for Engineers* (HKUST, Chasnov)
  <https://www.coursera.org/learn/numerical-methods-engineers> with free notes
  <https://www.math.hkust.edu.hk/~machas/numerical-methods.pdf>; MIT OCW 18.330.
- Free text: Driscoll & Braun, *FNC*, §4.3 Newton's method, §4.5 nonlinear systems, §4.6 quasi-Newton.
- Textbooks:
  - Burden & Faires, *Numerical Analysis* (4 syllabi): Ch. 10 (nonlinear systems, Newton, Broyden).
    **PRINT-DISABLED**. ES: *Análisis numérico* (Cengage, 10th ed. 2017; 9th ed. cited at FIUBA), **USER-UPLOAD (ES)** [`analisis-numerico-richard-burden-10ma`](https://archive.org/details/analisis-numerico-richard-burden-10ma).
  - Chapra & Canale, *Numerical Methods for Engineers* (2 syllabi): Newton–Raphson for systems, plus the
    chapters on Gauss elimination and LU. 7th ed. 2015 **BORROW**
    [`numericalmethods0000chap_w6f7`](https://archive.org/details/numericalmethods0000chap_w6f7).
    ES: *Métodos numéricos para ingenieros* (McGraw-Hill).
  - Kincaid & Cheney, *Numerical Analysis: Mathematics of Scientific Computing* (1991). **BORROW**
    [`numericalanalysi0000kinc`](https://archive.org/details/numericalanalysi0000kinc).
    ES: *Análisis numérico* (Addison-Wesley Iberoamericana, 1994).
- Documentation and tools: SciPy
  [`scipy.optimize.root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html);
  GNU Octave "Solvers" (`fsolve`) <https://docs.octave.org/latest/Solvers.html>; matplotlib
  [`semilogy`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.semilogy.html).

## Worked example

Solve $x^2 + y^2 = 4$, $e^x + y = 1$. Write $F(\mathbf{x}) = 0$ with

$$
F(x, y) = \begin{bmatrix} x^2 + y^2 - 4 \\ e^x + y - 1 \end{bmatrix},
\qquad
J(x, y) = \begin{bmatrix} 2x & 2y \\ e^x & 1 \end{bmatrix},
$$

and iterate: solve $J(\mathbf{x}_k)\, \Delta_k = -F(\mathbf{x}_k)$, then set
$\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta_k$.
Never form $J^{-1}$; solve the linear system ([Chapter 4](ch04-numerical-linear-algebra.md)).

```python
import numpy as np

def F(x):
    return np.array([x[0]**2 + x[1]**2 - 4, np.exp(x[0]) + x[1] - 1])

def J(x):
    return np.array([[2 * x[0], 2 * x[1]], [np.exp(x[0]), 1.0]])

x = np.array([1.0, -1.0])
for k in range(20):
    x = x + np.linalg.solve(J(x), -F(x))
    r = np.linalg.norm(F(x))
    print(k + 1, x, f"{r:.1e}")
    if r < 1e-12:
        break
```

From $(1, -1)$ the residual norms are about $8.6\cdot10^{-1}$, $4.0\cdot10^{-2}$, $1.0\cdot10^{-4}$, $8.4\cdot10^{-10}$:
the exponent roughly doubles each step (quadratic convergence), ending at $(1.0042, -1.7296)$. From $(-1, 1)$
Newton finds the other root, $(-1.8163, 0.8374)$. From $(0, 0)$ the Jacobian is singular and the first step
fails. In Octave, `fsolve(@(v) [v(1)^2 + v(2)^2 - 4; exp(v(1)) + v(2) - 1], [1; -1])` returns the first root.

## Lab

!!! example "Lab: a reusable Newton solver"

    Tools: Jupyter + NumPy + SciPy + matplotlib (Octave `fsolve` as an alternative for step 4).

    1. Write `newton(F, J, x0, tol=1e-10, max_iter=50, damping=1.0)` that returns the solution, a success flag
       and the history of residual norms. Handle a singular Jacobian (`np.linalg.LinAlgError`) by returning
       failure instead of crashing.
    2. Test it on the worked example from several starting points and plot the residual histories with
       `plt.semilogy`. Estimate the convergence order from $\log r_{k+1} / \log r_k$.
    3. Basins of attraction: on a 200×200 grid of starting points in $[-3, 3]^2$, record which root is reached (or
       failure) and show the map with `plt.imshow`. Where are the boundaries, and why does the line where
       $\det J = 2x - 2y e^x = 0$ matter?
    4. Compare iterations and function evaluations with `scipy.optimize.root(F, x0, method="hybr")` and
       `method="broyden1"`, or with Octave `fsolve`.
    5. Add damping (`damping=0.5`, or halve the step until the residual decreases) and rerun step 3. Which
       failures disappear, and what does damping cost near the solution?
    6. Keep the function: after [Chapter 15](../part-3-power-systems/ch15-ac-power-flow.md) you reuse it for a
       3-bus AC power flow.

## Exercises

1. By hand, apply Newton to $f(x) = x^2 - 2$ from $x_0 = 1$ for three steps and show that the error is roughly
   squared each time. *(Pen and paper)* *Answer:* $1.5$, $1.41667$, $1.414216$; errors $8.6\cdot10^{-2}$,
   $2.5\cdot10^{-3}$, $2.1\cdot10^{-6}$.
2. Newton on $f(x) = \arctan x$ converges from $x_0 = 1.3$ but diverges from $x_0 = 1.5$. Find the critical
   starting point where $x_1 = -x_0$ (it solves $2x = (1 + x^2)\arctan x$, about $1.3917$) and show that damping
   rescues the divergent case. *(Jupyter: NumPy + SciPy `brentq`)*
3. Apply Newton to $f(x) = (x - 1)^2$ from $x_0 = 0$. Show that the error only halves each step, and explain
   why a zero derivative at the root destroys quadratic convergence. *(Pen and paper)*
4. Preview of power flow: two buses joined by a lossless reactance $X$ (per unit) with $|V_1| = |V_2| = 1$
   transfer $P = \sin\delta / X$. Solve $\sin\delta - PX = 0$ for $\delta$ with Newton from $\delta_0 = 0$ for
   $PX = 0.5$, $0.99$ and $1.2$. Describe what happens in the last case and connect it with ToOp reporting
   non-converging load flows as a metric. *(Jupyter: NumPy + matplotlib)*
5. Broyden's method replaces $J$ by an approximation updated with a rank-1 correction each step. Implement
   the "good Broyden" update, keep the inverse up to date with Sherman–Morrison, and compare iteration counts
   with Newton on the worked example. *(Jupyter: NumPy)*
6. Solve the worked example with Octave `fsolve`, requesting the output structure
   (`[x, fval, info, output] = fsolve(...)`), and compare `output.iterations` with your Newton code.
   *(Octave)*
