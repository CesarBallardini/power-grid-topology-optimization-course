# Chapter 16. DC power flow

**Weeks:** 1.5 · **Semester:** 3 · **Prerequisites:** [Chapter 14](ch14-component-models.md), [Chapter 15](ch15-ac-power-flow.md)

!!! info "Why it matters for ToOp"

    The whole GPU solver is a DC power flow. `get_susceptance_matrices` builds
    $B_\text{bus} = A^\top \operatorname{diag}(b)\, A$ and $B_f = \operatorname{diag}(b)\, A$ with $b = 1/x$
    (`dc_solver/preprocess/helpers/ptdf.py`); a single slack bus is required because distributed slack cannot be
    combined with bus-split factors; an AC–DC mismatch term corrects DC flows
    (`dc_solver/jax/cross_coupler_flow.py`). The base-case DC flows are computed by `run_initial_loadflow`
    (`dc_solver/preprocess/convert_to_jax.py`) and stored as `unsplit_flow`.

## Topics

- Assumptions: flat voltage magnitudes, small angle differences, $r \ll x$, reactive power ignored.
- $P = B_\text{bus}\,\theta$, branch flows $F = B_f\,\theta$; slack elimination (revisits the grounded Laplacian
  of [Chapter 2](../part-1-math/ch02-linear-algebra.md) and the DC circuits of
  [Chapter 8](../part-2-circuits/ch08-dc-circuits.md) with susceptances instead of conductances).
- Phase shifters as equivalent injections; sign conventions.
- Accuracy of the DC approximation and when it fails; hot-start DC and AC–DC corrections.
- The time axis: one topology, many injection snapshots (see the note below).
- Tools: MATPOWER `makeBdc` and `rundcpf` in Octave; `pandapower.rundcpp`; `pypowsybl.loadflow.run_dc`.

## Learning outcomes

- Derive the DC power flow from the AC branch flow equation and state each approximation.
- Solve a DC power flow with slack elimination in NumPy or Octave and match `pandapower.rundcpp` or
  MATPOWER's `rundcpf`.
- Quantify the DC error against an AC solution per branch and relate it to loading, $r/x$ and voltage profile.

## Resources

- Papers (core readings):
  - Stott, Jardim & Alsaç, "DC Power Flow Revisited", *IEEE Trans. Power Systems* 24(3):1290–1300, 2009,
    doi:10.1109/TPWRS.2009.2021235.
  - Purchala, Meeus, Van Dommelen & Belmans, "Usefulness of DC power flow for active power flow analysis",
    IEEE PES General Meeting 2005, doi:10.1109/PES.2005.1489581.
- Free notes: Chatzivasileiadis, "Lecture Notes on Optimal Power Flow" (DTU), arXiv:1811.00943 (DC power flow
  and DC-OPF); Molzahn & Hiskens, "A Survey of Relaxations and Approximations of the Power Flow Equations",
  *Foundations and Trends in Electric Energy Systems* 4(1–2), 2019.
- Documentation: MATPOWER manual §3.7 "DC Modeling" and §4.2 "DC Power Flow"
  <https://matpower.org/docs/MATPOWER-manual.pdf>; pandapower DC power flow
  <https://pandapower.readthedocs.io/en/latest/powerflow/dc.html>.
- Textbook: Wood, Wollenberg & Sheblé 3rd ed. §6.18 "DC or linear power flow".
- ToOp: `notebooks/example1_dc_loadflow_example.ipynb`.

## Worked example

**From AC to DC.** For a branch from $f$ to $t$ with reactance $x$, tap ratio $\tau$ and phase shift
$\theta_\text{shift}$, the AC active flow is
$p_f = \frac{|V_f||V_t|}{x\tau}\sin(\theta_f - \theta_t - \theta_\text{shift})$ when $r = 0$. With
$|V| \approx 1$ and small angles (MATPOWER manual, eq. 3.23),

$$
p_f \approx b\,(\theta_f - \theta_t - \theta_\text{shift}), \qquad b = \frac{1}{x\tau}.
$$

**Matrix form.** With the branch–bus incidence matrix $A = C_f - C_t$ and $b$ the vector of branch
susceptances,

$$
F = B_f\,\theta + P_{f,\text{shift}}, \qquad B_f = \operatorname{diag}(b)\,A, \qquad
P = B_\text{bus}\,\theta + P_\text{bus,shift}, \qquad B_\text{bus} = A^\top \operatorname{diag}(b)\,A,
$$

where the shift terms are $P_{f,\text{shift}} = -\operatorname{diag}(b)\,\theta_\text{shift}$ and
$P_\text{bus,shift} = A^\top P_{f,\text{shift}}$: a phase shifter acts like a pair of opposite injections at its
terminals plus a direct flow on its own branch. $B_\text{bus}$ is singular (its rows sum to zero), so fix
$\theta_s = 0$ at the slack bus $s$, delete row and column $s$, and solve

$$
\tilde B\, \tilde\theta = \tilde P - \tilde P_\text{bus,shift}, \qquad F = B_f\,\theta + P_{f,\text{shift}}.
$$

The slack bus absorbs the imbalance $P_s = -\sum_{i \ne s} P_i$.

```python
import numpy as np

def dc_power_flow(frm, to, x, p, slack):
    n, m = len(p), len(frm)
    A = np.zeros((m, n)); A[np.arange(m), frm] = 1.0; A[np.arange(m), to] = -1.0
    b = 1.0 / x
    B = A.T @ np.diag(b) @ A
    keep = np.arange(n) != slack
    theta = np.zeros(n)
    theta[keep] = np.linalg.solve(B[np.ix_(keep, keep)], p[keep])
    return np.diag(b) @ A @ theta          # branch flows, from -> to
```

**Octave + MATPOWER.** `makeBdc` returns exactly these four objects, and `rundcpf` is a wrapper that sets the
model option to `'DC'` before calling `runpf` (manual §4.4 and §9.5.2):

```octave
define_constants;
mpc = ext2int(loadcase('case14'));
[Bbus, Bf, Pbusinj, Pfinj] = makeBdc(mpc);
results = rundcpf(mpc, mpoption('out.all', 0));
disp(results.branch(1:5, PF).')          % MW, from end
```

**Sign conventions.** ToOp's PSDF (`dc_solver/preprocess/helpers/psdf.py`) corresponds to a branch flow
$b_k(\theta_f - \theta_t + \varphi_k)$, i.e. $\varphi_k$ enters with the opposite sign of MATPOWER's
$\theta_\text{shift}$; the PowSyBl backend negates the grid model's shift angle in `get_shift_angles`
(`dc_solver/preprocess/powsybl/powsybl_backend.py`) under the comment `# TODO find out where this minus comes
from...`, and also negates PowSyBl's `p1` in `get_basecase_dc_branch_flows` because "Powsybl's p1 convention is
opposite to the solver's from-node to to-node orientation". Always check the orientation of flows and angles
before comparing two tools.

!!! note "The multi-timestep axis"

    Most ToOp arrays carry a leading time axis: `nodal_injections` has shape `(n_timesteps, n_bus)`,
    `unsplit_flow` has shape `(n_timesteps, n_branches)`, and the N-1 matrix has shape
    `(n_timesteps, n_failures, n_branches_monitored)` (`dc_solver/jax/types.py`,
    `dc_solver/jax/contingency_analysis.py`). The PTDF has no time axis: one topology is evaluated against
    several injection snapshots, and a flow is one matrix–vector product per snapshot. The metrics aggregate over
    time in different ways (`dc_solver/jax/aggregate_results.py`): overload energy takes the worst failure and then
    sums over timesteps, while the critical-branch count reports only the worst timestep. Harmonizing grid models
    of different timesteps into one joint optimization (`dc_solver/preprocess/harmonize.py`) is commented out
    "until we decide to take up the multi-timestep optimization again", and the example notebook runs with one
    timestep.

## Lab

!!! example "Lab: how wrong is the DC power flow?"

    Tools: Jupyter + NumPy + pandas + matplotlib + pandapower; ToOp notebooks. Optionally Octave + MATPOWER.

    1. Run `notebooks/example1_dc_loadflow_example.ipynb` (IEEE 57 in PowSyBl format). Inspect
       `static_information.dynamic_information.unsplit_flow` and explain its shape.
    2. On `pandapower.networks.case57()` (or `case14()`), compute DC flows with the function of the worked example,
       then run `pandapower.rundcpp` and compare with `net.res_line.p_from_mw` and `net.res_trafo.p_hv_mw`. Find
       and explain any systematic difference (per-unit base, taps, phase shifts, orientation).
    3. Run `pandapower.runpp` on the same network. Put DC and AC flows in a pandas `DataFrame` and plot the
       relative error per branch against AC loading and against $r/x$.
    4. Compare your DC flows with ToOp's `unsplit_flow` for the same grid, taking the orientation conventions of
       the worked example into account.
    5. Optional: repeat step 2 with MATPOWER's `rundcpf` on `case57` in Octave.

## Exercises

1. Starting from $p_f = \frac{|V_f||V_t|}{x}\sin(\theta_f - \theta_t)$, bound the relative error of the DC flow
   for $|\theta_f - \theta_t| \le 10^\circ$ and $|V| \in [0.95, 1.05]$ p.u.
2. Show that the flows of a DC power flow do not depend on which bus is the slack when the injections sum to
   zero, and that the voltage angles do.
3. *(Jupyter: NumPy)* Build $B_\text{bus}$ and $B_f$ for IEEE 14 from the branch table, solve the DC power flow
   with a sparse factorization (`scipy.sparse.linalg.splu`), and reuse the factorization for 1,000 random
   balanced injection vectors stacked as a `(n_timesteps, n_bus)` array. Compare the time with 1,000 separate
   solves.
4. *(Octave + MATPOWER)* For `case14`, verify that `Bbus` from `makeBdc` equals $A^\top \operatorname{diag}(b) A$
   with $b = 1/(x\tau)$, and that `Pfinj` is zero unless a branch has a nonzero `SHIFT`. Add a 3° shift to one
   transformer and check the equivalent-injection formula.
5. *(Jupyter: NumPy)* Implement ToOp's PSDF column
   $\text{PSDF}_{\ell k} = b_k(\delta_{\ell k} - \text{PTDF}_{\ell f_k} + \text{PTDF}_{\ell t_k})$
   (in radians, before ToOp's conversion to MW per degree) and confirm it against a
   DC power flow with a phase shift $\varphi_k$ modeled as equivalent injections. Then state which sign convention
   for the grid model's angle would make the minus in `get_shift_angles` necessary.
