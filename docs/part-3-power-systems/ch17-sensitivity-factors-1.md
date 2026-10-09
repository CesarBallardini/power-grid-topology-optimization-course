# Chapter 17. Sensitivity factors I: PTDF, PSDF, LODF and N-1 screening

**Weeks:** 3 · **Semester:** 3 · **Prerequisites:** [Chapter 3](../part-1-math/ch03-graph-theory.md), [Chapter 16](ch16-dc-power-flow.md)

!!! info "Why it matters for ToOp"

    These factors are how ToOp computes flows:

    - **PTDF**: `compute_ptdf` solves one sparse system for all non-slack columns,
      $H_{:,\mathcal N} = B_{f\,:,\mathcal R}\,\bigl(B_{\text{bus}\,\mathcal N,\mathcal R}\bigr)^{-1}$, with
      $\mathcal N$ the non-slack buses and $\mathcal R$ all buses except bus 0, the angle reference
      (`dc_solver/preprocess/helpers/ptdf.py`, adapted from pandapower's `makePTDF`).
    - **PSDF**: phase-shift columns appended to the PTDF, so one matrix product gives all flows
      (`dc_solver/preprocess/helpers/psdf.py`).
    - **LODF**: $\text{LODF}_{ab} = \dfrac{H_{a f_b} - H_{a t_b}}{1 - (H_{b f_b} - H_{b t_b})}$ with
      $\text{LODF}_{bb} = -1$, and N-1 flows $F^{(b)} = F^{(0)} + \text{LODF}_{:,b}\, F^{(0)}_b$
      (`dc_solver/jax/lodf.py`, `dc_solver/jax/contingency_analysis.py`). A zero denominator means branch $b$ is
      a bridge; ToOp flags the case as unsuccessful when $|1 - h_{bb}| \le 10^{-11}$.
    - **Injection outages**: a column of the PTDF times the lost power (`calc_injection_outage`).
    - **Slack and AC–DC correction**: the PTDF uses a single slack bus, and the base-case flows can be shifted
      towards AC flows by a per-branch mismatch term (see the worked example).

## Topics

- PTDF derivation from the DC model; slack dependence; single vs distributed slack.
- PSDF for phase-shifting transformers.
- LODF derivation by compensation; outage transfer distribution factors (OTDF); bridges and islanding
  (revisits [Chapter 3](../part-1-math/ch03-graph-theory.md)).
- Contingency screening with LODFs; generator and load outages.
- PTDF update after a disconnection, as the bridge to [Chapter 18](ch18-sensitivity-factors-2.md).
- Compensated PTDF.
- ToOp specifics: why the DC solver needs a single slack while its AC references use a distributed slack; the
  AC–DC mismatch correction; PowSyBl sign conventions.
- Tools: MATPOWER `makePTDF` and `makeLODF` (with the `mask_bridge` option) in Octave; NumPy.

## Learning outcomes

- Derive PTDF and LODF from $B\theta = P$ and implement them in NumPy.
- Explain why the LODF denominator is zero exactly for bridges ([Chapter 3](../part-1-math/ch03-graph-theory.md)).
- Screen N-1 contingencies for a 100-bus grid in one matrix operation and validate against brute force.
- Show which quantities depend on the slack choice (PTDF columns, injection-outage flows) and which do not
  (flows of balanced injections, LODF), and explain ToOp's AC–DC mismatch correction and its limits.

## Resources

- Textbook: Wood, Wollenberg & Sheblé 3rd ed., Ch. 7 "Power System Security": §7.3 contingency analysis,
  §7.4.1 linear sensitivity factors, **Appendix 7B "Calculation of Network Sensitivity Factors"** (7B.1 PTDF,
  7B.2 LODF, 7B.3 compensated PTDF). Verified against the publisher's table of contents. Required or
  recommended in 11 syllabi. **NOT ON IA** (3rd ed.), so use a library copy.
- Free: Overbye ECEN 615 sensitivity analysis lectures (<https://overbye.engr.tamu.edu/course-2/>), the best
  free substitute for Appendix 7B; MATPOWER manual §4.5 "Linear Shift Factors" (PTDF for single and distributed
  slack, LODF) and §9.5.4–9.5.5 (`makeLODF`, `makePTDF`) <https://matpower.org/docs/MATPOWER-manual.pdf>.
- Papers:
  - Ronellenfitsch, Timme & Witthaut, "A Dual Method for Computing Power Transfer Distribution Factors",
    *IEEE Trans. Power Systems*, doi:10.1109/TPWRS.2016.2589464. Open access: arXiv:1510.04645. Reading
    guide: §II (DC approximation and linear sensitivity) is required; §III–V (cycle space, dual method,
    example) are recommended; §VI–VII (implementation, applications) can be skimmed.
  - van Dijk, Westerbeck, Schewe, Benigni & Witthaut, "Unified algebraic deviation [derivation] of
    distribution factors in linear power flow", arXiv:2412.16164 (cited in ToOp's `jax/bsdf.py`). For this
    chapter read §III (background and notation) and §IV.1–IV.2 (branch modifications, line outages, phase
    shifters).
  - Srinivasan, Rao, Indulkar & Venkata, "On-Line Computation of Phase Shifter Distribution Factors and
    Lineload Alleviation", *IEEE Trans. PAS* 104:1656–1662, 1985, doi:10.1109/TPAS.1985.319195 (the PSDF
    reference cited in the ToOp dc_solver README).
- ToOp reference implementation in plain NumPy: `packages/dc_solver_pkg/tests/numpy_reference.py`
  (`calc_lodf`, `contingency_analysis`).

## Worked example

**PTDF.** Eliminate the slack $s$ as in [Chapter 16](ch16-dc-power-flow.md): $\tilde\theta = \tilde B^{-1}\tilde P$.
Then $F = B_f\theta = B_{f\,:,\mathcal N}\,\tilde B^{-1}\tilde P$, so

$$
H = \text{PTDF} \in \mathbb R^{n_l \times n_b}, \qquad
H_{:,\mathcal N} = B_{f\,:,\mathcal N}\,\tilde B^{-1}, \qquad H_{:,s} = 0 .
$$

$H_{\ell i}$ is the change of flow on branch $\ell$ when one MW is injected at bus $i$ and withdrawn at the slack.
ToOp's column choice ($\mathcal R$ instead of $\mathcal N$ for the angle reference) gives the same matrix,
because every row of $B_f$ and $B_\text{bus}$ sums to zero.

**Slack dependence.** For a distributed slack with non-negative weights $w$, $\mathbf 1^\top w = 1$, MATPOWER's
manual (eq. 4.37) gives $H_w = H_s\,(I - w\,\mathbf 1^\top)$. For a balanced injection change
($\mathbf 1^\top \Delta P = 0$) this yields $H_w \Delta P = H_s \Delta P$: flows do not depend on the slack. An
unbalanced change, such as losing a generator, does: with a single slack the whole loss is picked up at $s$.

**PSDF.** For a phase shifter on branch $k$ with angle $\varphi_k$ in ToOp's convention (branch flow
$b_k(\theta_{f_k} - \theta_{t_k} + \varphi_k)$), the angle acts as injections $-b_k\varphi_k$ at $f_k$ and
$+b_k\varphi_k$ at $t_k$ plus a direct flow $b_k\varphi_k$ on branch $k$:

$$
\text{PSDF}_{\ell k} = b_k\left(\delta_{\ell k} - H_{\ell f_k} + H_{\ell t_k}\right), \qquad
F = H P + \text{PSDF}\,\varphi ,
$$

which `compute_psdf` scales by `base_mva` and $\pi/180$ to give MW per degree.

**LODF by compensation.** Remove branch $k$ (from $f$ to $t$) by leaving it in the network and injecting $+x$ at $f$
and $-x$ at $t$ so that the injected power exactly equals the flow that the branch carries; then no power crosses
the cut ends. With $h_{\ell k} = H_{\ell f} - H_{\ell t}$ (the flow on $\ell$ for one MW sent from $f$ to $t$):

$$
x = F_k + h_{kk}\,x \;\Rightarrow\; x = \frac{F_k}{1 - h_{kk}}, \qquad
\Delta F_\ell = h_{\ell k}\,x = \underbrace{\frac{h_{\ell k}}{1 - h_{kk}}}_{\text{LODF}_{\ell k}} F_k ,
$$

and on the outaged branch itself $F_k + h_{kk}x - x = 0$, i.e. $\text{LODF}_{kk} = -1$. If $k$ is a bridge, every
MW sent from $f$ to $t$ must use branch $k$, so $h_{kk} = 1$ and the denominator vanishes. The same injections
change the PTDF: for any injection pattern, $\Delta F = \text{LODF}_{:,k}\, H_{k,:} P$, hence

$$
H' = H + \text{LODF}_{:,k}\, H_{k,:}, \qquad H'_{k,:} = 0 .
$$

With the matrix $\mathcal H = H\,(C_f - C_t)^\top$ of all $h_{\ell k}$, the full LODF matrix is
$\text{LODF} = \mathcal H\,\operatorname{diag}(1 - h_{kk})^{-1}$ with the diagonal set to $-1$ (MATPOWER manual,
eqs. 4.38–4.39). Because $\mathbf 1^\top (e_f - e_t) = 0$, the LODF does not depend on the slack either.

```python
import numpy as np

def lodf_column(H, frm, to, k):
    h = H[:, frm[k]] - H[:, to[k]]
    den = 1.0 - h[k]
    if abs(den) < 1e-11:                   # bridge: islanding
        raise ValueError("outage splits the grid")
    lodf = h / den
    lodf[k] = -1.0
    return lodf

# N-1 flows for all outages at once, F0 = H @ P (shape n_branches)
# N1[k, :] = F0 + lodf_column(H, frm, to, k) * F0[k]
```

**Octave + MATPOWER.**

```octave
mpc = ext2int(loadcase('case14'));
H  = makePTDF(mpc);              % single slack: the reference bus
L  = makeLODF(mpc, H, true);     % mask_bridge = true: NaN columns for bridges
nb = size(mpc.bus, 1);
w  = ones(nb, 1) / nb;           % a distributed slack
Hw = makePTDF(mpc, w);
disp(max(max(abs(makeLODF(mpc, Hw) - makeLODF(mpc, H)))))   % LODF is slack independent
```

### Slack choice in ToOp

- The DC solver uses a single slack bus. `compute_ptdf` takes `slack_bus` with the docstring "Cannot be
  distributed for the bsdf formulation to work" (`dc_solver/preprocess/helpers/ptdf.py`), and `calc_bsdf`
  (`dc_solver/jax/bsdf.py`) treats the case where the split station is the slack separately. The PowSyBl
  backend reads the slack from the network's `slackTerminal` extension.
- The AC references use a distributed slack: the example grids are prepared with `CGMES_DISTRIBUTED_SLACK`
  (`dc_solver/example_grids.py`), and the PowSyBl backend runs its AC and DC load flows with it by default
  ([Chapter 15](ch15-ac-power-flow.md)).
- The busbar-outage tests compare the DC solver with `pypowsybl.loadflow.run_dc(..., parameters=SINGLE_SLACK)`
  (`packages/dc_solver_pkg/tests/jax/test_busbar_outage.py`), the configuration that matches the solver's
  model.
- Consequence: base-case flows, LODFs and balanced reassignments agree between single and distributed slack,
  but injection outages do not, since ToOp's `calc_injection_outage` adds $H_{:,n}\,\Delta P$ and so assigns the
  whole loss to the slack bus.

### AC–DC mismatch correction in ToOp

The base-case flows stored in `unsplit_flow` are (`get_unsplit_flows` in `dc_solver/jax/cross_coupler_flow.py`)

$$
F^{(0)}_t = H P_t + \alpha\, m_t, \qquad m = F^\text{AC} - F^\text{DC}, \qquad \alpha \in [0, 1],
$$

with $\alpha$ = `ac_dc_interpolation` (default `0.0`, "whether to use the DC loadflow as the base loadflow (0) or
the AC loadflow (1)", `interfaces/messages/preprocess/preprocess_commands.py`). The mismatch is computed once,
for the base topology:

- PowSyBl backend, `get_ac_dc_mismatch` (`dc_solver/preprocess/powsybl/powsybl_backend.py`): the difference of
  AC and DC `p1` per branch, with the sign flipped because "powsybl has a different sign convention for the
  power flow"; zeros if the initial AC load flow did not converge.
- pandapower backend, `get_ac_dc_mismatch` (`dc_solver/preprocess/pandapower/pandapower_backend.py`): `runpp`
  minus `rundcpp` flows, repeated for every timestep (a `TODO` notes it should be recomputed per timestep); zeros
  if `runpp` does not converge.

Every topology starts from $F^{(0)}$, and the bus-split, disconnection and N-1 updates of
[Chapter 18](ch18-sensitivity-factors-2.md) act linearly on it. The offset is therefore never recomputed for the
new topology: after an outage of branch $k$ the correction on branch $\ell$ is simply
$\alpha(m_\ell + \text{LODF}_{\ell k} m_k)$. The correction is exact only for the base topology.

## Lab

!!! example "Lab: PTDF, PSDF and LODF against brute force"

    Tools: Jupyter + NumPy + pandapower; optionally Octave + MATPOWER.

    1. Implement PTDF, PSDF and LODF in NumPy for IEEE 14 and 118 (branch data from pandapower's `ppc`, or
       MATPOWER case files).
    2. Validate N-1 flows against brute-force DC power flows with each branch removed, and against
       `packages/dc_solver_pkg/tests/numpy_reference.py` (`contingency_analysis`).
    3. Show that your LODF fails for the bridges you found in [Chapter 3](../part-1-math/ch03-graph-theory.md),
       and compare with MATPOWER's `makeLODF(mpc, PTDF, true)` or pandapower's
       `pandapower.pypower.makeLODF.makeLODF`.
    4. Recompute the PTDF with a distributed slack. Confirm that base-case flows and LODFs are unchanged and
       measure how much the flows after a generator outage change.
    5. Apply the PTDF update after a disconnection and check it against the PTDF of the network without the
       branch.

## Exercises

1. For a three-bus triangle with equal reactances and slack at bus 3, derive the PTDF by hand, then the LODF
   matrix, and check $\text{LODF}_{\ell k}$ against the two-path current divider.
2. Prove that $h_{kk} = 1$ if and only if branch $k$ is a bridge, using the fact that $\tilde B$ is a grounded
   Laplacian.
3. *(Jupyter: NumPy)* Screen all N-1 branch outages of IEEE 118 with one broadcasted expression of shape
   `(n_outages, n_branches)`, and time it against 186 separate DC power flows.
4. *(Octave + MATPOWER)* For `case30`, compute `makePTDF` for a single slack and for a slack distributed in
   proportion to generator outputs. Which columns change most, and why do the flows of the base case not change?
5. *(Jupyter: NumPy + matplotlib)* Model a generator outage at bus $n$ with a single slack and with a
   distributed slack. Plot the difference in post-outage flows per branch, and relate it to the AC–DC mismatch
   that ToOp would see for injection outages.
6. Using the worked example, write the corrected flow on branch $\ell$ after a double outage $\{k_1, k_2\}$ when
   $\alpha = 1$, and explain in one paragraph when the constant-offset assumption is acceptable.
