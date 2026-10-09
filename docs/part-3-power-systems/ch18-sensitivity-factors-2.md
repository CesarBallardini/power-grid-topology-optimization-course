# Chapter 18. Sensitivity factors II: low-rank updates for topology changes

**Weeks:** 3.5 · **Semester:** 3 · **Prerequisites:** [Chapter 4](../part-1-math/ch04-numerical-linear-algebra.md), [Chapter 13](ch13-substations-grid-models.md), [Chapter 17](ch17-sensitivity-factors-1.md)

!!! info "Why it matters for ToOp"

    This is the heart of the DC solver. It never refactorizes $B$ for a new topology; it updates the PTDF on
    the GPU with low-rank corrections:

    - **Bus split (BSDF)**: a rank-1 update $H' = H + \beta\, p_c$ per split station (`dc_solver/jax/bsdf.py`), and
      the matching flow update $F' = F + c\,\beta$ driven by the cross-coupler flow $c$
      (`dc_solver/jax/cross_coupler_flow.py`).
    - **Disconnections and multi-outages (MODF)**: solve a $k \times k$ system $(I - \mathcal H_{OO})$ for $k$
      branches at once (`dc_solver/jax/disconnections.py`, `dc_solver/jax/multi_outages.py`).
    - **Susceptance changes (PST taps)**: a Woodbury correction with per-tap susceptance tables
      (`dc_solver/jax/branch_parameter_changes.py`, `dc_solver/jax/pst.py`).
    - **Injections**: reassignment to bus B, injection outages and busbar outages as PTDF columns times
      $\Delta P$ (`dc_solver/jax/injections.py`, `dc_solver/jax/busbar_outage.py`).
    - **Composition order** (`dc_solver/jax/compute_batch.py`): bus splits → disconnections → PST susceptance
      update → LODF and MODF matrices; then nodal injections and cross-coupler flows → N-0 update for
      disconnections (and PST angles) → contingency matrix (branch N-1, multi-outages, injection outages,
      busbar outages if enabled as N-1). Details in the worked example.

    JAX itself is introduced in [Chapter 31](../part-5-computing/ch31-jax-gpu-fundamentals.md), taught between
    Chapter 17 and this chapter; how ToOp batches these updates on the GPU is the subject of
    [Chapter 32](../part-5-computing/ch32-jax-in-toop.md).

## Topics

- LODF as a Sherman–Morrison update; PTDF update after a topological disconnection (revisits the rank-1
  updates of [Chapter 4](../part-1-math/ch04-numerical-linear-algebra.md)).
- Generalized / multiple-outage distribution factors (GLODF, MODF): $k \times k$ solves; closed-form
  $2 \times 2$ and $3 \times 3$ inverses with determinant checks.
- Branch parameter changes via Woodbury; MODF as the special case "susceptance goes to zero"; PST tap tables.
- **Bus split distribution factors**: splitting a node, the coupler PTDF row, PEDF, rewiring from/to vectors,
  cross-coupler flow as a KCL imbalance, and the "inject the coupler flow" view.
- Injection reassignment between bus A and bus B; injection outages after reassignment.
- Busbar outages as injection removal plus multi-outages, with a retry that keeps one skeleton branch
  (`docs/dc_solver/busbar_outage.md`).
- Composition of updates; islanding detection (zero denominators and determinants) and numerical tolerances.
- Extensions: voltage-sensitive distribution factors.

## Learning outcomes

- Derive the BSDF update from the outage of a zero-impedance coupler, implement it in NumPy, and verify it
  against a PTDF recomputed on the physically split network.
- Derive the MODF and the Woodbury susceptance update from the Sherman–Morrison–Woodbury identity and show
  that MODF is the case $\alpha = -1$.
- Reproduce ToOp's composition order for a bus split, a disconnection, an injection reassignment and an N-1
  screen, and explain why each step needs the results of the previous ones.
- Explain the cost of each update type as a function of the number of splits and outages, and why it fits
  GPU batching.

## Resources

- Papers (core readings, all with free versions):
  - Westerbeck, van Dijk, Viebahn, Merz & Witthaut, "Accelerated DC loadflow solver for topology
    optimization", IEEE PowerTech 2025, doi:10.1109/PowerTech59965.2025.11180422. Open access:
    arXiv:2501.17529. **The ToOp solver paper.**
  - van Dijk, Viebahn, Cijsouw & van Casteren, "Bus Split Distribution Factors", *IEEE Trans. Power Systems*
    39(3):5115–5125, 2024, doi:10.1109/TPWRS.2023.3331896. Preprint:
    <https://www.techrxiv.org/users/689474/articles/681212-bus-split-distribution-factors>
    (doi:10.36227/techrxiv.22298950.v1); TechRxiv blocks many automated clients, archived copy of June 2024:
    <https://web.archive.org/web/20240614093230/https://www.techrxiv.org/users/689474/articles/681212-bus-split-distribution-factors>.
  - Ronellenfitsch, Manik, Hörsch, Brown & Witthaut, "Dual theory of transmission line outages", *IEEE Trans.
    Power Systems*, 2017, doi:10.1109/TPWRS.2017.2658022. Open access: arXiv:1606.07276 (the MODF formulation
    cited in the ToOp dc_solver README).
  - Güler, Gross & Liu, "Generalized Line Outage Distribution Factors", *IEEE Trans. Power Systems*
    22(2):879–881, 2007, doi:10.1109/TPWRS.2006.888950.
  - Guo, Fu, Li & Shahidehpour, "Direct Calculation of Line Outage Distribution Factors", *IEEE Trans. Power
    Systems* 24(3):1633–1634, 2009, doi:10.1109/TPWRS.2009.2023273. `dc_solver/jax/multi_outages.py` cites its
    "equation 7" for the MODF PTDF update.
  - van Dijk et al., arXiv:2412.16164 ([Chapter 17](ch17-sensitivity-factors-1.md)).
  - Titz, Witthaut, van Dijk, Petrick & Westerbeck, "Voltage-sensitive distribution factors for contingency
    analysis and topology optimization", arXiv:2509.19976 (extension).
- Background: Hager, "Updating the Inverse of a Matrix", *SIAM Review* 31(2), 1989,
  doi:10.1137/1031049; Higham on Sherman–Morrison–Woodbury ([Chapter 4](../part-1-math/ch04-numerical-linear-algebra.md)).
- ToOp: `docs/dc_solver/busbar_outage.md`; tests that double as executable specifications:
  `packages/dc_solver_pkg/tests/jax/test_cross_coupler_flow.py` (cross-coupler flows against PTDF flows of the
  split grid), `packages/dc_solver_pkg/tests/jax/test_busbar_outage.py` (busbar outages against
  `pypowsybl.loadflow.run_dc` with `SINGLE_SLACK`), `packages/dc_solver_pkg/tests/jax/test_disconnections.py`,
  `packages/dc_solver_pkg/tests/jax/test_multi_outages.py`, `packages/dc_solver_pkg/tests/jax/test_pst.py`;
  NumPy reference `packages/dc_solver_pkg/tests/numpy_reference.py`.

## Reading guide

The papers are graduate level. For third-semester students, read them in this order and depth.

| Paper | Required | Recommended | Skippable on first reading |
|---|---|---|---|
| van Dijk et al., arXiv:2412.16164 | §III background and notation; §IV.1 branch modifications and line outages; §IV.4 BSDF; §V detecting islanding | §IV.2 phase-shifting transformers; §VI.1 finite topology modifications and line outages | §II literature overview; §IV.3 and §VI.2 bus merge factors; §IV.5 alternative treatment of bus splits; §VI.3 multiple bus splits |
| Westerbeck et al., arXiv:2501.17529 (ToOp) | §II DC loadflow theory; §III-A task definition; §IV-A injection–branch differentiation; §IV-D representing multiple outages | §III-C GPU architecture; §VI discussion | §III-B CPU architecture; §IV-B, §IV-C; §V case studies (look at the figures only) |
| van Dijk et al., "Bus Split Distribution Factors" (2024) | the definition and derivation of the BSDF; map each quantity to `calc_bsdf` | numerical validation | case studies. Section numbers are not given here: the preprint host blocks automated access, so they could not be checked. |
| Ronellenfitsch et al., arXiv:1606.07276 | §II primal formulation; §V.A single and §V.B multiple line outages | §III–IV cycles, dual graph and dual flows (needed to follow §V in full); §V.C computational aspects | §VI topology of cycle flows |
| Güler et al. (2007) | the whole letter (3 pages): matrix form of the generalized LODF | — | — |
| Guo et al. (2009) | the whole letter (2 pages): direct LODF and PTDF update after multiple outages | — | — |
| Titz et al., arXiv:2509.19976 | — | — | all, unless you work on voltage-sensitive extensions, for example in a capstone project ([Chapter 35](../part-6-capstone/ch35-capstone.md)) |

## Worked example

Notation: $H$ is the PTDF ($n_l \times n_b$), $F = HP$ the flows, and for a set $O$ of branches
$\mathcal H_{:,O} = H_{:,f_O} - H_{:,t_O}$ (column $j$ is the flow response to one MW sent from $f_{O_j}$ to
$t_{O_j}$), so $\mathcal H_{OO}$ is its $k \times k$ block on the rows of $O$.

### Disconnections and multi-outages (MODF)

For one branch $k$, [Chapter 17](ch17-sensitivity-factors-1.md) gave $H' = H + \text{LODF}_{:,k}\,H_{k,:}$. For
$k$ simultaneous outages, the compensating injections solve a $k \times k$ system instead of a scalar
equation:

$$
\text{MODF} = N\,(I - \mathcal H_{OO})^{-1}, \qquad N = \mathcal H_{:,O} \text{ with rows } O \text{ replaced by } -(I - \mathcal H_{OO}),
$$

$$
F' = F + \text{MODF}\,F_O, \qquad H' = H + \text{MODF}\,H_{O,:}, \qquad F'_O = 0,\; H'_{O,:} = 0 .
$$

`build_modf_matrix` computes this with `solve_and_check_det`, which uses closed-form inverses for $k = 2, 3$ and
an LU factorization otherwise, and reports failure when the determinant is (numerically) zero, i.e. when the
outage set disconnects the grid. `update_ptdf_with_modf` applies $H + \text{MODF}\,H_{O,:}$ (the "equation 7" of
Guo et al.).

### Susceptance changes (PST taps) by Woodbury

Changing the susceptances of branches $O$ from $b_O$ to $b_O + \Delta b_O$ changes the reduced matrix by
$A_O^\top \operatorname{diag}(\Delta b_O) A_O$. The Woodbury identity

$$
(\tilde B + U D V)^{-1} = \tilde B^{-1} - \tilde B^{-1} U \left(D^{-1} + V \tilde B^{-1} U\right)^{-1} V \tilde B^{-1}
$$

with $U = A_O^\top$, $V = A_O$, $D = \operatorname{diag}(\Delta b_O)$, and the identities
$A_O \tilde B^{-1} A_O^\top = \operatorname{diag}(b_O)^{-1}\mathcal H_{OO}$ and
$A_O \tilde B^{-1} = \operatorname{diag}(b_O)^{-1} H_{O,:}$, give, with $\alpha = \Delta b_O / b_O$ elementwise and
$E_O$ the $n_l \times k$ matrix with ones at $(O_j, j)$,

$$
C = \left(I + \operatorname{diag}(\alpha)\,\mathcal H_{OO}\right)^{-1} \operatorname{diag}(\alpha)\, H_{O,:}, \qquad
H' = H + \left(E_O - \mathcal H_{:,O}\right) C .
$$

This is exactly `update_ptdf_with_branch_parameter_change`. Setting $\alpha = -1$ (susceptance to zero) gives
the MODF update. ToOp stores, for every controllable PST, a table of effective susceptances per tap
(`pst_tap_susceptance_values`) and of shift angles in degrees per tap (`pst_tap_values`) in
`NodalInjectionInformation` (`dc_solver/jax/types.py`); a tap index `j` into these tables corresponds to the
grid model's tap position `j + grid_model_low_tap`. `prepare_pst_tap_state` looks up the susceptances for the requested taps,
the Woodbury update changes the PTDF, and `write_pst_taps_to_nodal_injections` writes the angles into the PSDF
columns of the injection vector.

### Bus splits (BSDF) and the cross-coupler flow

Split station $i$ into bus A (the branches that stay on node $i$) and bus B (a new node $b$; in ToOp's extended
PTDF its column starts as a copy of column $i$, `get_extended_ptdf`). Think of the unsplit station as A and B
joined by a coupler. Its flow from A to B follows from KCL at bus A:

$$
c = \sum_{\ell \in A_\text{to}} F_\ell - \sum_{\ell \in A_\text{from}} F_\ell + P_A = p_c\, P, \qquad
p_c = \sum_{\ell \in A_\text{to}} H_{\ell,:} - \sum_{\ell \in A_\text{from}} H_{\ell,:} + e_i^\top ,
$$

where $A_\text{to}$ ($A_\text{from}$) are bus-A branches whose to-end (from-end) is at the station. (If $i$ is the
slack, ToOp instead subtracts one from every entry and sets entry $i$ to zero.) Opening the coupler is the
outage of a zero-impedance branch that carries $c$, so, as for an LODF,

$$
F' = F + c\,\beta, \qquad H' = H + \beta\, p_c ,
$$

and $\beta$ is the limit of the coupler's LODF as its impedance goes to zero. Equivalently, **inject the coupler
flow**: in the split network, sending $c$ from A to B reproduces the unsplit flows, so
$\beta = H'_{:,i} - H'_{:,b}$ is the post-split PEDF from bus A to bus B. To compute $\beta$ from the pre-split
PTDF, emulate the split network in the unsplit one: an angle difference $\Delta$ between A and B changes each
bus-A branch $m$ (other end $o_m$, susceptance $b_m$) by $b_m\Delta$, which the unsplit network sees as injections
$+b_m\Delta$ at $o_m$ and $-b_m\Delta$ at $i$. Normalizing the exchange to one MW leaving A gives

$$
\beta_\ell = \frac{\sum_{m \in A} b_m\left(H_{\ell o_m} - H_{\ell i}\right) + \sigma_\ell\, b_\ell\,[\ell \in A]}
{\sum_{m \in A} b_m - \sum_{m \in A} b_m\left(p_{c,o_m} - p_{c,b}\right)},
\qquad \sigma_\ell = \begin{cases} +1 & \ell \in A_\text{from} \\ -1 & \ell \in A_\text{to} \end{cases}
$$

term by term the `nom`, `denom` and `g_sw` of `calc_bsdf`. A zero denominator (ToOp: $|d| < 10^{-5}$) means the
split islands part of the grid. Several splits are applied one after the other; each uses the PTDF left by the
previous ones (`compute_bus_splits`), and `compute_cross_coupler_flows` reapplies $F \leftarrow F + c\,\beta$ in
the same order, recomputing $c$ from the already updated flows. `test_compute_cross_coupler_flow_against_ptdf`
checks that the result equals $H' P'$.

```python
import numpy as np

def bsdf_split(H, frm, to, b, i, nb, bus_a, P):
    """H has the extra column nb (a copy of column i); bus_a lists branch ids that stay on node i."""
    a_from = [l for l in bus_a if frm[l] == i]
    a_to = [l for l in bus_a if to[l] == i]
    other = {l: (to[l] if frm[l] == i else frm[l]) for l in bus_a}
    p_c = H[a_to].sum(0) - H[a_from].sum(0)
    p_c[i] += 1.0                                         # i is not the slack
    num = sum(b[l] * (H[:, other[l]] - H[:, i]) for l in bus_a)
    for l in bus_a:
        num[l] += (1.0 if l in a_from else -1.0) * b[l]
    den = sum(b[l] for l in bus_a) - sum(b[l] * (p_c[other[l]] - p_c[nb]) for l in bus_a)
    beta = num / den
    c = p_c @ P
    return H + np.outer(beta, p_c), H @ P + c * beta     # updated PTDF and flows
```

### Injection reassignment, injection outages and busbar outages

- **Reassignment.** Moving injection $x$ from bus A to bus B is the balanced change $\Delta P_i = -x$,
  $\Delta P_b = +x$, so $\Delta F = H'_{:,b}\,x - H'_{:,i}\,x = -\beta\,x$. `get_reassignment_deltap` returns exactly
  these node/$\Delta P$ pairs (bus A node `rel_stat_map[sub_id]`, bus B node `n_stat + sub_id`). In the batch path,
  `get_injection_vector` writes the bus-A and bus-B sums into the injection vector and the bus-A injection
  enters $c$.
- **Injection outages** use $F + H'_{:,n}\,\Delta P$ at the node where the injection sits *after* reassignment
  (`get_all_outaged_injection_nodes_after_reassignment`).
- **Busbar outages** (`perform_outage_single_busbar`): remove the busbar's injections through its PTDF column,
  then outage all its branches with the MODF: $\hat F = F - H_{:,n}\,P_\text{bb}$ and
  $F' = \hat F + \text{MODF}_O\,\hat F_O$. If the MODF fails (the outage would island the grid), the computation is retried without the first valid
  branch, which stays as a skeleton branch; if both attempts fail the flows are NaN. The set $O$ is found on the
  physical station topology first, with propagation over disconnectors and double-connected assets
  (`docs/dc_solver/busbar_outage.md`).

### Composition order in `compute_symmetric_batch`

PTDF side (`compute_bsdf_lodf_static_flows`):

1. **Bus splits**: `compute_bus_splits` (BSDF, updates PTDF and from/to nodes, stores one $\beta$ per split).
2. **Disconnections**: `apply_disconnections` builds one MODF for all disconnected branches, updates the PTDF,
   sets their from/to nodes to an invalid index, and marks their N-1 cases to be zeroed. A failure (islanding)
   makes the whole topology unsuccessful.
3. **PST susceptance update**: `update_ptdf_with_branch_parameter_change`, only when PST taps are requested.
4. **Contingency factors**: `calc_lodf_matrix` for single branch outages and `build_modf_matrices` for
   multi-outages, on the fully updated PTDF; the success flags of injection outages and (if
   `enable_bb_outages` and `bb_outage_as_nminus1`) busbar outages inherit the topology's success.

Flow side:

5. **Nodal injections** for the chosen injection assignment (`compute_injections`).
6. **Cross-coupler flows**: `compute_cross_coupler_flows`, starting from `unsplit_flow` (which contains the AC–DC
   correction of [Chapter 17](ch17-sensitivity-factors-1.md)).
7. **N-0 after disconnections**: `update_n0_flows_after_disconnections` applies the disconnection MODF.
8. **PST angles**: if taps are requested, the angles are written into the injection vector and the N-0 flows are
   recomputed as $H'P$ (`update_n0_for_pst_taps`).
9. **Contingency matrix**: `contingency_analysis_matrix` concatenates branch N-1 (LODF), multi-outages (MODF),
   injection outages and, if enabled as N-1, busbar outages; N-1 rows of disconnected branches are set to zero.
   Otherwise, with `enable_bb_outages` on, busbar outages become a penalty relative to the unsplit grid
   (`get_busbar_outage_penalty_batched`).

## Documentation vs code: known discrepancies

- **Disconnection update.** `apply_single_disconnection_lodf` (`dc_solver/jax/disconnections.py`) implements
  $H' = H + \text{LODF}_{:,b} H_{b,:}$, sets row $b$ to zero and silently keeps the old PTDF on failure via
  `jax.lax.select`. The batch path does not use it: `compute_bsdf_lodf_static_flows` calls `apply_disconnections`,
  which applies one joint MODF for all disconnections and propagates a failure to the topology's success flag.
  The single-LODF version is used as a reference in `test_disconnections.py` and `test_multi_outages.py`.
- **Bus A and bus B labels.** The docstrings of `compute_cross_coupler_flows` and `_gather_bus_a_injection` say
  that `True` in a topology or injection vector means bus A. The code (`get_bus_data` in `dc_solver/jax/bsdf.py`,
  `get_injection_per_bus` in `dc_solver/jax/injections.py`) treats `False` entries as bus A, which stays on the
  original node, and `True` entries as bus B, the new node.
- **Failed busbar outages.** The docstring of `perform_outage_single_busbar` says the flows are set to zero if
  both attempts fail; the code sets them to NaN and uses zeros only for an invalid busbar index.
- **Inconsistent islanding tolerances.** LODF: $|1 - h_{kk}| > 10^{-11}$; BSDF: $|d| \ge 10^{-5}$;
  $2 \times 2$, $3 \times 3$ and LU solves: $|\det| > 10^{-10}$; but the $1 \times 1$ case of
  `solve_and_check_det` (`dc_solver/jax/unrolled_linalg.py`) tests `a != 0` exactly. In floating point a bridge
  often gives $1 - h_{kk} \approx 10^{-16}$ rather than zero, so a single disconnection of a bridge can pass this
  check.
- **PST path limitations.** `update_n0_for_pst_taps` recomputes the N-0 flows as $H'P$, which drops the AC–DC
  mismatch offset carried by `unsplit_flow`. `prepare_pst_tap_state` uses one tap state per topology (from the
  first timestep whose taps differ from the start) for the susceptance update, while the angles stay per
  timestep.

## Lab

!!! example "Lab: split, disconnect, screen, and compare with ToOp"

    Tools: Jupyter + NumPy; ToOp's NumPy reference and JAX solver (JAX from
    [Chapter 31](../part-5-computing/ch31-jax-gpu-fundamentals.md)); pandapower for the IEEE 14 data.

    1. Implement BSDF in NumPy for a substation of IEEE 14 (worked example): split it into two busbars, update the
       PTDF, and compare with a PTDF recomputed from a network where the node is physically split. Check that
       $\beta = H'_{:,i} - H'_{:,b}$.
    2. Compute the cross-coupler flow from KCL at bus A and confirm $F + c\,\beta = H'P$. Reassign one generator to
       bus B and confirm $\Delta F = -\beta x$.
    3. Implement the MODF and the Woodbury update. Disconnect two branches jointly and change one transformer's
       susceptance by 20 %; compare both with recomputed PTDFs, and show that $\alpha = -1$ reproduces the MODF.
    4. Compose, in ToOp's order, a bus split, a disconnection and an N-1 screen (LODF plus one busbar outage), and
       compare with brute-force DC power flows, with `packages/dc_solver_pkg/tests/numpy_reference.py`
       (`run_solver`) and with ToOp's `run_initial_loadflow` for the unsplit case.
    5. Apply the same steps in a different order (disconnection before split). Which final quantities are
       invariant, and which intermediate objects (the stored $\beta$ vectors, the cross-coupler flows) change?

## Exercises

1. Write the MODF for two outaged branches with the closed-form $2 \times 2$ inverse. Show that
   $\det(I - \mathcal H_{OO}) = 0$ when the two branches form a cut set, even if neither is a bridge alone.
2. *(Jupyter: NumPy)* Verify the Woodbury update on IEEE 118 for 1, 2 and 3 simultaneous susceptance changes,
   and time it against recomputing the PTDF. At what $k$ does recomputation win?
3. Derive $\beta = H'_{:,i} - H'_{:,b}$ from the "inject the coupler flow" argument, and use it to show that the
   bus-A branches' $\beta$ entries, oriented out of bus A, sum to one.
4. *(Jupyter: NumPy + matplotlib)* For every admissible two-way split of one IEEE 14 station, plot the
   cross-coupler flow $|c|$ against the maximum post-split loading. Relate the plot to ToOp's `cross_coupler_flow`
   penalty (`docs/topology_optimizer/metrics.md`).
5. Count floating-point operations for: one BSDF update, one $k$-outage MODF, the LODF matrix for $n_f$ N-1
   cases, and the N-1 flow matrix for $T$ timesteps. Explain which of them are embarrassingly parallel over a
   batch of topologies.
6. *(Jupyter: NumPy)* Generate random meshed grids with one radial branch, compute $1 - h_{kk}$ for the radial
   branch, and report how often it is exactly zero. Discuss which of ToOp's tolerances would catch the bridge
   ([Chapter 4](../part-1-math/ch04-numerical-linear-algebra.md)).
