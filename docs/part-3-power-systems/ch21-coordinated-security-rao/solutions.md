# Chapter 21 – Solutions

Solutions of the Lab and the Exercises of [Chapter 21](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`ieee14_flow_based.py`](src/ieee14_flow_based.py) (the Lab, NumPy and
pypowsybl), [`curative_timing.py`](src/curative_timing.py), [`domain_2d.py`](src/domain_2d.py),
[`glsk_sensitivity.py`](src/glsk_sensitivity.py) and [`time_coupling.py`](src/time_coupling.py) (Exercises 1, 2,
3 and 6) and [`orao_switches_toop.py`](src/orao_switches_toop.py) (Exercise 5, ToOp). Each module's `main()` prints
every number quoted below; the ToOp numbers were produced with ToOp `ba1874e` on CPU.

## Lab

`python ieee14_flow_based.py` runs the six steps on the base topology and on two topology actions. The functions
named below are in that module.

1. **IEEE 14 and its nodal PTDF.** `load_ieee14()` reads buses, lines, transformers, generators and loads from
   `pypowsybl.network.create_ieee14()` and `ptdf_matrix()` builds $H$ as in Chapter 17 with bus 1 as slack. Two
   details decide whether `H @ p` matches pypowsybl's DC flows: the per-unit base of the reactances, which pypowsybl
   gives in ohms (a line between two nominal voltages $V_1$, $V_2$ uses $V_1 V_2 / S_\text{base}$; a transformer
   uses the square of its side-2 voltage, where its ohms are expressed), and the transformer ratio. pypowsybl's
   default DC load flow keeps the off-nominal ratios $\rho = (U_{1,\text{rated}}/V_1)/(U_{2,\text{rated}}/V_2)$
   (0.978, 0.969 and 0.932 for the three transformers), which multiplies their reactance by $\rho$ in the DC
   equations; with `Parameters(distributed_slack=False, dc_use_transformer_ratio=False)` the textbook model applies
   and `H @ p` reproduces every pypowsybl flow to $2.6 \times 10^{-13}$ MW. The injection vector must be balanced:
   the data have 272.4 MW of generation set points against 259 MW of load, and in a lossless DC model the slack
   generator at bus 1 produces 219.0 MW, not its 232.4 MW set point. The net positions at the reference point are
   then $NP_A = 87.7$, $NP_B = -34.3$, $NP_C = -53.4$ MW (zones A = {1–5}, B = {6, 11, 12, 13},
   C = {7, 8, 9, 10, 14}), summing to zero.
2. **Zones and GLSKs.** `glsk(grid, 'generation')` spreads a zone's net position over its generators in proportion
   to `target_p`: zone A gets (0.853, 0.147, 0, 0, 0) over buses 1–5 (232.4 and 40 MW at buses 1 and 2; the 0 MW
   generators at buses 3, 6 and 8 get no weight). Zones B and C have no generation in the base case, so the
   generation GLSK falls back to flat there: 0.25 on each of buses 6, 11, 12, 13 and 0.2 on each of buses 7, 8, 9,
   10, 14. `glsk(grid, 'flat')` is flat in all three zones. Every column sums to one.
3. **Zonal PTDF and the pypowsybl cross-check.** $\text{PTDF}_z = H\,G$. For zone A with the generation GLSK,
   `pypowsybl_zone_factors()` builds the same zone with
   `create_zone_from_injections_and_shift_keys('A', ['B1-G', 'B2-G'], [0.853, 0.147])`, runs the DC sensitivity
   analysis with the parameters of step 1 and reads `get_sensitivity_matrix()`: the 20 factors agree with the NumPy
   column to $5.6 \times 10^{-17}$ (the zone factor is the shift-key-weighted sum of the injection factors, and the
   injection factors are the PTDF columns). A few rows, zones (A, B, C):

    | Branch | Generation GLSK | Flat GLSK |
    |---|---|---|
    | L1-2-1 | (−0.123, −0.633, −0.652) | (−0.573, −0.633, −0.652) |
    | L1-5-1 | (−0.024, −0.367, −0.348) | (−0.227, −0.367, −0.348) |
    | L2-3-1 | (0.004, −0.122, −0.138) | (−0.152, −0.122, −0.138) |
    | T5-6-1 | (−0.001, −0.600, −0.294) | (−0.007, −0.600, −0.294) |

    The B and C columns are identical because both GLSKs are flat there. The A column changes a lot: with 85 % of
    zone A's shift key on the slack bus, whose PTDF column is zero, the generation GLSK makes zone A almost
    invisible to the 135 kV lines, while the flat GLSK puts 60 % of every MW on buses 2–5 and loads L1-2 and L1-5.
    The slack choice does not matter for zone-to-zone PTDFs (differences of columns) but it does for a single
    zone's column, which is why only zone-to-zone values have a physical meaning.
4. **CNECs, $F_0$ and RAM.** `cnec_table()` makes 40 CNECs (20 branches, both directions) with
   $F_\max = \max(1.3\,|F_\text{ref}|,\ 20)$ MW (16 of them sit on the 20 MW floor), FRM $= 0.05\,F_\max$,
   $F_0 = F_\text{ref} - \text{PTDF}_z\,NP_\text{ref}$ and $\text{RAM}^\pm = F_\max - \text{FRM} \mp F_0$. The four
   smallest margins:

    | CNEC | $F_\text{ref}$ | $F_\max$ | $F_0$ | RAM |
    |---|---|---|---|---|
    | L3-4-1− | −24.2 | 31.4 | −36.1 | **−6.2** |
    | L7-8-1− | 0.0 | 20.0 | −10.7 | 8.3 |
    | L9-10-1− | 6.2 | 20.0 | −5.3 | 13.7 |
    | L10-11-1− | −2.8 | 20.0 | −3.6 | 15.4 |

    The negative RAM of L3-4-1 in the negative direction means that the point of zero net positions is outside the
    domain: with no exchange at all the line would carry 36 MW from bus 4 to bus 3, above its 31.4 MW stand-in
    limit. The market must trade (zone A exporting, as it does at the reference point) to keep this line within
    its limit. Every RAM is positive *at the reference point* by construction, since $|F_\text{ref}| \le F_\max/1.3$
    and $F_\max - \text{FRM} = 0.95\,F_\max$.
5. **The domain.** With $NP_C = -NP_A - NP_B$ each CNEC is a half-plane
   $(\text{PTDF}_A - \text{PTDF}_C)\,NP_A + (\text{PTDF}_B - \text{PTDF}_C)\,NP_B \le \text{RAM}$ (`half_planes()`).
   `domain_vertices()` intersects the boundary lines pairwise and keeps the points that satisfy all 40
   constraints: a pentagon with vertices (52.3, −76.3), (70.7, −83.1), (107.8, −47.4), (79.9, 9.5), (41.4, 23.7),
   whose edges belong to L10-11-1 (both directions), T4-7-1+, T5-6-1+ and L3-4-1− (`binding_cnecs()`). The reference
   point (87.7, −34.3) is inside. The domain is small and thin because the 12 kV branches have 20 MW limits and
   sensitivities to zones B and C of 0.25 to 0.6: ten to twenty MW of exchange fill them. `exchanges()` measures
   how far the reference point can move in each zone-to-zone direction before a CNEC binds, the flow-based
   counterpart of an available transfer capacity:

    | Direction | Extra exchange | Binding CNEC |
    |---|---|---|
    | A→B | 16.5 MW | T5-6-1+ |
    | B→A | 44.9 MW | L3-4-1− |
    | A→C | 13.7 MW | T4-7-1+ |
    | C→A | 40.0 MW | L3-4-1− |
    | B→C | 27.9 MW | T4-7-1+ |
    | C→B | 32.4 MW | T5-6-1+ |

    Exports from zone A are limited by the two transformers that feed zones B and C; imports into A are limited
    by L3-4 in its negative direction. The plot is a matplotlib polygon through the vertices with the reference
    point marked and the five binding CNECs as edge labels (not saved by the module; `lab.ipynb` leaves the figure
    to you).
6. **A topology action.** `topology_action(grid, f_max, 'L13-14-1')` opens line 13–14, keeps the limits of the
   surviving branches (limits are properties of the lines, not of the topology), and rebuilds PTDF, CNECs and
   domain. The new pentagon is bounded by L6-13-1+, T4-7-1+, T5-6-1+, L3-4-1− and L10-11-1−. Overlaying the two
   polygons and comparing the exchanges:

    | Direction | Base | L13-14 open | Change |
    |---|---|---|---|
    | A→B | 16.5 | 18.3 | +1.8 |
    | B→A | 44.9 | 41.8 | −3.0 |
    | A→C | 13.7 | 9.6 | −4.0 |
    | C→A | 40.0 | 39.8 | −0.2 |
    | B→C | 27.9 | 15.6 | −12.3 |
    | C→B | 32.4 | 25.9 | −6.5 |

    Only A→B gains: the line 13–14 was the parallel path between zones B and C, so every exchange that used it
    loses capacity, B→C most of all. The N-1 check of Chapter 19 (`security_metrics()`, LODF over every
    non-bridge outage, loading against the same $F_\max$) says: base topology worst N-0 loading 76.9 % (that is
    $1/1.3$, by construction), worst N-1 loading 300 % on L3-4-1 after the outage of L2-3-1, N-1 overload energy
    382.8 MW; with L13-14 open the N-0 worst rises to 81.1 %, the worst N-1 case is the same 300 % and the overload
    energy grows to 414.7 MW. The stand-in limits of step 4 make the base topology itself N-1 insecure, so the
    check is a comparison: the action makes N-1 security worse and would not pass. The module also tries line 3–4,
    the action that enlarges four of the six margins: it leaves no domain at all. Bus 3 becomes radial, so L2-3-1
    carries the whole 94.2 MW load of bus 3 whatever the net positions (zonal PTDFs exactly zero), above its
    91.1 MW limit: $\text{RAM} = -7.7$ MW with zero sensitivity is a constraint no exchange can satisfy. In N-0
    the line is at 103.4 %, so this action fails before any N-1 case is run (its N-1 energy, 285.4 MW, would be
    better). A remedial action is only a candidate when the base case stays within limits; the flow-based domain
    then tells the market how much room the action leaves.

## Exercise 1

The TATL is a *temporary* admissible loading: the line may carry up to 1 250 A for 15 minutes after the
contingency, provided the flow is brought back under the PATL (1 000 A) within that time. `verdict()` in
`curative_timing.py` encodes the three conditions.

- 1 180 A after the contingency, split active after 10 minutes, 940 A afterwards: the flow is under the TATL,
  the action arrives before the 15 minutes run out and leaves 940 A < 1 000 A. The curative action is
  sufficient; no preventive action is needed, and the market keeps the capacity.
- Activation takes 20 minutes: the line would stay at 1 180 A for 20 minutes, 5 minutes beyond the TATL
  duration. A preventive action (or a faster curative one) is needed to bring the post-contingency flow under
  the PATL, or at least to a level the TATL duration covers.
- Post-contingency flow 1 300 A: above the TATL, so the line is not allowed to carry it even for a second; the
  protection would trip it. Only a preventive action helps, since there is no time window in which to act.

A boundary case: 1 250 A and exactly 15 minutes pass (the TATL is inclusive in CRAC conventions); a curative
action that leaves 1 020 A does not pass, because the end state must respect the PATL.

## Exercise 2

The GeoGebra construction (not run here): five lines $a_k\,NP_A + b_k\,NP_B = \text{RAM}_k$, the inequality
side of each shaded, the intersection filled, a slider on one RAM. `domain_2d.py` holds one such set, with the
zonal PTDFs given relative to the balancing zone C (so $a_k = \text{PTDF}_{A,k}$, $b_k = \text{PTDF}_{B,k}$):

| CNEC | $\text{PTDF}_A$ | $\text{PTDF}_B$ | RAM (MW) | $s = \text{PTDF}_A - \text{PTDF}_B$ | RAM / $s$ |
|---|---|---|---|---|---|
| 1 | 0.5 | 0.1 | 200 | 0.4 | 500 |
| 2 | −0.3 | 0.4 | 180 | −0.7 | relieved |
| 3 | −0.4 | −0.5 | 250 | 0.1 | 2 500 |
| 4 | 0.2 | −0.6 | 220 | 0.8 | 275 |
| 5 | 0.6 | −0.4 | 150 | 1.0 | **150** |

An A→B exchange of $e$ MW is the point $NP = (e, -e, 0)$; CNEC $k$ sees $s_k e$ with $s_k = a_k - b_k$ and allows
$e \le \text{RAM}_k / s_k$ when $s_k > 0$ (a CNEC with $s_k < 0$ is relieved). The largest exchange is 150 MW,
bound by CNEC 5. The domain itself is the pentagon with vertices (−612.9, −9.7), (−117.6, −405.9), (7.1, −364.3),
(365.4, 173.1) and (269.6, 652.2); it contains the origin. Moving the slider: with RAM$_5$ = 200 MW the exchange
is 200 MW, still bound by CNEC 5; from RAM$_5$ = 275 MW on, CNEC 4 takes over at 275 MW and raising RAM$_5$
further (300, 400 MW) changes nothing. That is the general picture: enlarging the margin of the binding CNEC moves
the boundary until the next CNEC binds, and a remedial action that only relieves one element has a bounded payoff.

## Exercise 3

`glsk_sensitivity.compare()` rebuilds the worked example with the zone A shift keys (0.6, 0.4), (1.0, 0.0) and
(0.0, 1.0):

| GLSK A | Branch | $F_\text{ref}$ | $\text{PTDF}_{A-B}$ | $F_0$ | RAM |
|---|---|---|---|---|---|
| (0.6, 0.4) | 0–1 / 0–2 / 1–3 / 2–3 / 1–2 | 137.5 / 162.5 / 62.5 / 37.5 / 25.0 | 0.188 / 0.412 / 0.362 / 0.138 / 0.225 | 90.6 / 59.4 / −28.1 / 3.1 / −31.2 | 99.4 / 130.6 / 218.1 / 186.9 / 221.2 |
| (1.0, 0.0) | same | same | 0.438 / 0.562 / 0.312 / 0.188 / 0.125 | 28.1 / 21.9 / −15.6 / −9.4 / −6.2 | 161.9 / 168.1 / 205.6 / 199.4 / 196.2 |
| (0.0, 1.0) | same | same | −0.188 / 0.188 / 0.438 / 0.062 / 0.375 | 184.4 / 115.6 / −46.9 / 21.9 / −68.8 | 5.6 / 74.4 / 236.9 / 168.1 / 258.8 |

The reference flows are the same column three times: they come from the nodal injections, $F_\text{ref} = H\,p$,
and the GLSK plays no part in them. The zonal PTDFs, $F_0 = F_\text{ref} - \text{PTDF}_z\,NP_\text{ref}$ and the
margins all change, and so does the capacity the market sees: 317 MW of extra A→B exchange (branch 0–2 binding)
with (0.6, 0.4), 299 MW with (1.0, 0.0), 396 MW with (0.0, 1.0). Note the change of sign of the zone-to-zone PTDF of
branch 0–1 when zone A's export is assumed to come from bus 1: the same exchange then relieves the branch that the
other GLSKs load. The GLSK is a forecast of *where* in the zone the generators will react to the market; if the
real dispatch differs, the flow on each CNEC differs from the forecast by
$(\text{PTDF}_{\text{true}} - \text{PTDF}_{\text{GLSK}})\,\Delta NP$. This forecast error, together with load,
renewable and topology forecast errors between D-2 and real time, is what the flow reliability margin covers
(CACM Art. 22); the FRM is estimated statistically from the observed differences between forecast and realized
flows on each CNEC.

## Exercise 4

| ToOp artifact | CRAC element | What ToOp lacks |
|---|---|---|
| `nminus1_definition.json`, `contingencies` (`Contingency(id, elements=[GridElement])`, plus `BASECASE`) | Contingencies: a named list of network elements that trip together | Nothing structural; `BASECASE` is the preventive state. The `line_for_nminus1` mask (and `trafo_*`, `generator_for_nminus1` etc.) is the list of elements turned into single contingencies |
| `monitored_elements` and the `line_for_reward` mask | The network elements of the CNECs | A CNEC in a CRAC is one element *in one state* (after one contingency, at one instant); ToOp monitors every reward element after every contingency, the full product |
| `branch_limits.max_mw_flow` (N-0) and `max_mw_flow_n_1` | Thresholds of the CNECs: PATL for the preventive instant, TATL for the outage instant | Only MW thresholds (a CRAC also takes A and % of $I_\max$), only two instants: preventive (N-0 flows against `max_mw_flow`) and outage (N-1 flows against `max_mw_flow_n_1`). No auto and curative instants, no TATL durations |
| `action_set.json` `local_actions` (busbar assignments) and `disconnectable_branches` | Network actions: switch pairs (open/close) and terminal connections | All are preventive; the ORAO summary writes them under `preventive-actions-list` |
| the fact that every action may be applied | Usage rules | Only "available everywhere, preventive". No `onContingencyState`, `onFlowConstraint`, `onInstant` rules, so no action can be reserved for one contingency or triggered by one CNEC |

Three CRAC features with no representation in ToOp: (1) curative and automatic remedial actions, with their
instants, TATL durations and the flow-based idea that a curative action restores the PATL after the TATL window
(Exercise 1); (2) range actions, PST tap and HVDC set-point optimization as part of the remedial action search
(ToOp's `pst_ranges` are only read by the DC solver as fixed data, and it has no redispatching or
countertrading, hence no costs); (3) CNEC-specific usage rules and thresholds (per-instant, per-contingency
limits in A or %, voltage and angle CNECs, loop-flow and RAM-based objectives). Smaller gaps: a CRAC names
contingencies with several elements and busbar outages explicitly, while ToOp's multi-outages and busbar outages
are preprocessing options.

## Exercise 5

`orao_switches_toop.py` imports and preprocesses `data/grid_node_breaker/grid.xiidm` exactly as the Chapter 22
Lab does (cutoff voltage 10 kV; one relevant substation, VL2, survives preprocessing with ten actions, the unsplit
one included). Two findings change the exercise as stated:

- `run_pipeline` (the cell of `example2_small_grid_toop.ipynb`) fails on Windows outside the container: the DC
  optimizer names its TensorBoard folder after `datetime.now()`, and the colons of the time are not valid in a
  Windows path (`WinError 123`). So the DC optimization stage was **not run here**. Instead the module writes a
  `res.json` that lists every single action as a candidate and calls `perform_ac_analysis`, the same function the
  notebook runs after the DC stage, so that the `orao_summary.json` files come from ToOp's own writer.
- No topology of this grid is accepted. ToOp's `evaluate_acceptance` requires at most as many non-converging
  load flows as the unsplit grid and at least 5 % less AC `overload_energy_n_1`. The unsplit grid has an AC
  overload energy of 16.34 MW with 3 non-converging cases; the best split (action 5, 25.38 MW) is worse, the
  unsplit action 0 is rejected for lack of improvement, and a scan of the 80 split-plus-one-disconnection and 8
  disconnection-only candidates (run once, not kept as a test) finds every one rejected on convergence or
  overload energy. The exercise's "accepted topology" does not exist for this grid; `perform_ac_analysis` writes
  the ORAO summary of every assessed topology anyway, so the table below is that of the best split.

The ORAO summary is a dictionary `forced-actions → preventive-actions-list → actions`, each action
`{"type": "SWITCH", "id": "Open VL2_BREAKER", "switchId": "VL2_BREAKER", "open": true}`. Joined with
`net.get_switches()` (`switch_table()`) and with the bays of the action set (`bay_breakers()`,
`flag_disconnectors()`):

| Switch | Operation | Kind | Voltage level | Bay breaker | Needs breaker |
|---|---|---|---|---|---|
| VL2_BREAKER | open | BREAKER (coupler) | VL2 | – | no |
| L12_DISCONNECTOR_4_0 | open | DISCONNECTOR | VL2 | L12_BREAKER | **yes** |
| L12_DISCONNECTOR_4_2 | close | DISCONNECTOR | VL2 | L12_BREAKER | **yes** |

Action 5 moves line L1's bay from busbar BBS2_1 to BBS2_3 and opens the coupler between them. The list contains
the two disconnector operations of the bay but not its breaker: a disconnector cannot make or break load current
(Chapter 20), so the switching sequence must open `L12_BREAKER` first, then open the busbar disconnector 4_0,
close 4_2 and close the breaker again; the coupler `VL2_BREAKER`, a breaker, can be opened under load as the last
step. The ORAO list is a target state, not a sequence: the ordering and the bay breakers are what the operator or
OpenRAO's switch sequence has to add. (ToOp's logs call the result "Accepted Topology" while printing the
rejection reason; the verdict is the return value of `evaluate_acceptance`, `None` when accepted.)

## Exercise 6

`time_coupling.py` builds a 24 × 4 table of hourly overload energies (a daily load shape with topology 0 the
unsplit grid, topology 1 best in the morning, 2 best in the evening, 3 mediocre everywhere; `overload_table()`
prints it). (a) The single topology with the smallest total, the way ToOp scores one topology on all timesteps of a
run, is topology 1 with 262.5 MW. (b) The dynamic program `best_sequence(table, c)` minimizes
$\sum_h E[h, t_h] + c \cdot \#\{h : t_h \ne t_{h-1}\}$ over sequences: for each hour and topology, the best cost of
ending there is its own overload plus the cheaper of staying in the same topology or coming from the best previous
one plus $c$; a back-pointer array recovers the sequence (`test_time_coupling.py` checks it against brute force on
a six-hour day).

| Switching cost $c$ | Sequence | Overload | Switches | Total cost |
|---|---|---|---|---|
| 0 | 2 at hour 0, then 1 for hours 1–12, then 2 | 196.5 | 2 | 196.5 |
| 5 | 1 for hours 0–12, 2 for hours 13–23 | 199.2 | 1 | 204.2 |
| 10 | same | 199.2 | 1 | 209.2 |
| 20 | same | 199.2 | 1 | 219.2 |
| 63.5 | topology 1 all day | 262.5 | 0 | 262.5 |

The one-switch plan costs $199.2 + c$ and the fixed plan 262.5, so they coincide at $c^\star = 63.3$ MW per
change (63.5 on the 0.5 MW grid of `break_even()`): a switching cost of about a quarter of the day's overload
energy makes the single topology optimal. Below it, a planner who may switch once a day saves 63 MW of overload
energy. ToOp's choice of one topology per run is the $c \to \infty$ end of this trade-off; it is the right
choice when switching between hours is operationally expensive or unverified, and it is why a time-coupled RAO
(OpenRAO's MARMOT) needs a switching cost or a limit on the number of changes to be meaningful.
