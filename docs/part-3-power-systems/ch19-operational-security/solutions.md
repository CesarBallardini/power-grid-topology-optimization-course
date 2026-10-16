# Chapter 19 – Solutions

Solutions of the Lab and the Exercises of [Chapter 19](index.md) that have one. Try each task before reading its
solution. The code is in the chapter's `src` folder: [`case7.py`](src/case7.py) (SciPy and pandapower) and
[`case7_toop.py`](src/case7_toop.py) (ToOp). The numbers were produced with ToOp `ba1874e` on CPU. `python case7.py` prints every number of the Lab answers and of Exercise 10 (a) and (c); `python case7_toop.py <folder>` prints those of Exercises 7–10 (b).

## Lab, Part C: the price of security on the 7-bus case

1. **Binding constraints.** `binding(dispatch(range(11)).x)` returns two pairs: line 1–3 after the outage of
   line 1–2, and line 1–2 after the outage of line 1–3. Both lines carry exactly 240 MW. The cheap generator at
   bus 1 (capped at 250 MW) has only these two lines to leave the bus, so when one trips the other carries all of
   it, and the SCOPF holds it to the limit by moving 10 MW to the generator at bus 2.
2. **AC security.** The AC N-1 of the SCOPF dispatch peaks at 109.7 % (114.7 % for the plain OPF). Deriating the
   limits of the overloaded lines to (worst DC flow) × 100 / (worst AC loading) and solving again gives a SCOPF
   cost of $4,400/h, then $4,612/h, then $4,672/h, at which point the AC worst loading is at most 100 %
   (`ac_secure_dispatch()`). AC security costs $272/h more than DC security and $372/h more than the plain OPF.
3. **Line 1–3 limited to 150 MW.** The OPF stays at $4,300/h: its base-case flow on line 1–3 is 133 MW, below the
   new limit, and an OPF never looks at outages. The SCOPF costs $5,360/h with dispatch (146.3, 192.2, 1.5) MW.
   The outage of line 1–2 sends all of the generator at bus 1 through line 1–3, so that generator must stay
   below about 150 MW. Instead of 10 MW, about 104 MW move to the $20 generator at bus 2, which then reaches its
   own network limits (binding: line 1–3 after the outage of line 2–3, line 2–3 after the outage of line 2–5),
   and 1.5 MW come from the $35 generator at bus 7. The gap grows because the security constraint, not the
   price difference, sets the quantity moved. A remedial topology action does not help here for the same reason
   as in [Exercise 9](#exercise-9): the constraint is the cut at bus 1, and no split downstream changes it. Redispatch is the
   remedy.
4. **PowerWorld.** Expect the same dispatch and cost for a linear (DC) OPF and SCOPF: (250, 90, 0) MW at
   $4,300/h and (240, 100, 0) MW at $4,400/h. An AC OPF will cost a little more, because it includes losses and
   reactive flow; that is the same gap as in step 2. Differences in the limit handling (percent of MVA rating
   versus MW) show up as small dispatch changes.
5. **ToOp.** See [Exercise 7](#exercise-7) to [Exercise 10](#exercise-10).

## Exercise 7


`write_case` in the script writes the folder; the pass-through function returns
`(lf.n_0_matrix, lf.n_1_matrix)` and is defined once, at module level. `run_initial_loadflow` returns
`(overload_energy_n_0, overload_energy_n_1) = (0.0, 20.0)`, and the unsplit N-1 matrix gives, per outage,
the worst loaded line:

| Outage | Worst line | Loading |
|---|---|---|
| L1-2 | L1-3 | 104.2 % |
| L1-3 | L1-2 | 104.2 % |
| L2-3 | L1-3 | 71.8 % |
| L2-5 | L2-3 | 85.7 % |
| every other line | L1-3 or L2-5 | below 67 % |

The SciPy code gives the same 104.2 % (`n1_loading(dispatch().x).max()`). The two agree to the decimal
because both solve the same lossless DC equations, and the LODF update is exact, not an approximation.
The 20 MW follow from the definition: for each branch the worst contingency, then the sum over branches.
L1-3 carries 250 MW after the L1-2 outage (10 MW above its 240 MW), L1-2 carries 250 MW after the L1-3
outage (another 10 MW), and no other branch exceeds its limit in any contingency: 10 + 10 = 20 MW. Under
the "worst contingency per timestep" wording of `metrics.md` (see "Documentation vs code" above) the value
would be 10 MW; this case shows that the code sums per branch.

## Exercise 8


Choosing the set $B$ of branches that go to the second busbar with $2 \le |B| \le d - 2$ gives
$2^d - 2 - 2d$ configurations; each split appears twice ($B$ and its complement), so a substation with $d$
branches has $2^{d-1} - d - 1$ splits: 3 for $d = 4$, 10 for $d = 5$, 25 for $d = 6$ and none for $d = 3$.
Buses 3, 5 and 6 have four lines each, bus 4 has three (L3-4, L4-5, L4-6), so the prediction is
$3 \times (3 + 1) = 12$ actions. ToOp reports `n_actions_per_sub = [4, 4, 4]` for buses 3, 5 and 6, and
bus 4 is not in `relevant_nodes` at all. In the code, `enumerate_branch_actions_for_sub`
(`dc_solver/preprocess/action_set.py`) asserts $d \ge 4$ (smaller substations are dropped earlier),
enumerates $2^{d-1}$ configurations (the mirror image is not generated), removes those that isolate a
single branch (`exclude_isolations`) and then those that leave fewer than two non-bridge branches on a
busbar (`filter_splits_by_bridge_lookup`; no line of this grid is a bridge, so nothing more goes). Decoded,
the twelve actions are:

| Action | Substation | Branches on a busbar of their own |
|---|---|---|
| 0, 4, 8 | bus 3, 5, 6 | none (unsplit) |
| 1, 2, 3 | bus 3 | L3-6 + L1-3; L3-4 + L1-3; L3-4 + L3-6 |
| 5, 6, 7 | bus 5 | L5-7 + L2-5; L5-6 + L2-5; L5-6 + L5-7 |
| 9, 10, 11 | bus 6 | L3-6 + L4-6; L6-7 + L4-6; L6-7 + L3-6 |

The load of a substation stays with the branches that are not moved (`inj_actions` is all `False` here).

## Exercise 9


Kirchhoff's current law at bus 1: with L1-2 out, L1-3 is the only branch left at bus 1, so it carries the
bus's net injection, 250 MW of generation and no load, whatever the impedances or the topology downstream:
250 / 240 = 104.2 %. The same holds for L1-2 after the L1-3 outage. A split at buses 3, 5 or 6 changes how
the 250 MW spread beyond bus 3, never how they leave bus 1. ToOp confirms it:

| Action | Split | Largest N-0 change | Worst N-1 loading |
|---|---|---|---|
| 0, 4, 8 | unsplit | 0 MW | 104.2 % (L1-3 after L1-2 out) |
| 1 | bus 3: L3-6 + L1-3 | 44 MW | 125.0 % (L3-6 after L1-2 out) |
| 2 | bus 3: L3-4 + L1-3 | 31 MW | 125.0 % (L3-4 after L1-2 out) |
| 3 | bus 3: L3-4 + L3-6 | 137 MW | islanded (solver flag `False`, `nan`) |
| 5 | bus 5: L5-7 + L2-5 | 66 MW | 126.5 % (L2-3 after L1-3 out) |
| 6 | bus 5: L5-6 + L2-5 | 73 MW | 109.3 % (L2-3 after L1-3 out) |
| 7 | bus 5: L5-6 + L5-7 | 13 MW | 104.2 % |
| 9 | bus 6: L3-6 + L4-6 | 38 MW | 104.2 % |
| 10 | bus 6: L6-7 + L4-6 | 1 MW | 104.2 % |
| 11 | bus 6: L6-7 + L3-6 | 35 MW | 104.2 % |

No action gets below 104.2 %, four add a second overload, and action 3 turns line 2–5 into a bridge: the
busbar with L3-4 and L3-6 has no injection and no other link to buses 1 and 2, so already in the base case
L2-5 carries the whole 260 MW of load of buses 4–6 (108 % of its limit), and its outage islands buses 4–7
without generation. The `action_set_filter_bsdf_lodf` option, switched off in `write_case`, removes such
actions from the set ([Chapter 22](../ch22-import-preprocessing/index.md)). The only remedy is redispatch: the SCOPF
moves 10 MW from the $10 generator at bus 1 to the $20 one at bus 2, the $100/h of the worked example. In
general, a bus split redistributes flow among parallel paths; it cannot change the flow through a cut whose
remaining branches all lie on one side of a fixed injection.

## Exercise 10


(a) The old SCOPF dispatch loads L3-6 to 111.5 % after the L2-5 outage. The SCOPF with the tighter limit
costs $4,739/h with dispatch (240, 77.4, 22.6): the $35 generator at bus 7 must run, and the binding
constraints are now (L1-2 out, L1-3), (L1-3 out, L1-2) and (L2-5 out, L3-6). That is $339/h more than the
original SCOPF, the redispatch an operator would like to avoid.

(b) ToOp: the unsplit grid has `overload_energy_n_1 = 11.5 MW` (111.5 MW on a 100 MW line). Two of the
twelve actions clear every N-1 overload, both at bus 6: action 11, L3-6 + L6-7 on their own busbar, brings
L3-6 down to 60.2 MW, and action 9, L3-6 + L4-6, to 76.8 MW; the worst loading is then exactly 100.0 %, on
the lines 1–2 and 1–3 that the SCOPF had left at their limits. The third bus-6 split (L6-7 + L4-6) leaves
109.0 %, and every bus-3 and bus-5 split is worse than doing nothing (104.7 % to 240 %, or islanding).

(c) `split_bus(pandapower_net(pg, lim), 6, ['L3-6', 'L6-7'])` followed by `pandapower_n1` gives the same
60.2 MW on L3-6 and a worst DC loading of 100.0 %.

(d) After the L2-5 outage the 260 MW of load at buses 4, 5 and 6 must all come from bus 3 through the cut
{L3-4, L3-6}. The total through the cut is fixed by KCL, as in Exercise 9, but its split between the two
lines is not: it follows the impedances of the parallel paths behind the cut. Unsplit, L3-6 feeds bus 6
directly and bus 5 through L5-6 ($x = 0.14 + 0.10$). Paired with L6-7, whatever enters through L3-6 must
continue to bus 7 and reach bus 5 through L5-7 ($x = 0.14 + 0.12 + 0.15$), a much longer path than
L3-4 → L4-5 ($0.20$) or L3-4 → L4-6 ($0.22$), so most of the flow moves to L3-4 (199.8 MW of 200 MW after
the same outage). A split helps when the overloaded line has a parallel path downstream that the split
can lengthen or shorten; it is powerless when the line is the only path, as at bus 1
([Chapter 28](../../part-4-optimization/ch28-topology-optimization/index.md)). The AC check shows the price of the
thin DC margin: the split topology's worst AC N-1 loading is 116 % on L3-4, so this remedial action passes
ToOp's DC stage and fails its AC validation, which is the reason the two stages exist.
