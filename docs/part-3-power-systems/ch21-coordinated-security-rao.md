# Chapter 21. Coordinated security analysis, capacity calculation and remedial action optimization

**Weeks:** 1.5 · **Semester:** 4 · **Prerequisites:** [Chapter 19](ch19-operational-security.md)

!!! info "Why it matters for ToOp"

    ToOp is one TSO's non-costly remedial action optimizer inside a European process chain. Its inputs and
    outputs only make sense against that chain:

    - **Inputs from the process.** A day-ahead or intraday grid model per timestep (UCTE or CGMES), areas that
      decide what may be switched, what is monitored and what is failed (`AreaSettings` in
      `interfaces/messages/preprocess/preprocess_commands.py`), limit white lists and black lists from the German
      DACF/PRDx practice (`importer/pypowsybl_import/dacf_whitelists.py`,
      `importer/pypowsybl_import/network_analysis.py`, `docs/dc_solver/quickstart.md`), and artificial limits on
      border lines and DSO transformers (`importer/pypowsybl_import/loadflow_based_current_limits.py`, see
      [Chapter 22](ch22-import-preprocessing.md)).
    - **Outputs to the process.** For each accepted topology the AC stage writes a JSON file with a
      `forced-actions` → `preventive-actions-list` of switch operations in the style of OpenRAO
      (`changing_switches_to_orao_dict` in `optimizer/ac/summary.py`, folder `orao_summary`).
      `notebooks/openrao.ipynb` is only a stub that calls pypowsybl's RAO with placeholder CRAC and GLSK files.
    - **What ToOp does not do.** It has no CNECs with instants, no curative or automatic remedial actions, no
      GLSK or zonal PTDFs, no capacity calculation, and no switching between timesteps: one topology is scored
      against all timesteps of a run (`scoring_function` in `optimizer/dc/genetic_functions/scoring_functions.py`).

    Knowing the process tells you where ToOp's result goes next, and which assumptions a coordinated security
    analysis will check again.

## Topics

- **Grid models.** Individual and common grid models (IGM, CGM); year-ahead to intraday grid models and the
  methodology for building day-ahead and intraday CGMs (SO GL Art. 64–70); CGM building (SO GL Art. 79) and
  the CGM methodology for capacity calculation (CACM Art. 17, 28). DACF and IDCF (day-ahead and intraday
  congestion forecast) as the traditional names of these grid-model files, which is why ToOp's importer
  talks about DACF lists. Exchange formats from [Chapter 13](ch13-substations-grid-models.md).
- **Coordinated operational security analysis.** SO GL Art. 72 and 74 (day-ahead, intraday and close to real-time
  analyses on CGMs), Art. 75–78 (coordination methodology, regional coordination), regional coordination
  centres (RCCs). Remedial actions in operation must be consistent with those in capacity calculation (SO GL
  Art. 20(2), CACM Art. 25); the categories of remedial actions include topology changes and PST taps (SO GL
  Art. 22).
- **Capacity calculation (CACM, Regulation (EU) 2015/1222).** Capacity calculation regions (Art. 15); the
  flow-based approach (Art. 2(9), Art. 20) vs coordinated net transmission capacity; the content of a capacity
  calculation methodology: reliability margin, operational security limits and contingencies, generation shift
  keys and remedial actions (Art. 21–25); validation (Art. 26). Core CCR flow-based day-ahead market coupling
  as the operational example.
- **Flow-based concepts.**
    - CNEC: a critical network element monitored after a contingency (including the base case).
    - GLSK: how a change of a zone's net position is spread over its nodes, a matrix
      $G \in \mathbb{R}^{N \times Z}$ whose columns sum to one.
    - Zonal PTDF: $\text{PTDF}_z = \text{PTDF}_n\,G$ (revisits the nodal PTDF of
      [Chapter 17](ch17-sensitivity-factors-1.md)); zone-to-zone PTDFs are column differences and do not depend
      on the slack.
    - Flow at zero net positions: $F_0 = F_\text{ref} - \text{PTDF}_z\,NP_\text{ref}$.
    - Remaining available margin: $\text{RAM} = F_\text{max} - \text{FRM} - F_0$ (minus further operator
      adjustments), with the flow reliability margin FRM covering forecast uncertainty (CACM Art. 22).
    - Flow-based domain: $\{NP : \text{PTDF}_z\,NP \le \text{RAM},\ \sum_z NP_z = 0\}$, one half-space per CNEC
      and direction.
- **CRAC.** Contingencies, remedial actions and other constraints: the file that bundles contingencies, CNECs
  with their instants and limits, and remedial actions with usage rules.
- **Preventive, automatic and curative remedial actions.** Instants: before outage, after outage, after
  automatons, after curative actions. The preventive perimeter checks base-case CNECs against PATL and
  after-outage CNECs against TATL; curative perimeters then restore post-contingency flows within the time
  allowed by the temporary limit. A curative action that can be activated in time avoids a costly preventive one.
- **OpenRAO.** CASTOR's search tree: at each depth, create candidates by applying network actions, optionally
  optimize linear remedial actions (PSTs, HVDC setpoints) with an LP/MILP, run a security analysis per candidate,
  keep the best candidate, and stop when the objective (e.g. maximizing the minimum margin) no longer improves
  enough. Compare with ToOp's Map-Elites over millions of candidates
  ([Chapter 25](../part-4-optimization/ch25-quality-diversity.md)).
  MARMOT is OpenRAO's time-coupled RAO.
- **Uncertainty and time.** Forecast errors between D-1 and real time; reliability margins, conservative limits
  (ToOp's double limits and white-list limits are deterministic versions of this); why a topology chosen per
  hour needs switching costs and feasibility between hours, and why ToOp instead holds one topology for all
  timesteps of a run.

## Learning outcomes

- Place IGM/CGM building, coordinated security analysis, capacity calculation and remedial action optimization on
  the day-ahead to real-time timeline, citing the SO GL and CACM articles that define them.
- Compute zonal PTDFs from a nodal PTDF and a GLSK, the RAM of each CNEC, and the flow-based domain of a small
  grid, and cross-check the zonal PTDF with pypowsybl.
- Explain preventive vs curative remedial actions with PATL/TATL, and decide when a curative action is sufficient.
- Map ToOp's inputs and outputs (N-1 definition, masks, action set, ORAO summary) to CRAC concepts and state what
  a coordinated process would still add.

## Resources

- Regulations:
  - SO GL, Regulation (EU) 2017/1485 <https://eur-lex.europa.eu/eli/reg/2017/1485/oj/eng>: Art. 20–23
    (remedial actions), 64–71 (grid models), 72–78 (operational security analysis and regional coordination), 79
    (CGM building).
  - CACM, Regulation (EU) 2015/1222 <https://eur-lex.europa.eu/eli/reg/2015/1222/oj/eng>: Art. 2 (definitions),
    15 (capacity calculation regions), 17 (CGM methodology), 20–26 (flow-based and capacity calculation
    methodologies), 28–30 (CGM creation, regional calculation, validation).
- Paper: Van den Bergh, Boury & Delarue, "The Flow-Based Market Coupling in Central Western Europe: Concepts and
  definitions", *The Electricity Journal* 29(1):24–29, 2016, doi:10.1016/j.tej.2015.12.004.
- Documentation:
  - OpenRAO <https://powsybl.readthedocs.io/projects/openrao/en/latest/>: glossary
    <https://powsybl.readthedocs.io/projects/openrao/en/latest/glossary.html> (CGM, CNEC, CRAC, GLSK, RAM, PATL,
    TATL), CASTOR <https://powsybl.readthedocs.io/projects/openrao/en/latest/algorithms/castor.html>, multi-step
    optimization (instants and perimeters)
    <https://powsybl.readthedocs.io/projects/openrao/en/latest/algorithms/castor/rao-steps.html>, CRAC
    <https://powsybl.readthedocs.io/projects/openrao/en/latest/input-data/crac.html>, GLSK
    <https://powsybl.readthedocs.io/projects/openrao/en/latest/input-data/glsk.html>, MARMOT
    <https://powsybl.readthedocs.io/projects/openrao/en/latest/algorithms/marmot.html>.
  - pypowsybl RAO <https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/user_guide/rao.html> and
    zonal sensitivity analysis
    <https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/user_guide/sensitivity.html>.
  - JAO, Core flow-based day-ahead market coupling <https://www.jao.eu/core-fb-da-mc>; ENTSO-E CACM page
    <https://www.entsoe.eu/network_codes/cacm/> and capacity calculation regions with the Core CCR deliverables
    <https://www.entsoe.eu/network_codes/ccr-regions/>. The Core day-ahead capacity calculation methodology
    document itself: resource TBD.
  - ToOp: `save_orao_summary` in `optimizer/benchmark/benchmark_utils.py` and `changing_switches_to_orao_dict` in
    `optimizer/ac/summary.py` (the ORAO summary), `docs/dc_solver/quickstart.md` (white and
    black lists).

## Worked example

Four buses with equal reactances, zone A = {0, 1}, zone B = {2, 3}, slack at bus 0.

```python
import numpy as np

branches = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]    # x = 1 pu each
A = np.zeros((5, 4))
for k, (f, t) in enumerate(branches):
    A[k, f], A[k, t] = 1.0, -1.0
ns = [1, 2, 3]                                          # bus 0 is the slack
ptdf = np.zeros((5, 4))
ptdf[:, ns] = A[:, ns] @ np.linalg.inv((A.T @ A)[np.ix_(ns, ns)])

glsk = np.array([[0.6, 0.0], [0.4, 0.0], [0.0, 0.5], [0.0, 0.5]])   # rows: buses, columns: zones A, B
ptdf_zone = ptdf @ glsk                                 # (CNECs, zones)
ptdf_ab = ptdf_zone[:, 0] - ptdf_zone[:, 1]             # sensitivity to an A -> B exchange

p_ref = np.array([300.0, -50.0, -150.0, -100.0])        # reference nodal injections, MW
f_ref = ptdf @ p_ref
np_ref = np.array([p_ref[:2].sum(), p_ref[2:].sum()])   # net positions A, B
f_0 = f_ref - ptdf_zone @ np_ref                        # flows with all net positions at zero
ram = 200.0 - 10.0 - f_0                                # F_max = 200 MW, FRM = 10 MW, from -> to direction
print(np.round(ptdf_ab, 3))           # [0.188 0.412 0.362 0.138 0.225]
print(np.round(ram, 1))               # [ 99.4 130.6 218.1 186.9 221.2]
print(np.round(ram / ptdf_ab))        # [ 530.  317.  602. 1359.  983.]
```

Branch 0–2 limits the A→B exchange to 317 MW. A topology action that lowers its zone-to-zone PTDF or its $F_0$
enlarges the domain for the market; this is the capacity-calculation view of a remedial action.

## Lab

!!! example "Lab: zonal PTDF, RAM and a flow-based domain"

    Tools: Jupyter with NumPy, pandas and matplotlib ([Chapter 30](../part-5-computing/ch30-scientific-python.md)),
    pypowsybl.

    1. Load `pypowsybl.network.create_ieee14()`, run a DC load flow and build the nodal PTDF for all lines with
       your code from [Chapter 17](ch17-sensitivity-factors-1.md) (or with pypowsybl's DC sensitivity analysis
       on generator and load injections).
    2. Define three zones over the 14 buses, and two GLSKs: proportional to generator `target_p` where a zone
       has generation, and flat over all buses of the zone.
    3. Compute $\text{PTDF}_z$ for both GLSKs. Cross-check one zone with pypowsybl:
       `pypowsybl.sensitivity.create_zone_from_injections_and_shift_keys`, `set_zones` and
       `add_branch_flow_factor_matrix(branches_ids=..., variables_ids=[zone ids])` on a DC analysis run with
       `pypowsybl.loadflow.Parameters(distributed_slack=False)`; zone factors equal the injection factors
       weighted by the shift keys.
    4. Treat every line in both directions as a CNEC. The IEEE 14 model has no operational limits, so assume
       $F_\text{max}$ = 1.3 times the absolute base-case flow (at least 20 MW) and FRM = 5 % of $F_\text{max}$;
       compute $F_0$ and RAM from the reference flows and net positions.
    5. Plot the flow-based domain in the plane of two independent net positions (the third is minus their sum)
       with matplotlib, mark the reference point and label the constraining CNECs.
    6. Open one line (a simple topology action), recompute the PTDF, and overlay the new domain. Report which
       exchanges gained or lost capacity, and whether the change would pass an N-1 check from
       [Chapter 19](ch19-operational-security.md).

## Exercises

1. Pen and paper: a line has PATL 1 000 A and TATL 1 250 A for 15 minutes. After a contingency it carries 1 180 A.
   A curative busbar split reduces the flow to 940 A and takes 10 minutes to activate. Is a preventive action
   needed? What if activation takes 20 minutes, or the post-contingency flow is 1 300 A?
2. *(GeoGebra)* For two zones plus a third balancing zone, draw the half-planes of five CNECs with given zonal
   PTDFs and RAMs, shade the flow-based domain, and move a slider that increases the RAM of the binding CNEC.
   Find the largest A→B exchange before and after.
3. *(Jupyter: NumPy)* In the worked example, change the zone A GLSK from (0.6, 0.4) to (1.0, 0.0) and to
   (0.0, 1.0). Show that $\text{PTDF}_z$, $F_0$ and RAM change while the physical flows at the reference point do
   not, and explain why GLSK errors are a source of the reliability margin.
4. Pen and paper: map ToOp's `nminus1_definition.json`, the `line_for_reward` and `line_for_nminus1` masks,
   `branch_limits.max_mw_flow_n_1` and `action_set.json` to CRAC elements (contingencies, CNECs, instants, limits,
   remedial actions, usage rules). List three CRAC features that ToOp has no representation for.
5. *(Jupyter: pandas)* Run `notebooks/example2_small_grid_toop.ipynb` (`perform_ac_analysis` writes its output), load
   one `orao_summary` JSON of an accepted topology, build a table of switch actions (switch id, open or close),
   join it with the switches of the grid from pypowsybl (`get_switches`, including `kind` and `voltage_level_id`),
   and flag every disconnector operation that would need a breaker in the sequence
   ([Chapter 20](ch20-protection-short-circuit.md)).
6. *(Jupyter: NumPy)* Time coupling: for 24 hours and four candidate topologies with given hourly overload
   energies, compare (a) the best single topology for all hours, as ToOp does, with (b) the best hourly
   sequence under a switching cost per change, solved by dynamic programming. At what switching cost do the two
   coincide?
