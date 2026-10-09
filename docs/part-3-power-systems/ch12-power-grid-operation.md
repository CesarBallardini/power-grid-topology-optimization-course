# Chapter 12. The power grid and its operation

**Weeks:** 1.5 · **Semester:** 2 · **Prerequisites:** [Chapter 9](../part-2-circuits/ch09-ac-steady-state.md)

!!! info "Why it matters for ToOp"

    ToOp is built by transmission system operators (Elia in Belgium, 50Hertz in
    Germany) for their operational processes: day-ahead congestion forecasts, border lines with neighboring TSOs,
    transformers to distribution grids, control and observation areas (`importer/pypowsybl_import/`,
    `AreaSettings`).

    Topology optimization is one of the cheapest remedial actions an operator has: switching busbars costs
    almost nothing, while redispatch and countertrading cost hundreds of millions of euros per quarter in
    Germany alone (see the lab). Knowing who pays for congestion, and when in the planning timeline a
    topology can still be changed, explains why ToOp optimizes what it optimizes.

## Topics

- Generation, transmission, distribution; voltage levels; synchronous areas.
- Roles: TSOs, DSOs, regulators, ENTSO-E; European grid codes (System Operation Guideline).
- Electricity markets at awareness level: zonal markets, market coupling, nodal prices (LMP), redispatch.
- Congestion: why it appears (renewables far from load, cross-border flows) and what it costs.
- Remedial actions and their cost ranking: topology changes (busbar splitting, line switching), phase-shifter
  taps, redispatch, countertrading, curtailment.
- Operational planning timeline: day-ahead congestion forecast (DACF), intraday, real time. The coordinated
  European processes (common grid models, capacity calculation, remedial action optimization) are covered in
  [Chapter 21](ch21-coordinated-security-rao.md).

## Learning outcomes

- Describe the roles of TSOs, DSOs, regulators and ENTSO-E, and place DACF, intraday and real-time operation on
  a timeline.
- Compute the dispatch, nodal prices, congestion rent and redispatch cost of a two-node market with a line
  limit.
- Load public congestion-management and generation data with pandas and quantify the cost of congestion per
  MWh.

## Resources

- Courses: Coursera *Electric Power Systems* (University at Buffalo) <https://www.coursera.org/learn/electric-power-systems>;
  MIT OCW 6.061 <https://ocw.mit.edu/courses/6-061-introduction-to-electric-power-systems-spring-2011/>;
  DelftX *Smart Grids Integration and Modeling* (edX).
- Regulation: Commission Regulation (EU) 2017/1485, System Operation Guideline (SO GL):
  <https://eur-lex.europa.eu/eli/reg/2017/1485/oj/eng>.
- Data:
  - ENTSO-E Transparency Platform (load, generation, cross-border physical flows, congestion management data
    for all European bidding zones): <https://transparency.entsoe.eu/>.
  - SMARD, the Bundesnetzagentur's electricity market data portal (German generation, consumption, prices and
    quarterly congestion-management reports): <https://www.smard.de/en>; data download center
    <https://www.smard.de/en/downloadcenter/download-market-data>.
  - SMARD, "Congestion management in Q1 2026: lower volume of measures and costs" (volumes and estimated costs
    of redispatch and countertrading):
    <https://www.smard.de/page/en/topic-article/5892/221272/lower-volume-of-measures-and-costs>.
- Textbooks:
  - von Meier, *Electric Power Systems: A Conceptual Introduction* (Berkeley), 2nd ed. 2024. The best
    qualitative entry point. **NOT ON IA**.
  - Kirschen, *Power Systems: Fundamental Concepts and the Transition to Sustainability* (Wiley 2024).
    **NOT ON IA**.
  - Kirschen & Strbac, *Fundamentals of Power System Economics* (6 syllabi, the markets reference). **NOT ON
    IA**.

## Lab

!!! example "Lab: what congestion costs"

    Tools: Jupyter + pandas + matplotlib ([Chapter 30](../part-5-computing/ch30-scientific-python.md)).

    1. Read the SMARD congestion-management article for Q1 2026 (Resources). Put the reported figures in a
       small pandas `DataFrame` (quarter, total volume of measures in GWh, estimated cost in million euros, of
       which curtailment of renewables in GWh) for Q1 2025 and Q1 2026. Compute the average cost per MWh of
       congestion management and its change between the two quarters. The article reports 9,035 GWh and about
       882 million euros for Q1 2025, and 8,248 GWh and about 784 million euros for Q1 2026.
    2. From the SMARD download center, export one winter week of hourly German electricity generation by
       source and total consumption as CSV. Inspect the header, delimiter and number format before parsing it
       with `pandas.read_csv`. Plot wind generation and consumption on the same axes, and plot the residual
       load (consumption minus wind and solar) as a duration curve.
    3. Mark the hours with high wind and low residual load. In a short paragraph, explain why these are the
       hours in which north–south transmission corridors congest, and which remedial actions (from the Topics
       list) an operator would try first.
    4. Optional: on the ENTSO-E Transparency Platform, look up the cross-border physical flows of Belgium or
       Germany for the same week and relate them to your plot.

## Exercises

1. Two-node market, pen and paper. Node A has a generator with marginal cost 20 €/MWh and 1,000 MW capacity;
   node B has a generator with 60 €/MWh and 1,000 MW capacity and a load of 800 MW. The line A–B is limited to
   500 MW. (a) Find the nodal-pricing dispatch, both nodal prices and the congestion rent. (b) A zonal market
   ignores the line limit: find its dispatch and price, then the redispatch needed to respect the limit and its
   cost. *Hint:* (a) 500 MW from A, 300 MW from B, prices 20 and 60 €/MWh, rent 20,000 €/h; (b) 800 MW from A,
   redispatch 300 MW at a cost of 12,000 €/h.
2. *(GeoGebra)* Build the two-node market of Exercise 1 with sliders for the line limit and the load at B.
   Plot the redispatch cost and the congestion rent as functions of the line limit, and explain the kink.
3. Draw the operational planning timeline from two days ahead to real time. Place the day-ahead congestion
   forecast, intraday updates and real-time operation, and mark where a ToOp run fits and how much time it
   has.
4. *(Jupyter: pandas + matplotlib)* With the week of SMARD data from the lab, compute the daily share of wind
   and solar in consumption and the maximum hourly ramp of residual load. Which day would you expect to be the
   hardest for the grid operator, and why?
5. Rank busbar splitting, phase-shifter tap changes, redispatch, countertrading and renewable curtailment by
   cost and by speed of activation. For each, say whether a market participant is involved.
