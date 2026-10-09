# Chapter 13. Substations, grid models and data formats

**Weeks:** 2 · **Semester:** 2 · **Prerequisites:** [Chapter 3](../part-1-math/ch03-graph-theory.md), [Chapter 12](ch12-power-grid-operation.md)

!!! info "Why it matters for ToOp"

    Busbar splitting and reassignment are physical switching operations inside
    substations. ToOp maps node-breaker models (busbars, couplers, bays, breakers, disconnectors) to a bus-branch
    model and back (`docs/interfaces/asset_topology.md`, `importer/network_graph/`), enumerates coupler states
    that realize an electrical split (`dc_solver/preprocess/preprocess_switching.py`), and scores the switching
    effort as a Hamming distance (`docs/dc_solver/switching_distance.md`,
    `dc_solver/preprocess/helpers/switching_distance.py`). It reads UCTE-DEF, CGMES and XIIDM files and exports
    UCTE and PowerFactory DGS.

    This chapter is taught in semester 2, before the network matrices of
    [Chapter 14](ch14-component-models.md) and the bus-split factors of
    [Chapter 18](ch18-sensitivity-factors-2.md), which assume that you can read a station as a graph of busbars
    and couplers.

## Topics

- Substation arrangements: single and double busbar, breaker-and-a-half, ring; busbar couplers, sections,
  bays, breakers vs disconnectors.
- Node-breaker vs bus-breaker vs bus-branch models; topology processing (revisits connected components from
  [Chapter 3](../part-1-math/ch03-graph-theory.md)).
- **Electrical vs physical switching.** The *electrical* topology says which branch is on which electrical
  busbar (ToOp allows at most a two-way split, bus A and bus B); the *physical* topology says how every
  coupler and bay switch is set. Optimization works on the electrical topology; operators pay for the physical
  switching.
- **Switching distance in ToOp's code.** For a station with $n$ couplers, `make_separation_set` enumerates all
  $2^n$ open/closed coupler states (it raises an error for $n > 20$), builds the busbar graph for each state,
  and keeps the states whose graph has exactly two connected components. Each kept state gives a candidate
  (bus A, bus B) assignment of assets and its coupler state. The distance to a target electrical topology is
  the minimum Hamming distance over these candidates, checked for both labelings of A and B, and reported as
  a reassignment distance plus a coupler distance (`per_station_switching_distance`).
  `make_optimal_separation_set` removes candidates that duplicate another one (or its A/B mirror) and, for
  tables with at least `clip_at_size` rows (default 100), also those within `clip_hamming_distance` of another.
- "Topo-vect" (two nodes per substation) as the solver's view of a station.
- Data formats: UCTE-DEF, CGMES / CIM, PowSyBl IIDM/XIIDM, pandapower JSON, MATPOWER.
- Merging models from several TSOs (boundaries, tie lines); single-line diagrams.
- Forward link: state estimation and topology errors (a breaker status that is wrong in the model) are treated
  in [Chapter 22](ch22-import-preprocessing.md).

## Learning outcomes

- Draw the node-breaker, bus-breaker and bus-branch views of a double-busbar substation and convert between
  them.
- Enumerate by hand the coupler states of a small station that produce exactly two electrical busbars, and
  compute the switching distance of a target assignment as ToOp does.
- Name the main grid data formats used by European TSOs and state which of them keep node-breaker detail.

## Resources

- Textbook: Wood, Wollenberg & Sheblé 3rd ed., §6.2 "Conversion of equipment data to bus and branch data" and
  §6.3 "Substation bus processing".
- Free documentation:
  - PowSyBl / pypowsybl user guide (node-breaker and bus-breaker voltage levels, security analysis, sensitivity
    analysis): <https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/user_guide/index.html>; notebooks
    <https://github.com/powsybl/pypowsybl-notebooks>.
  - ENTSO-E CGMES library <https://www.entsoe.eu/data/cim/cim-for-grid-models-exchange/> and technical site
    <https://cim-cgmes.github.io/>.
  - UCTE data exchange format specification (cited in ToOp):
    <https://eepublicdownloads.entsoe.eu/clean-documents/pre2015/publications/ce/otherreports/UCTE-format.pdf>.
- Papers: Heidarifar & Ghasemi, "A Network Topology Optimization Model Based on Substation and Node-Breaker
  Modeling", *IEEE Trans. Power Systems* 31(1):247–255, 2016.
- Reference: Westinghouse, *Electrical Transmission and Distribution Reference Book* (1964). **BORROW**
  [`electricaltransm0000cent`](https://archive.org/details/electricaltransm0000cent).
- ToOp: `docs/dc_solver/switching_distance.md`, `docs/interfaces/asset_topology.md`.

## Worked example

A station has four physical busbars $b_1,\dots,b_4$ connected in a chain by three closed couplers
$c_{12}, c_{23}, c_{34}$. Opening a set of couplers splits the chain into connected components. With $n = 3$
there are $2^3 = 8$ coupler states:

- all closed: one component (skipped; this is the unsplit station);
- exactly one coupler open: two components, 3 states ($\{b_1\}|\{b_2 b_3 b_4\}$,
  $\{b_1 b_2\}|\{b_3 b_4\}$, $\{b_1 b_2 b_3\}|\{b_4\}$);
- two or three couplers open: three or four components (discarded).

Each of the three kept states defines a candidate: two Boolean rows $a_A, a_B \in \{0,1\}^m$ over the $m$
assets, where $a_A$ marks the assets whose bays connect to a busbar of the first component and $a_B$ those of
the second. For a target electrical topology $t \in \{0,1\}^m$ the code computes

$$
d(t, a) = \min\left(\lVert a_A \oplus \bar t \rVert_1 + \lVert a_B \oplus t \rVert_1,\;
\lVert a_A \oplus t \rVert_1 + \lVert a_B \oplus \bar t \rVert_1\right),
$$

where $\oplus$ is exclusive or and the minimum accounts for swapping the names A and B. Because the bus A and
bus B rows are compared separately, an asset connected to both busbars, or to neither, is counted correctly,
and a singly connected asset that changes busbar counts twice (one connection opened, one closed). ToOp keeps
the candidate with the smallest reassignment distance and reports its coupler distance (the Hamming distance
between the candidate's coupler state and the current one) separately.

```python
import itertools
import networkx as nx

couplers = [(0, 1), (1, 2), (2, 3)]
for is_open in itertools.product([True, False], repeat=len(couplers)):
    g = nx.Graph()
    g.add_nodes_from(range(4))
    g.add_edges_from(c for c, o in zip(couplers, is_open) if not o)
    parts = list(nx.connected_components(g))
    if any(is_open) and len(parts) == 2:
        print(is_open, parts)
```

## Documentation vs code: known discrepancies

- `docs/dc_solver/switching_distance.md` says the two-cluster assignments are enumerated with "a graph
  algorithm involving edge cuts". The code in `make_separation_set` (`dc_solver/preprocess/preprocess_switching.py`)
  brute-forces all $2^n$ coupler states with `itertools.product` and a connected-components test, with a hard
  limit of 20 couplers.
- The same document says a multi-connected asset keeps the busbar "with the most other branches connected to
  it". The helper used during station preparation, `fix_multi_connected_without_coupler`
  (`grid_helpers/asset_topology_helpers.py`), documents and implements removing the connection to the busbar
  with the lower index.
- `docs/topology_optimizer/metrics.md` states that the AC solver counts `switching_distance` slightly differently
  from the DC solver and calls it a bug.

## Lab

!!! example "Lab: from a node-breaker station to a switching distance"

    Tools: ToOp notebooks (set up in [Chapter 0](../part-0-orientation/ch00-orientation-and-setup.md)), pypowsybl,
    NetworkX ([Chapter 3](../part-1-math/ch03-graph-theory.md)).

    1. Work through `notebooks/example_asset_topology_analysis.ipynb`: inspect a voltage level's node-breaker
       graph and single-line diagram, and identify busbars, couplers and bays.
    2. For one bus group, list its couplers and busbars, and enumerate by hand (or with the snippet from the
       worked example) the coupler states that give exactly two electrical busbars.
    3. Call `prepare_for_separation_set` and `make_optimal_separation_set`
       (`dc_solver/preprocess/preprocess_switching.py`) on the same bus group and check that the separation set
       matches your enumeration.
    4. Choose a target electrical assignment of the station's branches and compute its switching distance with
       `per_station_switching_distance` (`dc_solver/preprocess/helpers/switching_distance.py`). Verify the
       reassignment and coupler distances by hand.

## Exercises

1. Draw a double-busbar, single-breaker substation with four line bays and one coupler. Give its node-breaker,
   bus-breaker and bus-branch representations, and list which switches change when one line moves to the other
   busbar.
2. Explain why a disconnector must not interrupt load current and how that constrains the sequence of
   switching operations for a busbar reassignment.
3. *(Jupyter: Python + NetworkX)* Generalize the worked example to a ring of $n$ busbars with $n$ couplers. Count
   the two-component coupler states for $n = 3,\dots,12$, compare with the closed form $\binom{n}{2}$, and time
   the brute force as $n$ grows toward ToOp's limit of 20.
4. *(Jupyter: NumPy)* Given the candidates as a Boolean array of shape (candidates, 2, assets) and a target
   vector, compute the reassignment distance of the worked example with and without the A/B swap in one
   vectorized expression. Construct an example where ignoring the swap overstates the distance.
5. *(pypowsybl)* Create `pypowsybl.network.create_four_substations_node_breaker_network()`, list its voltage
   levels and switches, and for one voltage level classify the switches into breakers and disconnectors.
