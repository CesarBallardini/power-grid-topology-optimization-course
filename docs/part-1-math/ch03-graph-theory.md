# Chapter 3. Graph theory for power grids

**Weeks:** 2 · **Semester:** 1 · **Prerequisites:** [Chapter 2](ch02-linear-algebra.md)

!!! info "Why it matters for ToOp"

    Branches whose outage splits the grid (**bridges**) cannot be handled by LODF
    and are removed from the N-1 set (`dc_solver/preprocess/helpers/find_bridges.py`, using `nx.bridges`).
    Busbars that are **articulation points** are excluded from busbar outages
    (`dc_solver/preprocess/preprocess_bb_outage.py`). A busbar split is valid only if the station graph has
    exactly **two connected components** (`dc_solver/preprocess/preprocess_switching.py`). The importer finds
    bays and couplers with **weighted shortest paths with cutoffs** (`importer/network_graph/`). Switching
    distance is a **Hamming distance** (`dc_solver/preprocess/helpers/switching_distance.py`).
    [Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md) shows why a bridge makes the LODF
    denominator vanish; [Chapter 4](ch04-numerical-linear-algebra.md) meets the same zero in a resistor network.

## Topics

- Graphs, multigraphs, directed vs undirected; adjacency and incidence matrices.
- Paths, cycles, trees, spanning trees; the Matrix-Tree theorem (preview of Laplacians).
- Connectivity, connected components, **bridges (cut edges)**, **articulation points (cut vertices)**,
  blocks; Tarjan's DFS algorithm; union–find.
- BFS/DFS, Dijkstra shortest paths; k-hop neighborhoods.
- Graph Laplacian $L = D - W = A^\top \operatorname{diag}(w) A$ (revisits [Chapter 2](ch02-linear-algebra.md)'s
  $A^\top \operatorname{diag}(g) A$; revisited in [Chapter 8](../part-2-circuits/ch08-dc-circuits.md) as the nodal
  conductance matrix), algebraic connectivity, effective resistance
  $R_{ij} = (e_i - e_j)^\top L^{+} (e_i - e_j)$ (link to
  [Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md)).
- Enumeration of subsets and bipartitions ($2^{n-1}$ busbar assignments).

## Learning outcomes

- Find bridges and articulation points by hand and with NetworkX, and explain why an outage of a bridge
  creates an island.
- Relate the number of zero eigenvalues of the Laplacian to the number of islands.
- Enumerate the open/closed states of the couplers of a small station and count those that produce exactly
  two electrical busbars.

## Resources

- Courses: Coursera *Introduction to Graph Theory* (UC San Diego) <https://www.coursera.org/learn/graphs>;
  NPTEL Graph Theory (IISc, Chandran) <https://nptel.ac.in/courses/106108054>; MIT OCW 6.042J graph units.
- Free texts:
  - Diestel, *Graph Theory*, free preview edition, Ch. 1 (bridges, cycle and cut space) and Ch. 3
    (connectivity): <https://diestel-graph-theory.com/basic.html>.
  - Spielman, *Spectral and Algebraic Graph Theory* (draft), Ch. 3 "The Laplacian and Graph Drawing",
    Ch. 11 "Walks, Springs, and Resistor Networks", Ch. 12 "Effective Resistance and Schur Complements":
    FREE <https://www.cs.yale.edu/homes/spielman/sagt/>.
  - Dörfler, Simpson-Porco & Bullo, "Electrical Networks and Algebraic Graph Theory: Models, Properties,
    and Applications", *Proc. IEEE* 106(5), 2018. Author PDF: <http://motion.me.ucsb.edu/pdf/2017k-dsb.pdf>.
    **The best bridge from Laplacians to power networks.**
  - Barabási, *Network Science*, Ch. 2 (graph theory) and Ch. 8 (robustness, cascades): FREE
    <https://networksciencebook.com/>.
  - Easley & Kleinberg, *Networks, Crowds, and Markets*, Ch. 2–3 (bridges and local bridges): FREE
    <https://www.cs.cornell.edu/home/kleinber/networks-book/>.
- Textbooks:
  - Diestel, *Graph Theory* (2 syllabi). 3rd ed. 2005 **BORROW**
    [`graphtheory0000dies`](https://archive.org/details/graphtheory0000dies).
  - West, *Introduction to Graph Theory* (2 syllabi). **PRINT-DISABLED**.
  - Bondy & Murty, *Graph Theory with Applications* (1976): §1.3 incidence and adjacency matrices, §2.2 cut
    edges, §2.3 cut vertices, §3.1–3.2 connectivity and blocks, §12.1 circulations and potential
    differences. **PRINT-DISABLED**.
  - Deo, *Graph Theory with Applications to Engineering and Computer Science* (1974), the most
    engineering-oriented classic (incidence, circuit and cut-set matrices; network analysis). **BORROW**
    [`graphtheorywitha0000deon`](https://archive.org/details/graphtheorywitha0000deon).
  - Cormen, Leiserson, Rivest & Stein, *Introduction to Algorithms*, 3rd ed., Ch. 22 (elementary graph
    algorithms; articulation points and bridges appear as a chapter problem) and Ch. 24 (shortest paths).
    **BORROW** [`introductiontoal0000unse_t3a6`](https://archive.org/details/introductiontoal0000unse_t3a6).
- Documentation and tools (NetworkX):
  [`bridges`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bridges.bridges.html),
  [`articulation_points`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.articulation_points.html),
  [`connected_components`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html),
  [`laplacian_matrix`](https://networkx.org/documentation/stable/reference/generated/networkx.linalg.laplacianmatrix.laplacian_matrix.html),
  [`resistance_distance`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.distance_measures.resistance_distance.html),
  [`from_pandas_edgelist`](https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_edgelist.html).

## Lab

!!! example "Lab: bridges, islands and busbar splits"

    Tools: Jupyter + pandas + NetworkX + NumPy + matplotlib (uses
    [Chapter 30](../part-5-computing/ch30-scientific-python.md) basics). The grid data is a plain edge list; no
    power system package is needed.

    1. Save the branch list of the IEEE 14-bus test case (lines and transformers, bus numbers from 1) as
       `ieee14_edges.csv` and load it with `pd.read_csv`:

        ```text
        from_bus,to_bus
        1,2
        1,5
        2,3
        2,4
        2,5
        3,4
        4,5
        4,7
        4,9
        5,6
        6,11
        6,12
        6,13
        7,8
        7,9
        9,10
        9,14
        10,11
        12,13
        13,14
        ```

    2. Build a graph with `nx.from_pandas_edgelist(df, "from_bus", "to_bus", create_using=nx.MultiGraph)`
       (real grids have parallel circuits, so a multigraph is the honest model). Add a duplicate of branch
       `(7, 8)` and check that `nx.bridges` no longer reports it: explain why a branch with a parallel circuit
       can never be a bridge. Remove the duplicate, list bridges and articulation points (expected: one
       bridge, `(7, 8)`, and one articulation point, bus 7) and draw the graph highlighting them.
    3. Compute the Laplacian with `nx.laplacian_matrix` and count its zero eigenvalues before and after removing
       the bridge. Compute the effective resistance across `(7, 8)` with unit weights and compare it with every
       other branch: what is special about the value 1?
    4. Open `dc_solver/preprocess/helpers/find_bridges.py` in ToOp and identify the NetworkX call it relies on
       and what it does with the result.
    5. Station graph: four busbars `0–3` joined by couplers `(0,1), (1,2), (2,3), (3,0)`. Enumerate all
       $2^4$ open/closed coupler states with `itertools.product`, skip the all-closed state and keep the states
       whose graph of closed couplers has exactly two connected components. Count states and distinct
       bipartitions (expected: 6 and 6). Add a coupler `(0,2)` and repeat (12 states, still 6 bipartitions).
       This is the idea behind `make_separation_set` in `dc_solver/preprocess/preprocess_switching.py`.
    6. Stretch: repeat steps 2–3 on a larger edge list, such as the IEEE 118-bus branch table exported to CSV
       once the pandapower basics of [Chapter 30](../part-5-computing/ch30-scientific-python.md) are covered.

## Exercises

1. By hand, find the bridges and articulation points of a graph made of two triangles joined by a single
   edge. Run Tarjan's DFS on it and write down the discovery times and low-link values. *(Pen and paper)*
2. Show that an edge is a bridge if and only if it lies on no cycle, and that removing a bridge from a
   connected graph leaves exactly two components. *(Pen and paper)*
3. For the IEEE 14-bus edge list, compute the algebraic connectivity (second-smallest Laplacian eigenvalue)
   and see how it changes when each branch is removed in turn. Plot the values as a bar chart and compare the
   worst branches with the bridge. *(Jupyter: NetworkX + NumPy + matplotlib)*
4. A station has $n$ busbars that can each be assigned to side A or B. Explain why there are $2^{n-1}$
   assignments up to swapping A and B, and $2^{n-1} - 1$ non-trivial ones. In the ring station of the lab, which
   non-trivial bipartition cannot be reached by opening couplers, and why? *(Pen and paper)*
5. Two topologies are encoded as Boolean vectors of busbar assignments. Write a Hamming distance function in
   NumPy and explain why ToOp also computes the distance to the inverted vector
   (`dc_solver/preprocess/helpers/switching_distance.py`). *(Jupyter: NumPy)*
6. Weighted shortest paths with a cutoff: on the IEEE 14-bus graph with unit weights, list all buses within
   2 hops of bus 5 with `nx.single_source_dijkstra_path_length(G, 5, cutoff=2)`. *(Jupyter: NetworkX)*
