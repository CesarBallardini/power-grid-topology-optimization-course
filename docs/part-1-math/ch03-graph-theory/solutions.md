# Chapter 3 – Solutions

Solutions of the Lab and the Exercises of [Chapter 3](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`grid_graph.py`](src/grid_graph.py) (edge list, bridges, Laplacian,
effective resistance, algebraic connectivity, k-hop neighbourhoods), [`ieee118.py`](src/ieee118.py) (the 118-bus
stretch), [`station_splits.py`](src/station_splits.py) (coupler states of a station), [`tarjan.py`](src/tarjan.py)
(DFS with discovery times and low-links) and [`hamming.py`](src/hamming.py) (switching distance). Each module's
`main()` prints every number quoted below; the worked examples run in [examples.ipynb](src/examples.ipynb).

## Lab

1. **Edge list.** `pd.read_csv('ieee14_edges.csv')` gives 20 rows with columns `from_bus` and `to_bus`; the 14
   buses appear at least once each, bus 8 exactly once (`load_edges` in `grid_graph.py`).
2. **Bridges and articulation points.** `nx.from_pandas_edgelist(df, 'from_bus', 'to_bus',
   create_using=nx.MultiGraph)` builds a multigraph with 14 nodes and 20 edges. After `G.add_edge(7, 8)` the
   multigraph has two parallel edges between buses 7 and 8 and `list(nx.bridges(G))` is empty: a bridge is an
   edge whose removal disconnects its two ends, and when a parallel circuit exists, removing one of the two
   leaves the other, so the ends stay connected (the two parallel edges form a cycle of length 2, see
   [Exercise 2](#exercise-2)). After `G.remove_edge(7, 8)` the result is `bridges = [(7, 8)]` and
   `articulation_points = [7]`: bus 8 is a leaf attached to bus 7 by a single transformer, so that transformer's
   outage islands bus 8, and the loss of bus 7 does the same. `draw_grid` colours the bridge red and draws bus 7
   as a larger red node; no other branch is red, since every other bus has at least two branches that lead to
   the rest of the grid.
3. **Zero eigenvalues and effective resistance.** `np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray())` has
   exactly one eigenvalue below $10^{-9}$ on the intact grid and two after removing `(7, 8)`
   (`count_zero_eigenvalues`), one per connected component: the Laplacian's null space is spanned by the
   indicator vectors of the islands. With unit weights the effective resistance across `(7, 8)` is exactly
   $1$, and every other branch is below it:

    | Branch | $R_{\text{eff}}$ | Branch | $R_{\text{eff}}$ |
    |---|---|---|---|
    | 7–8 | 1.0000 | 1–2, 1–5, 2–3, 3–4 | 0.6135 |
    | 6–11, 9–10, 10–11 | 0.7629 | 6–13 | 0.5541 |
    | 9–14, 13–14 | 0.7467 | 4–9 | 0.5449 |
    | 5–6 | 0.7260 | 4–5 | 0.4820 |
    | 6–12, 12–13 | 0.6385 | 2–4, 2–5 | 0.4538 |
    | 4–7, 7–9 | 0.6362 | | |

    A branch of resistance 1 in parallel with any other path has an effective resistance below 1, by the
    parallel rule $R = (1 + 1/R_{\text{path}})^{-1}$. The value $1$ means there is no parallel path at all: the
    whole unit current injected at bus 7 and withdrawn at bus 8 flows through the branch itself, which is the
    definition of a bridge. The equality $g_k R_{\text{eff},k} = 1$ for a bridge is the zero denominator of
    [Chapter 4](../ch04-numerical-linear-algebra/index.md) and the vanishing LODF denominator of
    [Chapter 17](../../part-3-power-systems/ch17-sensitivity-factors-1/index.md). The three branches at 0.7629 and
    the two at 0.7467 are the radial ring 6–11–10–9–14–13–6, the weakest loop of the grid.
4. **`find_bridges.py` in ToOp.** `find_bridges(from_node, to_node, number_of_branches, number_of_nodes)` first
   builds an `nx.MultiGraph` whose edges are keyed by the branch index (`_get_graph_with_branch_keys`), the same
   multigraph model as step 2, so parallel circuits are never bridges. It then calls `nx.bridges(graph_nx)`,
   turns the list of `(from, to)` pairs into an integer array, appends the reversed pairs (NetworkX returns each
   bridge in one orientation only, and a branch may be stored either way), and intersects that array with the
   `[from_node, to_node]` columns of the branch table through a structured view and `np.intersect1d(...,
   return_indices=True)`. The matched row indices set `True` in a Boolean mask `branch_is_bridge` of length
   `number_of_branches`, which is the output: a bridge flag per branch, used to drop those branches from the N-1
   outage set. The same file has `find_n_minus_2_safe_branches`, which removes each branch in turn from the
   keyed multigraph and counts the bridges that appear (`get_number_of_bridges_after_outage`), the N-2 version of
   the same test.
5. **Station coupler states.** With couplers `(0,1), (1,2), (2,3), (3,0)`, `itertools.product((0, 1), repeat=4)`
   gives 16 states; skipping the all-closed one and keeping the states whose closed-coupler graph has exactly two
   connected components leaves 6 states, all with exactly two couplers open, realizing 6 distinct bipartitions
   (`two_busbar_states` and `count_states_and_bipartitions` in `station_splits.py`):

    | Open couplers | Busbars |
    |---|---|
    | (0,1), (1,2) | {1} \| {0, 2, 3} |
    | (1,2), (2,3) | {2} \| {0, 1, 3} |
    | (2,3), (3,0) | {3} \| {0, 1, 2} |
    | (3,0), (0,1) | {0} \| {1, 2, 3} |
    | (0,1), (2,3) | {0, 3} \| {1, 2} |
    | (1,2), (3,0) | {0, 1} \| {2, 3} |

    Opening one coupler leaves a path (one component), opening three leaves three components. With the extra
    coupler `(0, 2)` there are 32 states, 12 of them with two components, but still the same 6 bipartitions: the
    diagonal gives new ways to realize `{1} | {0, 2, 3}` (four states) and `{3} | {0, 1, 2}` (four states), and no
    new split, because `{0, 2} | {1, 3}` still has a disconnected side ([Exercise 4](#exercise-4)). This is what
    `make_separation_set` does for real stations: enumerate coupler states, keep the two-component ones, and
    collapse them to distinct electrical bipartitions.
6. **IEEE 118-bus stretch.** `ieee118.py` exports the 186 branches (lines and transformers, bus numbers from 1)
   of `pandapower.networks.case118()`. The same functions give 9 bridges, `(8, 9), (9, 10), (12, 117), (68, 116),
   (71, 73), (85, 86), (86, 87), (110, 111), (110, 112)`, and 9 articulation points, buses 8, 9, 12, 68, 71, 85,
   86, 100 and 110 (bus 100 is a cut vertex without a bridge: its loss islands a block of ten buses that hangs
   off it through two branches). The Laplacian
   has one zero eigenvalue, exactly 9 branches have $R_{\text{eff}} = 1$, the bridges again, and the algebraic
   connectivity is $0.0282$, far below the 14-bus value $0.4584$: the bigger grid is sparser and has long radial
   tails.

## Exercise 1

Label the triangles $\{0, 1, 2\}$ and $\{3, 4, 5\}$ and join them by the edge $(2, 3)$. By hand: every edge of a
triangle lies on a cycle, so it is not a bridge; $(2, 3)$ lies on no cycle (any cycle through it would need a
second path between the triangles), so it is the only bridge, and its two ends, vertices 2 and 3, are the
articulation points (removing either leaves the other triangle cut off). Tarjan's DFS from vertex 0, visiting
neighbours in increasing order, gives (`tarjan.py`):

| Vertex | disc | low | How low is obtained |
|---|---|---|---|
| 0 | 0 | 0 | root |
| 1 | 1 | 0 | through child 2 |
| 2 | 2 | 0 | back edge 2–0 |
| 3 | 3 | 3 | through child 4, no back edge above 3 |
| 4 | 4 | 3 | through child 5 |
| 5 | 5 | 3 | back edge 5–3 |

The DFS tree is the path 0–1–2–3–4–5 with back edges 2–0 and 5–3. The tree edge $(2, 3)$ has
$\text{low}[3] = 3 > \text{disc}[2] = 2$: no vertex in the subtree of 3 can reach above 3, so $(2, 3)$ is a
bridge. Vertex 2 is an articulation point because it has a child, 3, with $\text{low}[3] \ge \text{disc}[2]$;
vertex 3 because $\text{low}[4] = 3 \ge \text{disc}[3] = 3$. The root 0 has one child, so it is not one. NetworkX
agrees: `nx.bridges` gives `[(2, 3)]` and `nx.articulation_points` gives `{2, 3}`.

## Exercise 2

Let $e = uv$ be an edge of a connected graph $G$.

*Bridge $\Rightarrow$ no cycle.* Suppose $e$ lies on a cycle $C$. Then $C - e$ is a $u$–$v$ path that avoids
$e$. Any walk in $G$ that used $e$ can replace that step by $C - e$, so every pair of vertices connected in $G$
is still connected in $G - e$: $e$ is not a bridge. By contraposition, a bridge lies on no cycle.

*No cycle $\Rightarrow$ bridge.* If $u$ and $v$ were still connected in $G - e$ by a path $P$, then $P + e$
would be a cycle through $e$. So if $e$ lies on no cycle, $u$ and $v$ are disconnected in $G - e$, which makes
$e$ a bridge.

*Exactly two components.* Let $w$ be any vertex and $P$ a $w$–$u$ path in $G$ (it exists, $G$ is connected). If
$P$ avoids $e$, $w$ is in the component of $u$ in $G - e$. Otherwise follow $P$ from $w$ until the first endpoint
of $e$ it meets; that initial segment avoids $e$, so $w$ is in the component of $u$ or in the component of $v$.
Hence $G - e$ has at most two components, and by the previous paragraph $u$ and $v$ lie in different ones:
exactly two. This is why the outage of a bridge always creates exactly one island, and why the Laplacian of
step 3 gains exactly one zero eigenvalue.

## Exercise 3

The algebraic connectivity of the intact 14-bus grid is $\lambda_2 = 0.4584$ (`algebraic_connectivity`, in
agreement with `nx.algebraic_connectivity`). Removing each branch in turn (`connectivity_after_removal`,
bar chart with `plot_connectivity`):

| Removed branch | $\lambda_2$ | Drop | Removed branch | $\lambda_2$ | Drop |
|---|---|---|---|---|---|
| 7–8 | 0.0000 | 0.4584 | 6–12 | 0.3930 | 0.0655 |
| 5–6 | 0.2017 | 0.2567 | 1–5 | 0.4173 | 0.0411 |
| 9–10 | 0.2611 | 0.1973 | 3–4 | 0.4292 | 0.0292 |
| 9–14 | 0.2624 | 0.1960 | 4–5 | 0.4301 | 0.0284 |
| 7–9 | 0.3014 | 0.1571 | 2–5 | 0.4343 | 0.0241 |
| 6–11 | 0.3116 | 0.1468 | 6–13 | 0.4398 | 0.0187 |
| 4–7 | 0.3160 | 0.1424 | 12–13 | 0.4459 | 0.0126 |
| 13–14 | 0.3307 | 0.1277 | 2–3 | 0.4486 | 0.0098 |
| 10–11 | 0.3543 | 0.1042 | 1–2 | 0.4569 | 0.0015 |
| 4–9 | 0.3735 | 0.0849 | 2–4 | 0.4583 | 0.0001 |

The bridge is in a class of its own: removing it disconnects the grid and $\lambda_2$ drops to exactly zero
(the second zero eigenvalue of step 3). Every other removal leaves a connected grid with $\lambda_2 > 0$. The
worst non-bridge is 5–6: the cut between the western buses 1–5 and the eastern buses 6–14 has only three
branches, 5–6, 4–7 and 4–9, and 5–6 is the one whose removal leaves the two sides most weakly joined. The next
ones, 9–10, 9–14, 7–9, 6–11, 4–7 and 13–14, are on the weak ring 6–11–10–9–14–13 and its connections to bus 4,
the same branches that had the largest effective resistances in step 3. Branches of the strongly meshed western
end (2–4, 1–2, 2–3) barely matter: bus 2 keeps three other branches when any one of them goes. Algebraic connectivity measures how well a graph resists being cut, so it ranks the branches that are
"almost bridges" without being one.

## Exercise 4

An assignment is a vector in $\{A, B\}^n$, so there are $2^n$ of them. Swapping the labels A and B maps each
vector to a different one (no vector is its own swap, because every busbar changes side) and pairs the $2^n$
vectors up, so there are $2^n / 2 = 2^{n-1}$ assignments up to swapping. Exactly one class is trivial, "all
busbars on the same side" (the unsplit station), leaving $2^{n-1} - 1$ non-trivial bipartitions:
$8$ and $7$ for the four-busbar station (`all_bipartitions` in `station_splits.py`).

The lab found only 6 bipartitions reachable by opening couplers, so one of the 7 is unreachable:
$\{0, 2\} \mid \{1, 3\}$ (`unreachable_bipartitions`). Opening couplers can only produce sides that are
connected through closed couplers, because a side is an electrical busbar: everything on it must be joined by
closed couplers. Busbars 0 and 2 are not adjacent in the ring, so to be on one side they need the path 0–1–2
or 0–3–2 closed, which drags busbar 1 or 3 along; the same holds for 1 and 3. Adding the diagonal coupler
$(0, 2)$ makes $\{0, 2\}$ connectable but leaves $\{1, 3\}$ disconnected, so the same bipartition stays
unreachable and the count stays at 6. In general a bipartition is reachable exactly when both sides induce
connected subgraphs of the coupler graph; this is the condition `make_separation_set` checks when it keeps
only the coupler states with exactly two components.

## Exercise 5

With Boolean vectors `a` and `b` of equal length, the Hamming distance is `np.count_nonzero(a != b)`
(`hamming_distance` in `hamming.py`). For `current = [1, 1, 0, 0, 1, 0]` and `target = [0, 0, 1, 1, 0, 0]` the
plain distance is 5, while the distance to the inverted target `~target = [1, 1, 0, 0, 1, 1]` is 1.

The labels A (`True`) and B (`False`) are arbitrary names for the two busbars of a split station: assigning the
assets to (A, B) or to (B, A) produces the same electrical topology, the one of [Exercise 4](#exercise-4) "up to
swapping". A distance that took the labels literally would say the two topologies above are 5 switching
operations apart, when they differ by moving one asset. ToOp therefore computes the reassignment distance
for the target and for its inverse and keeps the smaller one: in `per_station_switching_distance`
(`dc_solver/preprocess/helpers/switching_distance.py`) `reassignment_distance` and
`reassignment_distance_inv` are both evaluated, `invert = reassignment_distance_inv.min() <
reassignment_distance.min()` selects the orientation, and the function returns `invert` along with the distance
so the caller knows which labelling it used. `switching_distance` in `hamming.py` does the same and returns
`(1, True)` for the example.

## Exercise 6

`nx.single_source_dijkstra_path_length(G, 5, cutoff=2)` on the 14-bus graph with unit weights
(`k_hop_neighbourhood`) returns 11 buses with their distance:

| Distance | Buses |
|---|---|
| 0 | 5 |
| 1 | 1, 2, 4, 6 |
| 2 | 3, 7, 9, 11, 12, 13 |

Buses 8, 10 and 14 are three hops away and are cut off by `cutoff=2`: the algorithm stops expanding a node as
soon as its tentative distance exceeds the cutoff, so it never visits the far side of the grid. This is the call
ToOp's importer uses, with real branch weights, to find the bays and couplers around a busbar without walking
the whole network (`importer/network_graph/`).
