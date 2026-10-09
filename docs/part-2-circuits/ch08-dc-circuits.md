# Chapter 8. DC circuits and nodal analysis in matrix form

**Weeks:** 2.5 · **Semester:** 2 · **Prerequisites:** [Chapter 2](../part-1-math/ch02-linear-algebra.md), [Chapter 3](../part-1-math/ch03-graph-theory.md), [Chapter 7](ch07-electromagnetism.md)

!!! info "Why it matters for ToOp"

    The DC power flow is nodal analysis of a network of reactances: $B\theta = P$ has the same structure as
    $Gv = i$ for a resistor network. Superposition is why flows can be updated by adding contributions. When a
    substation is split into busbars A and B, the flow across the coupler is obtained from Kirchhoff's current
    law alone: it equals the **imbalance at busbar A**, the sum of the flows into A minus the flows out of A
    plus the injection assigned to A (`_compute_bus_a_imbalance` in `dc_solver/jax/cross_coupler_flow.py`).
    [Chapter 18](../part-3-power-systems/ch18-sensitivity-factors-2.md) shows how ToOp uses that number to update
    all flows after a split.

## Topics

- Circuit elements; KCL and KVL; series and parallel.
- **Nodal analysis and the nodal admittance (conductance) matrix; mesh analysis.** The nodal conductance
  matrix $G = A^\top \operatorname{diag}(g) A$ is the weighted Laplacian of
  [Chapter 2](../part-1-math/ch02-linear-algebra.md) and [Chapter 3](../part-1-math/ch03-graph-theory.md),
  revisited here with physical units.
- Linearity and superposition; Thévenin and Norton equivalents; maximum power transfer.
- **Network topology: graphs of circuits, trees and cotrees, incidence matrix formulation, Tellegen's
  theorem.**

## Learning outcomes

- Write the nodal equations of any resistive network directly in matrix form $Gv = i$.
- Show that $G$ is a weighted Laplacian and explain the role of the reference (ground) node.
- Compute a Thévenin equivalent between two nodes with two linear solves.
- Compute the current through a coupler joining two parts of a split node from KCL.

## Resources

- Courses:
  - MIT OCW 6.002 *Circuits and Electronics* (Agarwal, Spring 2007):
    <https://ocw.mit.edu/courses/6-002-circuits-and-electronics-spring-2007/>. Lecture videos on
    archive.org, **FREE** [`MIT6.002S07`](https://archive.org/details/MIT6.002S07).
  - Coursera *Linear Circuits 1: DC Analysis* (Georgia Tech): <https://www.coursera.org/learn/linear-circuits-dcanalysis>.
  - NPTEL *Networks and Systems* (V.G.K. Murti, IIT Madras): <https://nptel.ac.in/courses/108106075>.
- Free texts:
  - Ulaby, Maharbiz & Furse, *Circuit Analysis and Design*, 3rd ed. (2025), open access from Michigan
    Publishing: <https://services.publishing.umich.edu/Books/C/Circuit-Analysis-and-Design2>.
  - Kuphaldt, *Lessons in Electric Circuits*, Vol. I DC: <https://www.ibiblio.org/kuphaldt/electricCircuits/DC/>.
- Textbooks (the chapters on resistive circuits, nodal and mesh analysis, and network theorems):
  - Dorf & Svoboda, *Introduction to Electric Circuits* (6 syllabi, the most cited). **PRINT-DISABLED**.
    ES: *Circuitos eléctricos* (Alfaomega, 9th ed. 2016), **USER-UPLOAD (ES)** [`circuitos-electricos-dorf-circuitos-elec`](https://archive.org/details/circuitos-electricos-dorf-circuitos-elec).
  - Nilsson & Riedel, *Electric Circuits* (5 syllabi). **PRINT-DISABLED** (a partial 2015 custom edition is
    **BORROW** [`isbn_9781323502259`](https://archive.org/details/isbn_9781323502259)).
    ES: *Circuitos eléctricos* (Pearson, 7th ed. 2006; cited at FIUBA, U. de Chile, UPC), **USER-UPLOAD (ES)** [`CIRCUITOSELECTRICOSNILSSONYRIEDEL`](https://archive.org/details/CIRCUITOSELECTRICOSNILSSONYRIEDEL).
  - Alexander & Sadiku, *Fundamentals of Electric Circuits* (5 syllabi). **PRINT-DISABLED**.
    ES: *Fundamentos de circuitos eléctricos* (McGraw-Hill; 6th ed. required at UPC), **USER-UPLOAD (ES)** [`fundamentos-de-circuitos-electricos`](https://archive.org/details/fundamentos-de-circuitos-electricos).
  - Hayt, Kemmerly & Durbin, *Engineering Circuit Analysis* (4 syllabi). 2010 Tata McGraw-Hill printing
    **BORROW** [`engineeringcircu0000hayt_n1c5`](https://archive.org/details/engineeringcircu0000hayt_n1c5).
    ES: *Análisis de circuitos en ingeniería* (McGraw-Hill Interamericana, 8th ed. 2012; 7th ed. cited at
    FIUBA and UPC).
  - Irwin & Nelms, *Basic Engineering Circuit Analysis* (4 syllabi). 11th ed. **BORROW**
    [`basicengineering0000jdav`](https://archive.org/details/basicengineering0000jdav).
    ES: *Análisis básico de circuitos en ingeniería* (Limusa; 6th ed. required at UPC).
  - Nahvi & Edminister, *Schaum's Outline of Electric Circuits* (3 syllabi), for problem practice. 6th ed.
    **BORROW** [`schaumsoutlinese0000nahv_g9d4`](https://archive.org/details/schaumsoutlinese0000nahv_g9d4).
    ES: *Circuitos eléctricos y electrónicos* (Schaum, McGraw-Hill, 2005).
  - Desoer & Kuh, *Basic Circuit Theory* (1969), for graph-theoretic network analysis. **BORROW**
    [`basiccircuittheo0000deso`](https://archive.org/details/basiccircuittheo0000deso).
- Documentation and tools: GNU Octave "Linear Algebra" <https://docs.octave.org/latest/Linear-Algebra.html>;
  NumPy `numpy.linalg` <https://numpy.org/doc/stable/reference/routines.linalg.html>.

## Worked example

Two nodes plus ground. Conductances: 1 S from node 1 to ground, 2 S between nodes 1 and 2, 4 S from node 2 to
ground; a 3 A current source feeds node 1. KCL at each node, written by inspection (diagonal: sum of the
conductances at the node; off-diagonal: minus the conductance between the nodes):

$$
\underbrace{\begin{bmatrix} 1 + 2 & -2 \\ -2 & 2 + 4 \end{bmatrix}}_{G}
\begin{bmatrix} v_1 \\ v_2 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 0 \end{bmatrix}
\quad\Rightarrow\quad
v_1 = \frac{18}{14} \approx 1.286\ \text{V},\ \ v_2 = \frac{6}{14} \approx 0.429\ \text{V}.
$$

This $G$ is exactly $A^\top \operatorname{diag}(g) A$ with the ground row and column deleted, where $A$ is the
incidence matrix of the three branches over the nodes $\{0, 1, 2\}$.

## Lab

!!! example "Lab: a nodal solver you will reuse"

    Tools: Jupyter + NumPy (+ pandas for the branch table), or Octave.

    1. Write `nodal_solve(fr, to, g, inj, n, ground=0)` that builds $A$ and $G = A^\top \operatorname{diag}(g) A$,
       deletes the ground row and column, solves for the node voltages and returns them with the branch
       currents $g \odot (Av)$. Check it on the worked example.
    2. Use the network of the [Chapter 2](../part-1-math/ch02-linear-algebra.md) lab: 0.5 A injected at node 2,
       1 A withdrawn at node 5, and the ground node supplying the balance. Verify KCL at every node and the
       power balance $v^\top i = \sum_k g_k (Av)_k^2$.
    3. Superposition: solve for each source separately and check that the voltages add up.
    4. Thévenin: inject 1 A at node 5 and withdraw it at node 0; the voltage difference is the Thévenin
       resistance between those nodes. Compare with the effective resistance of
       [Chapter 3](../part-1-math/ch03-graph-theory.md).
    5. Split a node: in the same network, split node 2 into busbar A (keeping branches `(0, 2)`, `(1, 2)` and the
       0.5 A injection) and busbar B (a new node 6 that takes branch `(2, 4)`), joined by a coupler branch
       `(2, 6)` with a very large conductance ($10^6$ S). Solve the split network and read the coupler current.
       Compare it with the imbalance at A computed from the **unsplit** solution: flows into A over its
       branches plus the injection at A (expected: both about 0.766 A). This is the quantity computed in
       `dc_solver/jax/cross_coupler_flow.py`.
    6. Keep `nodal_solve`: it becomes the DC power flow in
       [Chapter 16](../part-3-power-systems/ch16-dc-power-flow.md) by replacing conductances with susceptances.

## Exercises

1. By hand, write the nodal equations of a Wheatstone bridge (4 nodes, 5 resistors, one current source) and
   find the condition on the resistors for zero current in the middle branch. *(Pen and paper)*
2. Solve the circuit of the worked example by mesh analysis and check that you get the same voltages. Then
   solve both formulations in Octave. *(Pen and paper, then Octave)*
3. Tellegen's theorem: for any branch voltages $u = Av$ and any branch currents $j$ that satisfy KCL
   ($A^\top j = i$), show that $u^\top j = v^\top i$. Use it to prove the power balance of lab step 2.
   *(Pen and paper)*
4. Show that grounding a different node changes all node voltages by the same constant and leaves every
   branch current unchanged. Check it with `nodal_solve`. *(Pen and paper, then Jupyter: NumPy)*
5. In lab step 5, lower the coupler conductance to $10^3$ S and to $10$ S. How does the coupler current depart
   from the KCL imbalance of the unsplit network, and why does the approximation need a closed, near-ideal
   coupler? *(Jupyter: NumPy)*
6. Read `_compute_bus_a_imbalance` in ToOp's `dc_solver/jax/cross_coupler_flow.py` and map each term of its
   formula to the quantities of lab step 5. *(Code reading)*
