# Chapter 8 – Solutions

Solutions of the Lab and the Exercises of [Chapter 8](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`nodal.py`](src/nodal.py) (the nodal solver and the lab network;
`python nodal.py` prints every number of the Lab answers and of Exercises 4 and 5), [`wheatstone.py`](src/wheatstone.py)
(`python wheatstone.py` prints those of Exercises 1 and 3) and [`mesh_nodal.m`](src/mesh_nodal.m) (Exercise 2,
Octave). The notebooks [`examples.ipynb`](src/examples.ipynb), [`lab.ipynb`](src/lab.ipynb) and
[`exercises.ipynb`](src/exercises.ipynb) import these modules. The ToOp function of Exercise 6 was read at
commit `ba1874e`.

Node numbers start at 0 and branch $k$ runs from `fr[k]` to `to[k]`; a positive branch current flows that
way. The lab network is the one of the [Chapter 2](../../part-1-math/ch02-linear-algebra/index.md) lab: branches
$(0,1)$, $(0,2)$, $(1,2)$, $(1,3)$, $(2,4)$, $(3,4)$, $(4,5)$ with conductances $10, 5, 8, 4, 6, 3, 2$ S.

## Lab

1. **`nodal_solve`.** The incidence matrix $A$ has one row per branch with $+1$ in the `fr` column and $-1$ in
   the `to` column; $A^\top \operatorname{diag}(g) A$ is the weighted Laplacian of the network, singular because
   its rows sum to zero. Deleting the ground row and column leaves the invertible $G$ of the chapter; the
   solver fills the ground voltage with 0 and returns $g \odot (Av)$ as branch currents. The injection given
   for the ground node is not used: ground supplies whatever balances the others. On the worked example it
   returns $v = (0, 1.2857, 0.4286)$ V $= (0, 18/14, 6/14)$ and the branch currents $1.2857$ A through the 1 S
   branch and $1.7143$ A through the 2 S and 4 S branches, which add up to the 3 A of the source.

    ```python title="nodal.py: nodal_solve"
    --8<-- "docs/part-2-circuits/ch08-dc-circuits/src/nodal.py:nodal_solve"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-2-circuits/ch08-dc-circuits/src){ target="_blank" } to see or download all the files of this chapter.

2. **The Chapter 2 network.** With $0.5$ A into node 2, $1$ A out of node 5 and node 0 grounded, the voltages
   are $v = (0, -0.03041, -0.03918, -0.08889, -0.16686, -0.66686)$ V and the branch currents, in branch order,
   $(0.30409, 0.19591, 0.07018, 0.23392, 0.76608, 0.23392, 1.00000)$ A. KCL: $A^\top i$ equals the injection
   at every node to $10^{-12}$ once the ground injection is written as the $+0.5$ A it must supply (the two
   branches leaving node 0 carry $0.30409 + 0.19591 = 0.5$ A). Power balance: the sources deliver
   $v^\top i = 0.5\,(0) + 0.5\,(-0.03918) - 1\,(-0.66686) = 0.647271$ W and the branches dissipate
   $\sum_k g_k (Av)_k^2 = \sum_k i_k^2 / g_k = 0.647271$ W, the same number ([Exercise 3](#exercise-3) proves
   it must be).

3. **Superposition.** Solving with the $0.5$ A alone and with the $-1$ A alone and adding the two voltage
   vectors reproduces the voltages of step 2 to $3\cdot10^{-17}$ (`superposition_check`). The solution map
   $i \mapsto v = G^{-1} i$ is linear, so it respects sums; this is the property that lets
   [Chapter 17](../../part-3-power-systems/ch17-sensitivity-factors-1/index.md) build flows from one
   sensitivity per injection and add them.

4. **Thévenin.** Injecting 1 A at node 5 and withdrawing it at node 0 gives $v_5 - v_0 = 0.71189\ \Omega$
   (`thevenin_resistance`, one linear solve: the second solve of the chapter's "two linear solves" is only
   needed when the open-circuit voltage is wanted too). The Chapter 3 effective resistance
   $(e_5 - e_0)^\top L^+ (e_5 - e_0)$ with the pseudoinverse of the full Laplacian is $0.71189\ \Omega$ too,
   to $10^{-12}$: they are the same quantity. Sanity check: node 5 hangs off node 4 through the single 2 S
   branch ($0.5\ \Omega$), so the Thévenin resistance must exceed $0.5\ \Omega$; the rest of the network
   between node 4 and ground adds $0.21189\ \Omega$.

    ```python title="nodal.py: Thévenin and effective resistance"
    --8<-- "docs/part-2-circuits/ch08-dc-circuits/src/nodal.py:thevenin"
    ```

5. **Split a node.** `split_node_2` moves branch $(2, 4)$ to a new node 6 (busbar B), keeps $(0, 2)$, $(1, 2)$
   and the $0.5$ A on node 2 (busbar A) and adds the coupler branch $(2, 6)$ with $10^6$ S. The coupler
   current is $0.76608$ A from A to B, across a drop of $7.7\cdot10^{-7}$ V. From the **unsplit** solution,
   the imbalance at A is the flow into A over its two branches plus the injection at A:
   $0.19591 + 0.07018 + 0.5 = 0.76608$ A, and the two agree to $10^{-6}$. They must: in the unsplit network
   KCL at node 2 says that whatever enters through $(0, 2)$, $(1, 2)$ and the source leaves through $(2, 4)$,
   and after the split the only way from A to $(2, 4)$ is the coupler. When the coupler is ideal (zero
   voltage drop) the split network has the same voltages as the unsplit one, so every other current is
   unchanged and the coupler simply makes the hidden internal current of node 2 visible. This is the number
   ToOp computes in `_compute_bus_a_imbalance` ([Exercise 6](#exercise-6)).

    ```python title="nodal.py: split_node_2"
    --8<-- "docs/part-2-circuits/ch08-dc-circuits/src/nodal.py:split"
    ```

6. **Keep `nodal_solve`.** With susceptances $b_k = 1/x_k$ in place of conductances, angles in place of
   voltages and power injections in place of currents, the same function is the DC power flow
   $B\theta = P$ of [Chapter 16](../../part-3-power-systems/ch16-dc-power-flow/index.md). Nothing to
   compute here.

## Exercise 1

Number the nodes 0 (bottom, ground), 1 (top), 2 (left) and 3 (right); $R_1$ joins 1–2, $R_2$ joins 1–3,
$R_3$ joins 2–0, $R_4$ joins 3–0 and $R_5$ is the middle branch 2–3; the source pushes $I$ into node 1 and
takes it from node 0. With $g_k = 1/R_k$, KCL at the three ungrounded nodes reads

$$
\begin{bmatrix}
g_1 + g_2 & -g_1 & -g_2 \\
-g_1 & g_1 + g_3 + g_5 & -g_5 \\
-g_2 & -g_5 & g_2 + g_4 + g_5
\end{bmatrix}
\begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}
=
\begin{bmatrix} I \\ 0 \\ 0 \end{bmatrix}.
$$

The middle current is $g_5 (v_2 - v_3)$, zero exactly when $v_2 = v_3$. Put $v_2 = v_3 = w$ in the second and
third equations: the $g_5$ terms cancel and they become $g_1 v_1 = (g_1 + g_3)\, w$ and
$g_2 v_1 = (g_2 + g_4)\, w$, which are compatible only if $g_1 (g_2 + g_4) = g_2 (g_1 + g_3)$, that is
$g_1 g_4 = g_2 g_3$, or in resistances

$$
R_1 R_4 = R_2 R_3 \qquad \Big(\tfrac{R_1}{R_3} = \tfrac{R_2}{R_4}\Big).
$$

Both arms then divide $v_1$ in the same ratio, $w = v_1 R_3/(R_1 + R_3) = v_1 R_4/(R_2 + R_4)$, and the
condition does not involve $R_5$ or $I$: a balanced bridge stays balanced for any middle resistor and any
source. Numeric check with `bridge_currents`: $R = (100, 200, 50, 100, 1000)\ \Omega$ satisfies
$R_1 R_4 = R_2 R_3 = 10^4$ and gives $v_2 - v_3 = 0$ and a middle current of $0$ to $10^{-12}$; changing $R_4$ to
$120\ \Omega$ gives $v_2 - v_3 = -3.839$ V and $-3.839$ mA through the middle branch.

```python title="wheatstone.py: bridge"
--8<-- "docs/part-2-circuits/ch08-dc-circuits/src/wheatstone.py:bridge"
```

## Exercise 2

Resistances $R_a = 1\ \Omega$ (node 1 to ground), $R_b = 1/2\ \Omega$ (nodes 1–2), $R_c = 1/4\ \Omega$
(node 2 to ground). Take mesh 1 as the loop of the current source and $R_a$, and mesh 2 as the loop
$R_a$–$R_b$–$R_c$, both clockwise. The source fixes $I_1 = 3$ A; KVL around mesh 2,
$R_a (I_2 - I_1) + R_b I_2 + R_c I_2 = 0$, gives $I_2 = R_a I_1 / (R_a + R_b + R_c) = 3 / 1.75 = 12/7 =
1.7143$ A. Then $v_1 = R_a (I_1 - I_2) = 9/7 = 1.2857$ V and $v_2 = R_c I_2 = 3/7 = 0.4286$ V, the voltages of
the worked example ($18/14$ and $6/14$). The mesh currents are the branch currents of step 1: $I_1 - I_2 =
1.2857$ A in the 1 S branch and $I_2 = 1.7143$ A in the series pair. In Octave `mesh_nodal` solves
$G v = i$ and the mesh system $R\, I = (I_s, 0)$ and returns both; the Octave test block checks that they
agree to $10^{-12}$. Nodal analysis needs one equation per ungrounded node (2), mesh analysis one per
independent loop (2 here, with one trivial because of the current source); for grids with thousands of buses
and a current-source model of injections the nodal form wins, which is why power flow is nodal.

```octave title="mesh_nodal.m"
--8<-- "docs/part-2-circuits/ch08-dc-circuits/src/mesh_nodal.m"
```

## Exercise 3

$u^\top j = (Av)^\top j = v^\top (A^\top j) = v^\top i$: two lines, using only $u = Av$ (KVL, the branch
voltages are differences of node potentials) and $A^\top j = i$ (KCL). Nothing relates $u$ to $j$: they may
come from different circuits on the same graph, and the theorem still holds. For the power balance take
$j = i_\text{branch} = \operatorname{diag}(g)\, Av$ of the same network: $u^\top j = (Av)^\top
\operatorname{diag}(g)\, (Av) = \sum_k g_k (Av)_k^2$, the Joule power, while $v^\top i$ is the power delivered
by the sources (ground included, at potential 0). So $\sum_k g_k (Av)_k^2 = v^\top i$, which is lab step 2.
`tellegen_check` runs the stronger statement: the voltages of the lab network ($g = 10, 5, 8, 4, 6, 3, 2$ S)
and the currents of a unit-conductance network with the same injections give $u^\top j = 0.647271$ W $=
v^\top i$; with random conductances for the second network the two sides still agree to $10^{-12}$.

```python title="wheatstone.py: tellegen_check"
--8<-- "docs/part-2-circuits/ch08-dc-circuits/src/wheatstone.py:tellegen"
```

## Exercise 4

Let $L = A^\top \operatorname{diag}(g) A$ be the full Laplacian and $i$ a balanced injection vector
($\mathbf 1^\top i = 0$, the ground node's share included). Grounding node $r$ means solving $Lv = i$ with
$v_r = 0$; grounding node $s$ means solving the same $Lv' = i$ with $v'_s = 0$. The null space of $L$ of a
connected network is spanned by $\mathbf 1$ ([Chapter 2](../../part-1-math/ch02-linear-algebra/index.md)), so
every solution of $Lv = i$ is $v + c\mathbf 1$ for one constant $c$: $v' = v - v_s \mathbf 1$. All voltages
shift by the same constant and $u = Av' = Av - v_s A\mathbf 1 = Av$ because each row of $A$ sums to zero, so
every branch current $g \odot u$ is unchanged. Check with `nodal_solve` (ground 0, then 3): the voltages
become $(0.08889, 0.05848, 0.04971, 0, -0.07797, -0.57797)$ V, each exactly $0.08889$ V $= -v_3$ above the
ground-0 solution, and the branch currents agree to $2\cdot10^{-16}$. The condition "balanced" matters: the
solver ignores the injection at the ground node, so if the injections of the other nodes do not sum to the
negative of the ground's, grounding another node changes the problem (a different node then supplies the
balance) and the currents do change. That is why `INJ` in `nodal.py` lists the $+0.5$ A of node 0.

## Exercise 5

| Coupler conductance | Coupler current | Voltage across the coupler | Departure from $0.76608$ A |
|---|---|---|---|
| $10^6$ S | $0.76608$ A | $7.7\cdot10^{-7}$ V | $< 10^{-6}$ |
| $10^3$ S | $0.76517$ A | $7.7\cdot10^{-4}$ V | $-0.12\ \%$ |
| $10$ S | $0.68443$ A | $0.068$ V | $-10.7\ \%$ |

The KCL argument of lab step 5 assumes that A and B are at the same potential, so that the split network has
the unsplit voltages and only the hidden internal current of node 2 becomes visible. A coupler of finite
conductance $g_c$ is a resistor in series with branch $(2, 4)$: the path from A to node 4 now has resistance
$1/6 + 1/g_c$ instead of $1/6\ \Omega$, so less current takes it ($0.684$ A at $10$ S, where the coupler adds
$0.1\ \Omega$ to $0.167\ \Omega$) and more leaves A through $(0, 2)$ and $(1, 2)$; the potentials of every
node move, and the "imbalance at A" of the unsplit solution no longer describes the coupler. At $10^3$ S
the extra $1$ m$\Omega$ is $0.6\ \%$ of the path and the error is $0.12\ \%$; at $10^6$ S it is invisible.
ToOp's update after a split ([Chapter 18](../../part-3-power-systems/ch18-sensitivity-factors-2/index.md)) uses
the unsplit flows and the imbalance formula, which is exact only for a **closed, ideal** coupler: a closed
coupler in a real substation is a busbar section with a negligible impedance, so the assumption holds; an
open coupler is the opposite case (the two busbars become two buses with no flow between them), handled as
a different topology, not as a weak coupler. `split_node_2(coupler_g)` prints the table.

## Exercise 6

`_compute_bus_a_imbalance` in `toop_engine_dc_solver/jax/cross_coupler_flow.py` (ToOp commit `ba1874e`)
computes, per timestep,

```text
imbalance_a = sum(flows[branch_to_a]) - sum(flows[branch_from_a]) + bus_a_injection
```

and returns it as the cross-coupler flow, zeroing it when the substation index is out of range or when the
configuration mask puts no branch on busbar A (`jnp.all(~substation_configuration)`). Term by term against lab
step 5:

| ToOp | Lab step 5 |
|---|---|
| `flows` (timesteps × branches, "potentially already updated by previous splits") | the branch currents $i = g \odot (Av)$ of the **unsplit** solution |
| `branch_to_a`: branches whose flow direction is towards bus A | branches $(0, 2)$ and $(1, 2)$, whose `to` end is node 2: $0.19591 + 0.07018$ A |
| `branch_from_a`: branches whose flow direction is away from bus A | none here (branch $(2, 4)$ leaves from node 2 but is assigned to busbar B) |
| `bus_a_injection`: net power injection assigned to bus A | the $0.5$ A kept on busbar A |
| `substation_configuration` (`True` = branch on bus A) | the choice $\{(0, 2), (1, 2)\} \to$ A, $\{(2, 4)\} \to$ B |
| `imbalance_a` | the coupler current $0.76608$ A from A to B |
| `.get(mode='fill', fill_value=0.0)` on padded indices | substations have different numbers of branches; the padding entries contribute 0 |

"Towards A" and "away from A" refer to the orientation of each branch (its `fr`/`to` convention), not to the
sign of the flow, exactly as the lab sums signed currents $i_k$ with the sign of $A$. The function works in
power (MW) where the lab works in amperes, and over many timesteps at once (the `axis=1` sums), but the
formula is the KCL of lab step 5 and nothing more: with the flows of the unsplit grid it is exact for a closed
ideal coupler ([Exercise 5](#exercise-5)), and [Chapter 18](../../part-3-power-systems/ch18-sensitivity-factors-2/index.md)
shows how this one number then updates every other flow.
