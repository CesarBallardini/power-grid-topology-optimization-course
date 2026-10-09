# Chapter 29. Learning-based topology control

**Weeks:** 2 · **Semester:** 5 · **Prerequisites:** [Chapter 5](../part-1-math/ch05-probability.md), [Chapter 24](ch24-evolutionary-algorithms.md), [Chapter 28](ch28-topology-optimization.md), [Chapter 31](../part-5-computing/ch31-jax-gpu-fundamentals.md)

!!! info "Why it matters for ToOp"

    ToOp is a search-based optimizer: it contains no learned policy, no neural surrogate and no gradients. A
    search of its packages finds no `jax.grad`, `jax.value_and_grad`, `jax.jacfwd`, `jax.jacrev`, `jax.vjp` or
    `jax.jvp`; JAX is used for batching, compilation and GPU execution. Yet most of the recent literature on
    topology control is learning-based, and ToOp shares its vocabulary:

    - ToOp's DC solver documentation introduces its topology encoding through "the topo-vect format as introduced
      by grid2ops" (`docs/dc_solver/quickstart.md`), the representation of Grid2Op, the environment of the L2RPN
      reinforcement learning competitions.
    - The example-grid generator can write Grid2Op-compatible chronics (`save_grid2op_compatible` in
      `dc_solver/example_grids.py`).
    - The DC-screening plus AC-validation design of [Chapter 28](ch28-topology-optimization.md) is a
      surrogate-assisted search: a cheap model filters candidates for an expensive one. Neural surrogates and RL
      policies are alternative ways to propose or rank candidates, and they need the same kind of simulation check
      before an operator sees them.

    This chapter gives you enough reinforcement learning, graph learning and automatic differentiation to read
    that literature critically, to benchmark a learned agent against search, and to design hybrids.

## Topics

- **Reinforcement learning fundamentals:** Markov decision processes $(\mathcal S, \mathcal A, P, R, \gamma)$;
  return $G_t = \sum_{k \ge 0} \gamma^k R_{t+k+1}$; policies; value functions $V^\pi$ and $Q^\pi$ and the Bellman
  equations; dynamic programming; temporal-difference learning and tabular Q-learning,
  $Q(s,a) \leftarrow Q(s,a) + \alpha\,[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$; function approximation and DQN
  (replay buffer, target network); policy gradient and actor–critic methods; PPO at intuition level (clipped
  policy updates); exploration ($\varepsilon$-greedy, entropy bonus).
- **Topology control as RL:** Grid2Op environments, observations (line loadings `rho`, `topo_vect`), actions
  (bus assignments, line status), rewards and game over; the L2RPN competitions; baselines (do-nothing,
  rule-based "expert" agents); multi-objective RL for topology control (revisits
  [Chapter 27](ch27-pareto-multi-objective.md)).
- **The action-space size problem:** bus assignments grow exponentially with the number of elements per
  substation; reductions by unitary action sets, action pruning by simulation, hierarchical agents; the
  connection with ToOp's action enumeration and validity filters
  ([Chapter 22](../part-3-power-systems/ch22-import-preprocessing.md)).
- **Graph neural network power-flow surrogates:** the grid as a graph, message passing, the graph neural solver
  of Donon et al., generalization to unseen topologies, accuracy vs speed, physics-based losses.
- **Surrogate-assisted evolutionary search:** pre-screening, model management, uncertainty; comparison with
  ToOp's DC/AC multi-fidelity design.
- **Automatic differentiation:** forward and reverse mode, JVPs and VJPs; differentiable DC power flow (the
  Jacobian of flows with respect to injections is the PTDF of
  [Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md)); why discrete switching decisions are not
  differentiable and what relaxations do about it (awareness).
- **Limits and safety of learned controllers:** distribution shift, rare contingencies, no feasibility or
  optimality guarantees, reward misspecification, interpretability; why TSOs pair learned proposals with
  simulation checks (Grid2Op's `obs.simulate`, ToOp's AC validation) and keep a human in the loop.

## Learning outcomes

- Formulate grid topology control as an MDP, implement tabular Q-learning for a small Grid2Op environment, and
  compare it with do-nothing, random and greedy-by-simulation baselines over repeated seeds.
- Explain why the topology action space defeats naive RL and name three reduction techniques.
- Train a small neural surrogate of DC flows on NumPy-generated data and report its error and speed-up against
  the exact solver.
- Use JAX automatic differentiation to recover the PTDF from a differentiable DC power flow, and argue why ToOp's
  discrete search does not use gradients.

## Resources

- Free texts and courses:
  - Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018): Ch. 3 (finite MDPs), Ch. 4
    (dynamic programming), Ch. 6 (temporal-difference learning, Q-learning), Ch. 9 (function approximation),
    Ch. 13 (policy gradient methods). FREE <http://incompleteideas.net/book/the-book-2nd.html>.
  - OpenAI, *Spinning Up in Deep RL*: "Key Concepts in RL" and the PPO page. FREE
    <https://spinningup.openai.com/en/latest/>.
  - Hugging Face, *Deep Reinforcement Learning Course*: units 1–3 (Q-learning, DQN) and the PPO unit. FREE
    <https://huggingface.co/learn/deep-rl-course/unit0/introduction>.
  - Hamilton, *Graph Representation Learning* (Morgan & Claypool, 2020): Ch. 5 "The Graph Neural Network Model"
    and Ch. 6 "Graph Neural Networks in Practice". FREE (pre-publication PDF)
    <https://www.cs.mcgill.ca/~wlh/grl_book/>.
- Papers (reinforcement learning and topology control):
  - Mnih et al., "Human-level control through deep reinforcement learning", *Nature* 518:529–533, 2015,
    doi:10.1038/nature14236 (DQN).
  - Schulman, Wolski, Dhariwal, Radford & Klimov, "Proximal Policy Optimization Algorithms", arXiv:1707.06347.
  - Marot et al., "Learning to run a power network challenge for training topology controllers", *Electric
    Power Systems Research* 189:106635, 2020, doi:10.1016/j.epsr.2020.106635; arXiv:1912.04211. Retrospective:
    Marot et al., "Learning to run a Power Network Challenge: a Retrospective Analysis", arXiv:2103.03104.
  - Lehna, Viebahn, Marot, Tomforde & Scholz, "Managing power grids through topology actions: A comparative
    study between advanced rule-based and reinforcement learning agents", *Energy and AI* 14:100276, 2023,
    doi:10.1016/j.egyai.2023.100276.
  - van der Sar, Zocca & Bhulai, "Optimizing Power Grid Topologies with Reinforcement Learning: A Survey of
    Methods and Challenges", arXiv:2504.08210. **Main survey for this chapter.**
  - Lautenbacher, Rajaei, Barbieri, Viebahn & Cremer, "Multi-Objective Reinforcement Learning for Power Grid
    Topology Control", IEEE PowerTech 2025, doi:10.1109/PowerTech59965.2025.11180236; arXiv:2502.00040.
  - Viebahn, Naglic, Marot, Donnot & Tindemans, "Potential and challenges of AI-powered decision support for
    short-term system operations", CIGRE Paris Session 2022.
- Papers (surrogates and differentiation):
  - Donon, Clément, Donnot, Marot, Guyon & Schoenauer, "Neural networks for power flow: Graph neural solver",
    *Electric Power Systems Research* 189:106547, 2020, doi:10.1016/j.epsr.2020.106547.
  - Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", arXiv:1609.02907.
  - Jin, "Surrogate-assisted evolutionary computation: Recent advances and future challenges", *Swarm and
    Evolutionary Computation* 1(2):61–70, 2011, doi:10.1016/j.swevo.2011.05.001.
  - Baydin, Pearlmutter, Radul & Siskind, "Automatic Differentiation in Machine Learning: a Survey", *JMLR* 18,
    2018, <https://jmlr.org/papers/v18/17-468.html>; arXiv:1502.05767.
- Documentation: Grid2Op <https://grid2op.readthedocs.io/en/latest/>; LightSim2Grid (fast backend)
  <https://lightsim2grid.readthedocs.io/en/latest/>; L2RPN <https://l2rpn.chalearn.org/>; Gymnasium
  <https://gymnasium.farama.org/>; Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>; JAX,
  "Automatic differentiation" <https://docs.jax.dev/en/latest/automatic-differentiation.html>.
- Textbook status: no surveyed syllabus teaches this topic; the texts above are free.

## Worked example

A DC power flow written in JAX is differentiable. Its Jacobian with respect to the non-slack injections is the
PTDF, so automatic differentiation reproduces [Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md)
without deriving anything:

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

lines = jnp.array([[0, 1], [1, 2], [0, 2]])        # 3-bus ring, bus 0 is the slack
b = jnp.array([10.0, 5.0, 8.0])                    # line susceptances
A = jnp.zeros((3, 3)).at[jnp.arange(3), lines[:, 0]].set(1.0).at[jnp.arange(3), lines[:, 1]].set(-1.0)
B_f = b[:, None] * A                               # branch flows = B_f @ theta
B_bus = A.T @ B_f                                  # nodal susceptance matrix

def flows(p_noslack):
    theta = jnp.concatenate([jnp.zeros(1), jnp.linalg.solve(B_bus[1:, 1:], p_noslack)])
    return B_f @ theta

p = jnp.array([-1.0, 0.5])                         # injections at buses 1 and 2
print(flows(p))              # [ 0.6176 -0.3824 -0.1176]
print(jax.jacfwd(flows)(p))  # PTDF columns of buses 1 and 2: [[-0.7647 -0.2941] [0.2353 -0.2941] [-0.2353 -0.7059]]
```

The gradient exists with respect to continuous quantities (injections, susceptances) but not with respect to a
switching decision $z_\ell \in \{0, 1\}$. Relaxing $z_\ell$ to $[0, 1]$ and scaling $b_\ell$ by it gives
gradients, but the optimum of the relaxation need not be near a good binary topology, which is one reason ToOp
searches discrete topologies directly.

## Lab

!!! example "Lab: a learned agent against exhaustive one-step search"

    **Tools:** Jupyter with NumPy, pandas and matplotlib; Grid2Op with the LightSim2Grid backend (CPU only).
    Alternative track: NumPy and JAX only.

    1. **Environment.** Create `grid2op.make("l2rpn_case14_sandbox", test=True, backend=LightSimBackend())`. List the
       number of substations and lines, the observation fields you will use (`rho`, `topo_vect`, `line_status`)
       and the unitary topology actions from `env.action_space.get_all_unitary_topologies_set(env.action_space)`.
    2. **Baselines.** Run do-nothing, a random topology agent and a greedy agent that, whenever the maximum
       `rho` exceeds a threshold, calls `obs.simulate(action)` for every unitary topology action and plays the one
       with the lowest simulated maximum loading (an exhaustive one-step search). Use 10 seeds and episodes of
       288 steps; record survival time, mean maximum loading and wall-clock time per step.
    3. **Tabular Q-learning.** Discretize the state (for example the bin of the maximum `rho` times the index of
       the most loaded line), restrict the actions to do-nothing plus the 20 actions the greedy agent chose most
       often, and train with $\varepsilon$-greedy exploration. Evaluate the greedy policy of the learned $Q$ on the
       same seeds as the baselines and compare with the statistics of
       [Chapter 26](ch26-experimental-methodology.md). Discuss what the agent cannot know that the exhaustive
       search checks at every step.
    4. **Optional.** Replace the table with a small DQN (Stable-Baselines3 through a Gymnasium wrapper of the
       environment) and compare training cost with the greedy agent's simulation cost.

    **Alternative track (surrogate).** Generate 20,000 random injection vectors for IEEE 30, compute DC flows with
    your Chapter 16–17 code, train a two-layer MLP in JAX to predict all flows, and report the relative error
    per line and the speed-up. Then switch off one line not seen in training and measure how the error grows:
    this is the generalization problem that graph neural solvers address.

## Exercises

1. Write the Bellman optimality equation for a two-state, two-action MDP with given transition probabilities and
   rewards, and solve it by value iteration by hand for $\gamma = 0.9$. *(pen and paper)*
2. A substation has 6 elements and 2 busbars. Count the bus assignments (with and without the symmetry of
   swapping the busbars), then the joint actions for 14 such substations. Relate the count to ToOp's per-station
   action sets and to the reduction techniques of this chapter. *(pen and paper)*
3. Extend the worked example to differentiate the flows with respect to the susceptances $b$, and verify one
   column with finite differences. What does this gradient tell an operator? *(Jupyter: JAX + NumPy)*
4. Implement surrogate-assisted pre-screening in your Chapter 24 GA: evaluate every offspring with a cheap
   linear surrogate fitted to the evaluated population, and compute the exact fitness only for the best 20%.
   Compare evaluations-to-target with the plain GA over 10 seeds. *(Jupyter: NumPy + SciPy)*
5. Write a one-page brief for a TSO on whether to deploy an RL topology agent, covering distribution shift, rare
   contingencies, guarantees, the role of simulation checks and how ToOp could validate the agent's proposals.
   *(pen and paper)*
