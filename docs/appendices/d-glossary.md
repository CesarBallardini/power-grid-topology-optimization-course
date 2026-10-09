# Appendix D — Glossary

Acronyms and technical terms used in the book, each with a short definition, the chapter where it is introduced,
and its Spanish equivalent. Hovering over an acronym anywhere in the book shows its expansion.

## Spanish terms and their sources

Every Spanish term was looked up in Spanish-language sources rather than translated. The tag after each term
names the source where it was found:

| Tag | Source |
|---|---|
| GRA | Grainger & Stevenson, *Análisis de sistemas de potencia* (McGraw-Hill, Latin American translation) |
| WEE | Weedy, *Sistemas eléctricos de gran potencia* (Reverté, Spain) |
| FRA | Fraile Mora, *Máquinas eléctricas* (Spanish original, Spain) |
| CHA | Chapman, *Máquinas eléctricas* (McGraw-Hill, Latin American translation) |
| ALE / HAY | Alexander & Sadiku, *Fundamentos de circuitos eléctricos*; Hayt et al., *Análisis de circuitos en ingeniería* |
| BUR / STR / CHP | Burden & Faires, *Análisis numérico*; Strang, *Álgebra lineal y sus aplicaciones*; Chapra & Canale, *Métodos numéricos para ingenieros* |
| TAH / HIL | Taha, *Investigación de operaciones*; Hillier & Lieberman, *Introducción a la investigación de operaciones* |
| SOGL | Regulation (EU) 2017/1485, official Spanish text: *directriz sobre la gestión de la red de transporte de electricidad* |
| CACM | Regulation (EU) 2015/1222, official Spanish text: *directriz sobre la asignación de capacidad y la gestión de las congestiones* |
| REE | Red Eléctrica: online glossary and operating procedures (*procedimientos de operación*) |
| FIUBA / UNLP | Course programs of Universidad de Buenos Aires and Universidad Nacional de La Plata |
| WIKI | Spanish Wikipedia (weaker source, used only when the others give nothing) |
| † | No Spanish source found: suggested translation; in practice the English acronym is usually kept |

Two regional conventions show up across the sources. Books from Spain (WEE, FRA) and the EU and REE texts use
*tensión*, *nudo*, *falta* and *relé*; the Latin American translations (GRA, CHA) use *voltaje*, *barra*, *falla*
and *relevador*. Argentine course programs mix them (*tensión* with *barra* and *falla*). Both are listed where
they differ.

## Acronyms

| Acronym | Expansion | Meaning | Spanish | Ch. |
|---|---|---|---|---|
| AC | Alternating current | Current and voltage that vary sinusoidally in time; AC power flow models the grid with complex voltages | *corriente alterna (CA)*; *flujo de potencia en CA* [GRA, FIUBA] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| BSDF | Bus split distribution factor | Sensitivity that updates the PTDF when a substation node is split into two busbars; a rank-1 update | *factor de distribución por separación de barras* † | [18](../part-3-power-systems/ch18-sensitivity-factors-2.md) |
| CACM | Capacity Allocation and Congestion Management | EU guideline (Regulation 2015/1222) on cross-zonal capacity calculation and market coupling | *directriz sobre la asignación de capacidad y la gestión de las congestiones* [CACM] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| CGM | Common grid model | Pan-European grid model merged from the individual grid models (IGM) of each TSO | *modelo de red común* [CACM] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| CGMES | Common Grid Model Exchange Standard | ENTSO-E profile of CIM used to exchange grid models; one of ToOp's input formats | CGMES (acronym kept) † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| CIM | Common Information Model | IEC standard data model for power systems on which CGMES is based | CIM (*modelo de información común*) † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| CNEC | Critical network element and contingency | A monitored element paired with a contingency, used in capacity calculation | *elemento crítico de la red y contingencia* [CACM, partial] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| CRAC | Contingency list, remedial actions and additional constraints | Input file of remedial action optimization (OpenRAO) | *lista de contingencias, medidas correctoras y restricciones adicionales* [SOGL, CACM, composite] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| DACF | Day-ahead congestion forecast | Grid model and security analysis for the next day, prepared by each TSO | *previsión de congestiones del día siguiente* † | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| DC | Direct current | Constant current; the "DC power flow" is a linear approximation of AC power flow, not a DC network | *corriente continua (CC)*; *flujo de potencia de cd* [GRA] | [16](../part-3-power-systems/ch16-dc-power-flow.md) |
| DSO | Distribution system operator | Company operating the distribution grid | *gestor de la red de distribución (GRD)* [SOGL]; *distribuidor* [REE, UNLP] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| EA | Evolutionary algorithm | Population-based search using selection, variation and replacement | *algoritmo evolutivo* [CHP] | [24](../part-4-optimization/ch24-evolutionary-algorithms.md) |
| ECDF | Empirical cumulative distribution function | Fraction of runs reaching a result, used to compare stochastic optimizers | *función de distribución empírica* † | [26](../part-4-optimization/ch26-experimental-methodology.md) |
| ENTSO-E | European Network of Transmission System Operators for Electricity | Association of European TSOs; publishes grid codes, data formats and processes | *REGRT de Electricidad* [SOGL] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| GA | Genetic algorithm | Evolutionary algorithm with a genotype, mutation and crossover; ToOp's DC optimizer is one | *algoritmo genético (AG)* [TAH, HIL] | [24](../part-4-optimization/ch24-evolutionary-algorithms.md) |
| GLODF | Generalized line outage distribution factor | LODF for several simultaneous outages; in ToOp's code this is the MODF | *factor de distribución generalizado por salida de líneas* † | [18](../part-3-power-systems/ch18-sensitivity-factors-2.md) |
| GLSK | Generation and load shift key | How a change of a zone's net position is distributed over its generators and loads | *pauta de variación de la generación y la carga* [CACM, partial] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| GNN | Graph neural network | Neural network operating on graph-structured data, e.g. as a power flow surrogate | *red neuronal de grafos* [WIKI] | [29](../part-4-optimization/ch29-learning-based-control.md) |
| GPU | Graphics processing unit | Massively parallel processor on which ToOp evaluates batches of topologies | *unidad de procesamiento gráfico (GPU)* [WIKI] | [31](../part-5-computing/ch31-jax-gpu-fundamentals.md) |
| HVDC | High-voltage direct current | Transmission link with converter stations; modeled by ToOp as injections | *corriente continua de alta tensión (HVDC)* [SOGL] | [11](../part-2-circuits/ch11-transformers-psts-hvdc.md) |
| IDCF | Intraday congestion forecast | Intraday update of the DACF process | *previsión de congestiones intradiaria* † | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| IGD | Inverted generational distance | Distance from a reference front to an approximation; IGD+ is its Pareto-compliant variant | *distancia generacional invertida* † | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| IIDM / XIIDM | (Extended) iTesla Internal Data Model | PowSyBl's grid data model and its XML serialization; ToOp's processed grid snapshot | IIDM / XIIDM (acronym kept) † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| JIT | Just-in-time compilation | Compiling a function when first called with given input shapes; `jax.jit` | *compilación en tiempo de ejecución (JIT)* [WIKI] | [31](../part-5-computing/ch31-jax-gpu-fundamentals.md) |
| KCL / KVL | Kirchhoff's current law / voltage law | Currents into a node sum to zero; voltages around a loop sum to zero | *ley de corrientes de Kirchhoff (LCK)* / *ley de tensiones de Kirchhoff (LTK)* [ALE]; *ley de voltajes (LVK)* [HAY] | [8](../part-2-circuits/ch08-dc-circuits.md) |
| LCC / VSC | Line-commutated converter / voltage-source converter | The two HVDC converter technologies | *convertidor conmutado por línea* / *convertidor en fuente de tensión* † | [11](../part-2-circuits/ch11-transformers-psts-hvdc.md) |
| LMP | Locational marginal price | Price of electricity at a node, including congestion and losses | *precio marginal nodal* †; *factores de nodo* [UNLP] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| LODF | Line outage distribution factor | Fraction of a branch's pre-outage flow that moves to another branch when it trips | *factor de distribución por salida de línea* † | [17](../part-3-power-systems/ch17-sensitivity-factors-1.md) |
| LP | Linear programming | Optimization with a linear objective and linear constraints | *programación lineal (PL)* [TAH, HIL, FIUBA] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| LU | LU factorization | Writing a matrix as lower times upper triangular to solve linear systems | *factorización LU* [BUR]; *descomposición LU* [CHP] | [4](../part-1-math/ch04-numerical-linear-algebra.md) |
| MAP-Elites | Multi-dimensional Archive of Phenotypic Elites | Quality-diversity algorithm that keeps the best solution per descriptor cell; ToOp's DC search | MAP-Elites (name kept) | [25](../part-4-optimization/ch25-quality-diversity.md) |
| MDP | Markov decision process | Formal model of sequential decisions used in reinforcement learning | *proceso de decisión markoviano* [HIL] | [29](../part-4-optimization/ch29-learning-based-control.md) |
| MILP | Mixed-integer linear programming | LP with some integer variables; exact formulation of transmission switching | *programación lineal entera mixta* [FIUBA]; *programación entera mixta* [HIL] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| MINLP | Mixed-integer nonlinear programming | Nonlinear program with integer variables, e.g. AC transmission switching | *programación entera mixta no lineal* [HIL, partial] | [28](../part-4-optimization/ch28-topology-optimization.md) |
| MODF | Multi-outage distribution factor | Generalized LODF for k simultaneous outages from a k×k solve; used for disconnections, three-winding transformers and busbar outages | *factor de distribución por salidas múltiples* † | [18](../part-3-power-systems/ch18-sensitivity-factors-2.md) |
| MOEA | Multi-objective evolutionary algorithm | Evolutionary algorithm that approximates a Pareto front | *algoritmo evolutivo multiobjetivo* [WIKI] | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| MOME | Multi-Objective MAP-Elites | Quality-diversity algorithm that keeps a Pareto front in every cell | MOME (acronym kept) | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| N-1 | N-1 criterion | The grid must stay within limits after the loss of any single element | *criterio N-1* [SOGL, REE]; *fallo simple* [REE] | [19](../part-3-power-systems/ch19-operational-security.md) |
| NP-hard | Nondeterministic polynomial-time hard | Class of problems at least as hard as every problem in NP; transmission switching is one | *NP-difícil* [WIKI] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| NSGA-II | Non-dominated Sorting Genetic Algorithm II | Classic multi-objective EA using non-dominated sorting and crowding distance | NSGA-II (name kept) | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| OLTC | On-load tap changer | Mechanism that changes transformer taps while energized | *cambiador de tomas en carga* [SOGL]; *cambiador de derivación bajo carga* [GRA] | [11](../part-2-circuits/ch11-transformers-psts-hvdc.md) |
| OPF | Optimal power flow | Power flow that also optimizes dispatch under network constraints | *flujo de potencia óptimo (FPO)* [GRA] | [28](../part-4-optimization/ch28-topology-optimization.md) |
| OTDF | Outage transfer distribution factor | PTDF of a branch after a given outage | *factor de distribución de transferencia ante contingencia* † | [17](../part-3-power-systems/ch17-sensitivity-factors-1.md) |
| OTS | Optimal transmission switching | Choosing which lines to open to minimize cost or overloads | *conmutación óptima de líneas de transmisión* † | [28](../part-4-optimization/ch28-topology-optimization.md) |
| PATL / TATL | Permanent / temporary admissible transmission loading | Continuous and time-limited current ratings of a branch | *carga permanente admisible* / *sobrecarga transitoria admisible* [SOGL, REE, partial] | [19](../part-3-power-systems/ch19-operational-security.md) |
| PEDF | Power exchange distribution factor | Flow change for a transfer between two nodes; used inside the BSDF derivation | *factor de distribución de intercambio de potencia* † | [18](../part-3-power-systems/ch18-sensitivity-factors-2.md) |
| PQ / PV bus | Load bus / generator (voltage-controlled) bus | Bus types in power flow: P and Q given, or P and voltage magnitude given | *nudo PQ (de carga)* [WEE]; *barra de carga* [GRA] / *nudo PV (de generación)* [WEE]; *barra de voltaje controlado* [GRA] | [15](../part-3-power-systems/ch15-ac-power-flow.md) |
| PSDF | Phase shift distribution factor | Branch flow change per degree of a phase-shifting transformer's angle | *factor de distribución del desfasador* † | [17](../part-3-power-systems/ch17-sensitivity-factors-1.md) |
| PST | Phase-shifting transformer | Transformer that shifts the voltage angle to steer active power; its taps are a ToOp action | *transformador desfasador* [SOGL, REE]; *transformador de defasamiento* [GRA] | [11](../part-2-circuits/ch11-transformers-psts-hvdc.md) |
| PTDF | Power transfer distribution factor | Change of flow on a branch per unit of power injected at a node and withdrawn at the slack | *factor de distribución de la transferencia de energía* [CACM] | [17](../part-3-power-systems/ch17-sensitivity-factors-1.md) |
| p.u. | Per unit | Quantity expressed as a fraction of a chosen base value | *por unidad (p.u.)* [GRA, WEE, FRA, FIUBA] | [10](../part-2-circuits/ch10-three-phase-per-unit.md) |
| QD | Quality-diversity | Optimization that seeks many high-performing, behaviorally different solutions | *calidad-diversidad (QD)* † | [25](../part-4-optimization/ch25-quality-diversity.md) |
| RAO | Remedial action optimization | Selecting remedial actions that restore security at least cost (e.g. OpenRAO) | *optimización de medidas correctoras* [CACM, partial] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| RL | Reinforcement learning | Learning a policy from rewards by interacting with an environment | *aprendizaje por refuerzo* [WIKI] | [29](../part-4-optimization/ch29-learning-based-control.md) |
| SCOPF | Security-constrained optimal power flow | OPF that also enforces N-1 constraints | *flujo de potencia óptimo con restricciones de seguridad* † | [28](../part-4-optimization/ch28-topology-optimization.md) |
| SDP / SOCP | Semidefinite / second-order cone programming | Convex problem classes used to relax AC-OPF | *programación semidefinida* [WIKI] / *programación cónica de segundo orden* † | [28](../part-4-optimization/ch28-topology-optimization.md) |
| SE | State estimation | Computing the most likely grid state from redundant, noisy measurements | *estimación de estado* [SOGL, GRA] | [22](../part-3-power-systems/ch22-import-preprocessing.md) |
| SLD | Single-line diagram | One-line schematic of a substation or network | *esquema unifilar* [SOGL, FRA]; *diagrama unifilar* [GRA, FIUBA] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| SO GL | System Operation Guideline | EU guideline (Regulation 2017/1485) on operational security, contingencies and remedial actions | *directriz sobre la gestión de la red de transporte de electricidad* [SOGL] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| SpPS | Special protection scheme | Automatic rule-based action triggered by grid conditions; simulated in ToOp's cascade module | *sistema especial de protección* [SOGL]; *automatismo de teledisparo* [REE] | [20](../part-3-power-systems/ch20-protection-short-circuit.md) |
| SVC | Static var compensator | Thyristor-controlled reactive power device | *compensador estático de potencia reactiva* [WEE, FIUBA] | [11](../part-2-circuits/ch11-transformers-psts-hvdc.md) |
| SVD | Singular value decomposition | Factorization $A = U \Sigma V^\top$; gives rank, pseudoinverse and conditioning | *descomposición en valores singulares* [BUR, CHP] | [2](../part-1-math/ch02-linear-algebra.md) |
| TSO | Transmission system operator | Company operating the high-voltage grid (e.g. Elia, 50Hertz) | *gestor de la red de transporte (GRT)* [SOGL]; *operador del sistema* [REE] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| UCTE-DEF | UCTE data exchange format | Fixed-width text format for European grid models; one of ToOp's input formats | UCTE-DEF (acronym kept) † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| WLS | Weighted least squares | Estimation method minimizing weighted squared measurement residuals; standard for state estimation | *mínimos cuadrados ponderados* [WIKI; GRA, partial] | [22](../part-3-power-systems/ch22-import-preprocessing.md) |
| XLA | Accelerated Linear Algebra | Compiler that JAX uses to generate CPU and GPU code | XLA (name kept) | [31](../part-5-computing/ch31-jax-gpu-fundamentals.md) |

## Terms

### Power systems and circuits

| Term | Meaning | Spanish | Ch. |
|---|---|---|---|
| Active power | Real power $P$, the average power delivered, in W or MW | *potencia activa* [SOGL, FRA]; *potencia real* [GRA, CHA] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| Admittance / impedance | $Y = G + jB$ and $Z = R + jX = 1/Y$ | *admitancia* / *impedancia* [all textbooks] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| Apparent power | $\lvert S \rvert = \sqrt{P^2 + Q^2}$, in VA or MVA | *potencia aparente* [FRA, CHA, ALE] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| Bad data | Measurements with gross errors that state estimation must detect | *datos erróneos* [GRA] | [22](../part-3-power-systems/ch22-import-preprocessing.md) |
| Bay | Group of switchgear connecting one line, transformer or load to the busbars | *posición* [REE] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Branch | In ToOp, any line or transformer between two nodes | *rama* [GRA, partial] | [14](../part-3-power-systems/ch14-component-models.md) |
| Breaker-and-a-half | Substation layout with three breakers for every two circuits | *interruptor y medio* [REE] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Bus admittance matrix | $Y_{\text{bus}}$, relating nodal current injections to nodal voltages | *matriz de admitancias de nudos* [WEE]; *matriz de admitancias de barra* [GRA] | [14](../part-3-power-systems/ch14-component-models.md) |
| Bus-branch model | Grid represented as electrical nodes and branches, without switches | *modelo barra-rama* † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Busbar | Conductor in a substation to which circuits connect | *barra (barras colectoras)* [SOGL, REE]; *embarrado* [WIKI] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Busbar coupler | Breaker that connects two busbars of the same substation | *acoplamiento de barras* [REE]; *interruptor acoplador* [SOGL] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Busbar outage | Loss of a whole busbar and every element connected to it; a multi-outage in ToOp | *indisponibilidad de barra* † | [18](../part-3-power-systems/ch18-sensitivity-factors-2.md) |
| Busbar reassignment | Moving a line or injection from one busbar to another within a substation | *cambio de barra (reasignación)* † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Busbar splitting | Opening couplers so a substation runs as two electrical nodes | *separación de barras* [REE] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Capacity calculation | Computing cross-zonal transmission capacity for the market | *cálculo de capacidad* [CACM] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| Cascading failure | Sequence of trips where each outage overloads further elements | *fallo en cascada* (ES) / *falla en cascada* (LatAm) † | [20](../part-3-power-systems/ch20-protection-short-circuit.md) |
| Circuit breaker | Switch able to interrupt load and fault currents | *interruptor* [SOGL, REE, GRA]; *disyuntor* [WEE] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Complex power | $S = V I^* = P + jQ$ | *potencia compleja* [GRA, FRA, WEE] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| Congestion | State in which a network element would exceed its limit | *congestión* [CACM, REE] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| Congestion management | Measures that keep flows within limits (redispatch, topology changes, countertrading) | *gestión de las congestiones* [CACM] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| Contingency | Possible outage of one or more elements considered in security analysis | *contingencia* [CACM, SOGL, GRA] | [19](../part-3-power-systems/ch19-operational-security.md) |
| Contingency analysis | Computing the grid state after each contingency in a list | *análisis de contingencias* [SOGL, GRA] | [17](../part-3-power-systems/ch17-sensitivity-factors-1.md) |
| Countertrading | Cross-zonal exchange initiated by TSOs to relieve congestion | *intercambio compensatorio* [SOGL, CACM] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| Critical branch | In ToOp, a branch above its limit; counted by the critical branch count metric | *elemento crítico de la red* [CACM] | [19](../part-3-power-systems/ch19-operational-security.md) |
| Cross-coupler flow | Power that would flow through an open busbar coupler if it were closed; ToOp's KCL imbalance at busbar A | *flujo a través del acoplamiento* † | [18](../part-3-power-systems/ch18-sensitivity-factors-2.md) |
| Disconnector | Switch that isolates equipment but cannot interrupt load current | *seccionador* [REE, WEE, FRA] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Distance protection | Relay that trips when the apparent impedance falls inside a zone of the R-X plane | *protección de distancia* [WIKI]; *relé de distancia* [WEE] | [20](../part-3-power-systems/ch20-protection-short-circuit.md) |
| Distributed slack | Power imbalance shared among several generators instead of a single slack bus | *nudo de compensación distribuido* † | [15](../part-3-power-systems/ch15-ac-power-flow.md) |
| Double busbar | Substation layout with two busbars and a coupler | *doble barra* [REE] | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Double limits | In ToOp, operational bands (90–100 % of rating) used by the `*_limited` metrics | *límites dobles* † | [19](../part-3-power-systems/ch19-operational-security.md) |
| Fast decoupled load flow | Power flow method using constant, decoupled P–θ and Q–V matrices | *flujo de potencia desacoplado rápido* [GRA, partial] | [15](../part-3-power-systems/ch15-ac-power-flow.md) |
| Fault level | Short-circuit power at a node | *potencia de cortocircuito* [FRA]; *nivel de cortocircuito* [WEE] | [20](../part-3-power-systems/ch20-protection-short-circuit.md) |
| Flat start | Power flow initialization with all voltages at 1 p.u. and 0° | *inicio plano* [GRA] | [15](../part-3-power-systems/ch15-ac-power-flow.md) |
| Flow-based market coupling | Market clearing that respects grid constraints expressed with zonal PTDFs | *acoplamiento de mercados basado en flujos* [CACM, REE, composite] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| Injection | In ToOp, any generator, load, battery or converter station connected to a node | *inyección* † | [16](../part-3-power-systems/ch16-dc-power-flow.md) |
| Island / islanding | Part of the grid disconnected from the rest after an outage | *isla*; *funcionamiento en isla* [REE] | [17](../part-3-power-systems/ch17-sensitivity-factors-1.md) |
| Line switching | Opening or closing a line to change power flows | *maniobra de líneas* [FRA, partial] | [28](../part-4-optimization/ch28-topology-optimization.md) |
| Line-to-line voltage | Voltage between two phases, $\sqrt{3}$ times the phase voltage | *tensión de línea (tensión compuesta)* [FRA, WEE]; *voltaje línea a línea* [GRA] | [10](../part-2-circuits/ch10-three-phase-per-unit.md) |
| Load-break switch | Switch that can interrupt load current but not fault current | *interruptor-seccionador* † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Node-breaker model | Grid represented with physical nodes, busbar sections and switches | *modelo nodo-interruptor* † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Off-nominal turns ratio | Transformer ratio different from the ratio of the base voltages | *relación de transformación no nominal* [GRA, adapted] | [11](../part-2-circuits/ch11-transformers-psts-hvdc.md) |
| Outage | Element out of service, forced or planned | *indisponibilidad* [SOGL]; *salida (de línea)* [GRA]; *descargo* (planned) [REE] | [17](../part-3-power-systems/ch17-sensitivity-factors-1.md) |
| Overload | Flow above an element's admissible limit | *sobrecarga* [SOGL, REE] | [19](../part-3-power-systems/ch19-operational-security.md) |
| Overload energy | ToOp's main metric: sum over branches of flow above the limit, worst contingency per branch | *energía de sobrecarga* † | [19](../part-3-power-systems/ch19-operational-security.md) |
| Per-unit system | Normalizing quantities by base values so transformers disappear from equations | *sistema por unidad* [CHA]; *valores por unidad* [FRA] | [10](../part-2-circuits/ch10-three-phase-per-unit.md) |
| Phasor | Complex number representing the amplitude and phase of a sinusoid | *fasor* [all textbooks] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| Power factor | $\cos \varphi = P / \lvert S \rvert$ | *factor de potencia* [all textbooks] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| Reactive power | $Q$, power oscillating between source and fields, in var | *potencia reactiva* [SOGL, all textbooks] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| Redispatch | Changing generator outputs to relieve congestion; the costly alternative to topology changes | *redespacho* [SOGL, REE] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| Relay | Device that detects abnormal conditions and commands breakers | *relé* [WEE, FRA, REE]; *relevador* [GRA] | [20](../part-3-power-systems/ch20-protection-short-circuit.md) |
| Relevant substation | In ToOp, a substation eligible for splitting under the preprocessing rules | *subestación relevante* † | [22](../part-3-power-systems/ch22-import-preprocessing.md) |
| Remedial action | Action that restores or preserves operational security; preventive (before) or curative (after a contingency) | *medida correctora* [CACM]; *acción preventiva* / *acción correctiva poscontingencia* [REE] | [12](../part-3-power-systems/ch12-power-grid-operation.md) |
| Short-circuit current | Current flowing during a fault | *corriente de cortocircuito* [SOGL, GRA, WEE, FRA] | [20](../part-3-power-systems/ch20-protection-short-circuit.md) |
| Slack bus | Reference bus whose injection balances the power flow and whose angle is zero | *nudo de referencia (flotante)* [WEE]; *barra de compensación* [GRA] | [15](../part-3-power-systems/ch15-ac-power-flow.md) |
| Susceptance / reactance | Imaginary parts of admittance and impedance; DC power flow uses $b = 1/x$ | *susceptancia* / *reactancia* [all textbooks] | [9](../part-2-circuits/ch09-ac-steady-state.md) |
| Switching distance | In ToOp, the number of switch operations needed to realize a topology (Hamming distance) | *distancia de maniobra* † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Symmetrical components | Decomposition of unbalanced three-phase quantities into positive, negative and zero sequence | *componentes simétricas* [GRA, WEE, FRA, FIUBA] | [10](../part-2-circuits/ch10-three-phase-per-unit.md) |
| Synchro-check | Verification that voltage angle and magnitude differences are small before closing a breaker | *verificación de sincronismo* † | [20](../part-3-power-systems/ch20-protection-short-circuit.md) |
| Tap | Winding connection point that changes a transformer's ratio or phase shift | *toma* [SOGL, FRA]; *derivación (tap)* [GRA, CHA] | [11](../part-2-circuits/ch11-transformers-psts-hvdc.md) |
| Thermal limit | Maximum current an element can carry without overheating | *límite térmico* [CACM]; *capacidad térmica* [SOGL, REE] | [19](../part-3-power-systems/ch19-operational-security.md) |
| Three-winding transformer | Transformer with three windings, modeled as three branches around a star node | *transformador de tres devanados* [GRA]; *de tres arrollamientos* [WEE] | [11](../part-2-circuits/ch11-transformers-psts-hvdc.md) |
| Tie line | Line connecting two control areas or TSO grids | *línea de interconexión* [SOGL]; *línea de enlace* [WEE] | [14](../part-3-power-systems/ch14-component-models.md) |
| Topological remedial action | Remedial action that changes the grid topology (switching, busbar splitting) | *medida correctora topológica*; *cambios topológicos* [SOGL, REE] | [0](../part-0-orientation/ch00-orientation-and-setup.md) |
| Topology | Connectivity of the grid given by switch states; in ToOp, a set of station actions, disconnections and PST setpoints | *topología de la red* [SOGL, CACM] | [0](../part-0-orientation/ch00-orientation-and-setup.md) |
| Topo-vect | Grid2Op's vector giving the busbar (1 or 2) of every element in each substation | topo-vect (term kept) † | [13](../part-3-power-systems/ch13-substations-grid-models.md) |
| Transmission capacity | Maximum power that can be transferred between areas | *capacidad de transporte* [WEE]; *capacidad de intercambio* [CACM] | [21](../part-3-power-systems/ch21-coordinated-security-rao.md) |
| Voltage stability | Ability to keep acceptable voltages after disturbances | *estabilidad de tensión* [SOGL, WEE, FIUBA] | [20](../part-3-power-systems/ch20-protection-short-circuit.md) |

### Mathematics and numerical methods

| Term | Meaning | Spanish | Ch. |
|---|---|---|---|
| Articulation point | Vertex whose removal disconnects a graph (cut vertex) | *vértice de corte (punto de articulación)* [WIKI] | [3](../part-1-math/ch03-graph-theory.md) |
| Bridge | Edge whose removal disconnects a graph (cut edge); its LODF denominator is zero | *arista de corte (puente)* [WIKI] | [3](../part-1-math/ch03-graph-theory.md) |
| Condition number | $\kappa(A) = \lVert A \rVert \lVert A^{-1} \rVert$, the sensitivity of a solution to perturbations | *número de condición* [BUR, CHP, STR] | [4](../part-1-math/ch04-numerical-linear-algebra.md) |
| Connected component | Maximal set of vertices joined by paths; an island in a grid | *componente conexa* [WIKI] | [3](../part-1-math/ch03-graph-theory.md) |
| Convergence | An iteration approaching the solution within a tolerance | *convergencia* [GRA, BUR, CHP] | [6](../part-1-math/ch06-newton-raphson.md) |
| Incidence matrix | Branch-by-node matrix with +1 and −1 at each branch's end nodes | *matriz de incidencia* [STR, GRA] | [2](../part-1-math/ch02-linear-algebra.md) |
| Jacobian | Matrix of partial derivatives of a vector function; the core of Newton–Raphson | *matriz jacobiana (jacobiano)* [GRA, WEE, BUR] | [6](../part-1-math/ch06-newton-raphson.md) |
| Laplacian matrix | $L = A^\top \operatorname{diag}(w) A$; the DC susceptance matrix is a weighted Laplacian | *matriz laplaciana* [WIKI] | [3](../part-1-math/ch03-graph-theory.md) |
| Newton–Raphson method | Iterative method solving $f(x) = 0$ with Jacobian linearizations | *método de Newton-Raphson* [GRA, WEE, CHP, FIUBA] | [6](../part-1-math/ch06-newton-raphson.md) |
| Rank-one update | Change of a matrix by an outer product $u v^\top$ | *actualización de rango uno* [STR, partial] | [4](../part-1-math/ch04-numerical-linear-algebra.md) |
| Sherman–Morrison–Woodbury formula | Formula for the inverse of a matrix after a low-rank update (matrix inversion lemma) | *fórmula de Sherman-Morrison-Woodbury* [BUR]; *lema de inversión de matrices* | [4](../part-1-math/ch04-numerical-linear-algebra.md) |
| Spanning tree | Subgraph that connects all vertices without cycles | *árbol de expansión* [TAH, HIL]; *árbol generador* [STR] | [3](../part-1-math/ch03-graph-theory.md) |
| Sparse matrix | Matrix with mostly zero entries, stored compactly | *matriz dispersa* [BUR, CHP]; *matriz rala* [STR] | [4](../part-1-math/ch04-numerical-linear-algebra.md) |

### Optimization and learning

| Term | Meaning | Spanish | Ch. |
|---|---|---|---|
| Branch and bound | Exact method for integer programs that explores and prunes a tree of subproblems | *ramificación y acotamiento* [TAH, HIL]; *ramificación y poda* [WIKI] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| Constraint | Condition a feasible solution must satisfy | *restricción* [HIL, TAH]; *condición de vínculo* [FIUBA] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| Crossover | Variation operator combining two parents | *cruce* [TAH, HIL] | [24](../part-4-optimization/ch24-evolutionary-algorithms.md) |
| Decision variable | Quantity chosen by the optimizer | *variable de decisión* [HIL, TAH] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| Descriptor | Feature that places a solution in a MAP-Elites cell; in ToOp, e.g. split substations and switching distance | *descriptor de comportamiento* † | [25](../part-4-optimization/ch25-quality-diversity.md) |
| Dominance (Pareto) | Solution *a* dominates *b* if it is no worse in every objective and better in at least one | *dominancia de Pareto* [WIKI] | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| Duality | Pairing of an optimization problem with a dual whose optimum bounds it | *dualidad*; *problema dual* [HIL, TAH, FIUBA] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| Elitism | Keeping the best solutions unchanged into the next generation | *elitismo* [WIKI] | [24](../part-4-optimization/ch24-evolutionary-algorithms.md) |
| Epsilon-constraint method | Optimizing one objective while bounding the others | *método de la ε-restricción* † | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| Feasible region | Set of all solutions satisfying the constraints | *región factible* [HIL, STR]; *espacio factible* [TAH] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| Fitness | Score used for selection; ToOp's DC fitness is a negated weighted sum of metrics | *función de aptitud* [TAH] | [24](../part-4-optimization/ch24-evolutionary-algorithms.md) |
| Heuristic / metaheuristic | Rule of thumb / general strategy for guiding approximate search | *heurística* / *metaheurística* [TAH, HIL, FIUBA] | [24](../part-4-optimization/ch24-evolutionary-algorithms.md) |
| Hyperparameter | Setting of an algorithm chosen before running it | *hiperparámetro* [WIKI] | [26](../part-4-optimization/ch26-experimental-methodology.md) |
| Hypervolume indicator | Volume of objective space dominated by a front, bounded by a reference point | *indicador de hipervolumen* † | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| Mutation | Variation operator that randomly changes a solution | *mutación* [TAH, HIL] | [24](../part-4-optimization/ch24-evolutionary-algorithms.md) |
| Non-dominated sorting | Ranking a population into successive Pareto fronts | *ordenamiento no dominado* † | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| Objective function | Function the optimizer minimizes or maximizes | *función objetivo* [HIL, TAH, CHP] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| Pareto front | Set of objective vectors of non-dominated solutions | *frente de Pareto* [WIKI] | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |
| Repertoire | Archive of elite solutions indexed by descriptor cells in MAP-Elites | *repertorio* † | [25](../part-4-optimization/ch25-quality-diversity.md) |
| Reproducibility | Obtaining the same results when an experiment is repeated | *reproducibilidad* [WIKI] | [26](../part-4-optimization/ch26-experimental-methodology.md) |
| Shadow price | Change of the optimal objective per unit relaxation of a constraint | *precio sombra* [HIL]; *precio dual* [TAH]; *valor marginal* [FIUBA] | [23](../part-4-optimization/ch23-optimization-modeling.md) |
| Simulated annealing | Local search that accepts worse moves with a decreasing probability | *recocido simulado* [TAH]; *templado simulado* [HIL] | [24](../part-4-optimization/ch24-evolutionary-algorithms.md) |
| Surrogate model | Cheap approximation of an expensive evaluation | *modelo sustituto* [WIKI] | [29](../part-4-optimization/ch29-learning-based-control.md) |
| Weighted-sum scalarization | Combining objectives into one by a weighted sum; misses unsupported Pareto points | *escalarización por suma ponderada* †; *método de pesos* [TAH] | [27](../part-4-optimization/ch27-pareto-multi-objective.md) |

### Computing

| Term | Meaning | Spanish | Ch. |
|---|---|---|---|
| Automatic differentiation | Computing exact derivatives of programs by applying the chain rule to elementary operations | *diferenciación automática* [WIKI] | [29](../part-4-optimization/ch29-learning-based-control.md) |
| Benchmark | Standard problem set or procedure for comparing performance | *banco de pruebas (benchmark)* [WIKI] | [26](../part-4-optimization/ch26-experimental-methodology.md) |
| Processed grid folder | ToOp's on-disk contract between stages: grid snapshot, masks, static information, action set | *carpeta de red preprocesada* † | [22](../part-3-power-systems/ch22-import-preprocessing.md) |
| Pytree | Nested container of arrays that JAX transformations traverse | pytree (term kept) † | [31](../part-5-computing/ch31-jax-gpu-fundamentals.md) |
| Static vs dynamic information | In ToOp, compile-time grid structure (`SolverConfig`) vs traced, batch-varying arrays (`DynamicInformation`) | *información estática / dinámica* † | [32](../part-5-computing/ch32-jax-in-toop.md) |
| Tracing | JAX recording the operations of a function on abstract values to compile it | *trazado* † | [31](../part-5-computing/ch31-jax-gpu-fundamentals.md) |
| Vectorization | Applying an operation to a whole batch at once; `jax.vmap` | *vectorización* [WIKI, CHP] | [31](../part-5-computing/ch31-jax-gpu-fundamentals.md) |
