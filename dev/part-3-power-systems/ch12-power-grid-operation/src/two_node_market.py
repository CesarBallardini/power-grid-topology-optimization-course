"""Nodal and zonal clearing of a two-node market with one line limit (Chapter 12, Exercises 1 and 2).

Node A has a cheap generator, node B an expensive one and the load. Under nodal pricing the line limit is part of
the market clearing; under zonal pricing it is ignored and the operator restores it afterwards by redispatch.
Costs are in euros per MWh, powers in MW, so every money figure is in euros per hour.
"""

from dataclasses import dataclass

import numpy as np

COST_A, COST_B = 20.0, 60.0  # marginal costs in euros per MWh
CAP_A, CAP_B = 1000.0, 1000.0  # generator capacities in MW
LOAD_B, LIMIT = 800.0, 500.0  # load at node B and line limit of the exercise, MW


@dataclass(frozen=True)
class Nodal:
    """Result of a nodal market clearing: dispatch, nodal prices, flow and congestion rent."""

    gen_a: float
    gen_b: float
    price_a: float
    price_b: float
    flow: float
    rent: float


@dataclass(frozen=True)
class Zonal:
    """Result of a zonal clearing followed by redispatch: dispatch and price, then the redispatch and its cost."""

    gen_a: float
    gen_b: float
    price: float
    redispatch_mw: float
    redispatch_cost: float


def nodal(limit: float = LIMIT, load: float = LOAD_B) -> Nodal:
    """Clear the two-node market with the line limit enforced.

    The cheap generator at A serves as much of the load as the line (and its own capacity) allows; the rest comes
    from B. The price at B is the cost of the last generator serving B, the price at A the cost of generator A
    while it has spare capacity. The congestion rent is the price difference times the flow.

    Args:
        limit: Transfer limit of the line A–B in MW.
        load: Load at node B in MW; it must not exceed the generation reachable at B.
    """
    flow = min(load, CAP_A, limit)
    gen_b = load - flow
    if gen_b > CAP_B:
        raise ValueError('the load at B cannot be served')
    price_b = COST_B if gen_b > 0 else COST_A
    price_a = COST_A if flow < CAP_A else price_b
    return Nodal(flow, gen_b, price_a, price_b, flow, (price_b - price_a) * flow)


def zonal(limit: float = LIMIT, load: float = LOAD_B) -> Zonal:
    """Clear the market as one zone, then redispatch to bring the line flow back within its limit.

    The zonal clearing stacks the generators by cost and ignores the line. The flow it implies is all of A's
    output; whatever exceeds the limit is moved from A to B, and the operator pays the cost difference for each
    MW moved (cost-based redispatch).
    """
    gen_a = min(load, CAP_A)
    gen_b = load - gen_a
    price = COST_B if gen_b > 0 else COST_A
    moved = max(0.0, gen_a - limit)
    return Zonal(gen_a, gen_b, price, moved, moved * (COST_B - COST_A))


def sweep(limits: np.ndarray, load: float = LOAD_B) -> tuple[np.ndarray, np.ndarray]:
    """Redispatch cost and congestion rent for each line limit, the two curves of Exercise 2."""
    cost = np.array([zonal(lim, load).redispatch_cost for lim in limits])
    rent = np.array([nodal(lim, load).rent for lim in limits])
    return cost, rent


def main() -> None:
    """Print the solution of Exercise 1 and the kink of Exercise 2."""
    n = nodal()
    print(f'nodal   : A {n.gen_a:.0f} MW at {n.price_a:.0f} EUR/MWh, B {n.gen_b:.0f} MW at {n.price_b:.0f} EUR/MWh')
    print(f'          flow {n.flow:.0f} MW, congestion rent {n.rent:,.0f} EUR/h')
    z = zonal()
    print(f'zonal   : A {z.gen_a:.0f} MW, B {z.gen_b:.0f} MW, one price {z.price:.0f} EUR/MWh')
    print(f'          redispatch {z.redispatch_mw:.0f} MW, cost {z.redispatch_cost:,.0f} EUR/h')
    limits = np.array([0.0, 200.0, 400.0, 500.0, 600.0, 799.0, 800.0, 1000.0])
    cost, rent = sweep(limits)
    print('limit MW        ' + ' '.join(f'{v:8.0f}' for v in limits))
    print('redispatch EUR/h' + ' '.join(f'{v:8.0f}' for v in cost))
    print('rent EUR/h      ' + ' '.join(f'{v:8.0f}' for v in rent))


if __name__ == '__main__':
    main()
